"""
Tests for password hashing and JWTs, including backward compatibility.

The migration off python-jose + passlib is only safe if data already in the
database keeps working. Two compatibility tests below mint artefacts with
the OLD libraries and verify them with the NEW code; they skip if the old
libraries are not installed, which is the normal state once `legacy` is
uninstalled.
"""

from __future__ import annotations

from datetime import UTC, timedelta

import jwt
import pytest

from app.security import (
    ALGORITHM,
    BCRYPT_MAX_PASSWORD_BYTES,
    PasswordTooLongError,
    constant_time_compare,
    create_access_token,
    decode_access_token,
    hash_password,
    needs_rehash,
    verify_password,
)

PASSWORD = "correct horse battery staple"


# --- passwords -------------------------------------------------------------

def test_hash_and_verify_roundtrip():
    h = hash_password(PASSWORD)
    assert verify_password(PASSWORD, h)
    assert not verify_password("wrong password entirely", h)


def test_hashes_are_salted_and_therefore_differ():
    assert hash_password(PASSWORD) != hash_password(PASSWORD)


def test_hash_uses_the_modern_bcrypt_prefix_and_cost():
    h = hash_password(PASSWORD)
    assert h.startswith("$2b$")
    assert int(h.split("$")[2]) >= 12


def test_overlong_password_is_rejected_not_truncated():
    """
    Verified against the pinned v1 stack: bcrypt hashes a 100-character
    password happily and the result matches its first 72 bytes, so such a
    user could authenticate with a prefix. Now an explicit error.
    """
    too_long = "a" * (BCRYPT_MAX_PASSWORD_BYTES + 1)
    with pytest.raises(PasswordTooLongError, match="72"):
        hash_password(too_long)


def test_a_password_at_exactly_the_limit_is_accepted():
    at_limit = "b" * BCRYPT_MAX_PASSWORD_BYTES
    assert verify_password(at_limit, hash_password(at_limit))


def test_multibyte_passwords_are_measured_in_bytes_not_characters():
    """A 3-byte character means the limit is 24 such characters, not 72."""
    over = "é" * 40  # 80 bytes
    with pytest.raises(PasswordTooLongError):
        hash_password(over)
    under = "é" * 30  # 60 bytes
    assert verify_password(under, hash_password(under))


def test_verify_tolerates_an_overlong_legacy_password():
    """
    A user whose password predates the length check must still log in: we
    compare against the same first 72 bytes bcrypt originally hashed.
    """
    import bcrypt as _b

    legacy_plain = "z" * 100
    stored = _b.hashpw(legacy_plain.encode()[:72], _b.gensalt(rounds=4)).decode()
    assert verify_password(legacy_plain, stored)


def test_verify_returns_false_for_a_corrupt_hash_instead_of_raising():
    """A bad row is an auth failure, not a 500."""
    for bad in ("", "not-a-hash", "$2b$broken", "$"):
        assert verify_password(PASSWORD, bad) is False


def test_needs_rehash_flags_weaker_work_factors():
    import bcrypt as _b

    weak = _b.hashpw(PASSWORD.encode(), _b.gensalt(rounds=4)).decode()
    assert needs_rehash(weak)
    assert not needs_rehash(hash_password(PASSWORD))
    assert needs_rehash("garbage")


def test_verifies_a_preexisting_passlib_hash():
    """Hashes written by the v1 passlib code must still verify."""
    passlib_ctx = pytest.importorskip(
        "passlib.context", reason="legacy extra not installed"
    )
    ctx = passlib_ctx.CryptContext(schemes=["bcrypt"], deprecated="auto")
    assert verify_password(PASSWORD, ctx.hash(PASSWORD))


# --- tokens ----------------------------------------------------------------

def test_token_roundtrip_carries_the_claims():
    token = create_access_token("user-123", "a@b.com")
    claims = decode_access_token(token)
    assert claims["sub"] == "user-123"
    assert claims["email"] == "a@b.com"
    assert "exp" in claims and "iat" in claims


def test_expired_token_is_rejected():
    from fastapi import HTTPException

    token = create_access_token("u", "a@b.com", expires_delta=timedelta(seconds=-10))
    with pytest.raises(HTTPException) as exc:
        decode_access_token(token)
    assert exc.value.status_code == 401
    assert "expired" in exc.value.detail.lower()


def test_tampered_token_is_rejected():
    from fastapi import HTTPException

    token = create_access_token("u", "a@b.com")
    head, payload, sig = token.split(".")
    tampered = f"{head}.{payload}.{'A' * len(sig)}"
    with pytest.raises(HTTPException):
        decode_access_token(tampered)


def test_alg_none_token_is_rejected():
    """
    The classic JWT confusion attack. Passing `algorithms=[...]` explicitly
    is what prevents it; this test fails if someone removes that argument.
    """
    from fastapi import HTTPException

    forged = jwt.encode({"sub": "admin", "exp": 9999999999}, key="", algorithm="none")
    with pytest.raises(HTTPException):
        decode_access_token(forged)


def test_token_signed_with_another_key_is_rejected():
    from fastapi import HTTPException

    forged = jwt.encode(
        {"sub": "admin", "exp": 9999999999}, "a-different-secret", algorithm=ALGORITHM
    )
    with pytest.raises(HTTPException):
        decode_access_token(forged)


def test_token_without_required_claims_is_rejected():
    from fastapi import HTTPException

    from app.security import _check_secret

    no_sub = jwt.encode({"exp": 9999999999}, _check_secret(), algorithm=ALGORITHM)
    with pytest.raises(HTTPException):
        decode_access_token(no_sub)


def test_verifies_a_token_minted_by_python_jose():
    """
    Tokens issued by the v1 implementation must keep working, or every
    logged-in user is signed out by the deploy.
    """
    jose_jwt = pytest.importorskip("jose.jwt", reason="legacy extra not installed")
    from datetime import datetime

    from app.security import _check_secret

    token = jose_jwt.encode(
        {
            "sub": "legacy-user",
            "email": "old@example.com",
            "exp": datetime.now(UTC) + timedelta(hours=1),
            "iat": datetime.now(UTC),
        },
        _check_secret(),
        algorithm=ALGORITHM,
    )
    assert decode_access_token(token)["sub"] == "legacy-user"


# --- misc ------------------------------------------------------------------

def test_constant_time_compare():
    assert constant_time_compare("abc", "abc")
    assert not constant_time_compare("abc", "abd")
    assert not constant_time_compare("abc", "abcd")


def test_production_rejects_a_default_secret(monkeypatch):
    """
    v1 shipped a real default SECRET_KEY, so a missing env var would have
    silently signed tokens with a publicly known value.
    """
    import app.security as sec

    class FakeSettings:
        app_env = "production"
        secret_key = "change-me-in-production-use-a-long-random-string"

    monkeypatch.setattr(sec, "settings", FakeSettings())
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        sec._check_secret()


def test_development_tolerates_a_default_secret(monkeypatch):
    import app.security as sec

    class FakeSettings:
        app_env = "development"
        secret_key = "change-me-in-production-use-a-long-random-string"

    monkeypatch.setattr(sec, "settings", FakeSettings())
    assert sec._check_secret()
