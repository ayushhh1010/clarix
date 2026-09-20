"""
Password hashing, JWT issuance/verification, and the current-user dependency.

Migrated off `python-jose` + `passlib` onto `PyJWT` + `bcrypt` directly.
Three measured reasons, not a preference:

1.  passlib 1.7.4 is unmaintained and reads `bcrypt.__about__`, which was
    removed in bcrypt 4.1. Against the pinned bcrypt 4.2.0 it logs
    "(trapped) error reading bcrypt version" with an AttributeError
    traceback on every backend load. It still works -- passlib traps it --
    but the library is visibly broken against its own dependency.

2.  bcrypt silently truncates at 72 bytes. Verified: hashing a 100-character
    password succeeds and the result is indistinguishable from hashing its
    first 72 bytes, so those users can authenticate with a prefix of their
    password. That is now an explicit error rather than a silent downgrade.

3.  python-jose is less actively maintained than PyJWT for what we use it
    for (HS256 sign/verify).

WIRE COMPATIBILITY. Both changes are backward compatible with data already
in the database, and this is asserted in tests rather than assumed:

  * JWTs are HS256 over the same claims, so tokens minted by the previous
    python-jose implementation still verify here
    (tests/test_security.py::test_verifies_a_token_minted_by_python_jose).
  * bcrypt hashes are unchanged in format; existing `$2b$` hashes verify
    (tests/test_security.py::test_verifies_a_preexisting_passlib_hash).

No password reset or re-issue is required by this migration.
"""

from __future__ import annotations

import hmac
import logging
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db

logger = logging.getLogger(__name__)
settings = get_settings()

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24 * 7  # 7 days

# bcrypt hashes at most 72 bytes of input and discards the rest. Enforced
# rather than truncated: see the module docstring.
BCRYPT_MAX_PASSWORD_BYTES = 72

# Work factor. 12 is the current common default; raising it is a
# backward-compatible change because the cost is encoded in each hash.
BCRYPT_ROUNDS = 12

# Rejected at startup in production. The v1 default was a real string that
# would have signed tokens if the env var were ever missing.
_INSECURE_SECRETS = {
    "",
    "change-me-in-production-use-a-long-random-string",
    "your-secret-key-here",
    "secret",
}


class PasswordTooLongError(ValueError):
    """Raised instead of letting bcrypt silently discard the tail."""


def _check_secret() -> str:
    secret = settings.secret_key
    if settings.app_env == "production" and secret in _INSECURE_SECRETS:
        raise RuntimeError(
            "SECRET_KEY is unset or left at its default in production. "
            "Generate one with: python -c \"import secrets; "
            "print(secrets.token_urlsafe(64))\""
        )
    return secret


# --- passwords -------------------------------------------------------------

def hash_password(password: str) -> str:
    """
    Hash a password with bcrypt.

    Raises PasswordTooLongError above 72 bytes rather than truncating. The
    alternative mitigation -- pre-hashing with SHA-256 -- would change the
    hash of every existing password and lock out every current user, so it
    is not available to us without a migration.
    """
    encoded = password.encode("utf-8")
    if len(encoded) > BCRYPT_MAX_PASSWORD_BYTES:
        raise PasswordTooLongError(
            f"password is {len(encoded)} bytes; bcrypt accepts at most "
            f"{BCRYPT_MAX_PASSWORD_BYTES} and would silently discard the rest"
        )
    return bcrypt.hashpw(encoded, bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    """
    Check a password against a stored hash.

    Over-long passwords are compared against their first 72 bytes rather
    than rejected: a user whose password predates the length check must
    still be able to log in. New passwords cannot be created that way.

    Never raises on malformed stored hashes -- a corrupt row is an
    authentication failure, not a 500.
    """
    encoded = plain.encode("utf-8")[:BCRYPT_MAX_PASSWORD_BYTES]
    try:
        return bcrypt.checkpw(encoded, hashed.encode("utf-8"))
    except (ValueError, TypeError) as exc:
        logger.warning("malformed password hash rejected: %s", type(exc).__name__)
        return False


def needs_rehash(hashed: str) -> bool:
    """Whether a stored hash used a lower work factor than we now require."""
    try:
        cost = int(hashed.split("$")[2])
    except (IndexError, ValueError):
        return True
    return cost < BCRYPT_ROUNDS


# --- tokens ----------------------------------------------------------------

def create_access_token(
    user_id: str, email: str, expires_delta: timedelta | None = None
) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": user_id,
        "email": email,
        "iat": now,
        "exp": now + (expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)),
    }
    return jwt.encode(payload, _check_secret(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Verify and decode a token.

    `algorithms=[ALGORITHM]` is not optional: accepting the token's own `alg`
    header is the classic JWT confusion attack, where an attacker re-signs
    with `alg: none` or swaps HS256 for an asymmetric algorithm.
    """
    try:
        return jwt.decode(
            token,
            _check_secret(),
            algorithms=[ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def constant_time_compare(a: str, b: str) -> bool:
    """For comparing reset tokens and similar secrets."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


# --- FastAPI dependency ----------------------------------------------------

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
):
    """Extract and validate the bearer token, returning the User row."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload"
        )

    from app.models import User

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
        )

    return user
