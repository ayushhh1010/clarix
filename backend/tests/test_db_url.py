"""
Database URL normalisation, against the forms providers actually emit.

The deploy failure this prevents
--------------------------------
Neon's dashboard gives you

    postgresql://user:pass@ep-x.aws.neon.tech/db?sslmode=require&channel_binding=require

Pasted verbatim, that used to break twice. Alembic only recognised
`postgresql+asyncpg` and `postgres://`, so a plain `postgresql://` fell
through to psycopg2, which is not installed. And asyncpg does not accept
libpq's spelling of the TLS option:

    TypeError: connect() got an unexpected keyword argument 'sslmode'

Both were confirmed against a real Neon instance before this was written.
"""

from __future__ import annotations

import pytest

from app.db_url import normalise_database_url, redact

# The exact shapes providers hand out.
NEON = ("postgresql://u:p@ep-x.ap-southeast-1.aws.neon.tech/db"
        "?sslmode=require&channel_binding=require")
SUPABASE = "postgresql://postgres:p@db.abc.supabase.co:5432/postgres?sslmode=require"
HEROKU = "postgres://u:p@ec2-1-2-3-4.compute.amazonaws.com:5432/dbname"
ALREADY_ASYNC = "postgresql+asyncpg://u:p@host/db?ssl=require"


# --- driver selection ------------------------------------------------------

@pytest.mark.parametrize("url", [NEON, SUPABASE, HEROKU, ALREADY_ASYNC])
def test_every_provider_form_reaches_asyncpg(url):
    assert normalise_database_url(url, driver="asyncpg").startswith(
        "postgresql+asyncpg://"
    )


@pytest.mark.parametrize("url", [NEON, SUPABASE, HEROKU, ALREADY_ASYNC])
def test_every_provider_form_reaches_psycopg(url):
    """
    Alembic cannot run on asyncpg. A `postgresql://` that is not rewritten
    falls through to psycopg2, which this project does not install.
    """
    assert normalise_database_url(url, driver="psycopg").startswith(
        "postgresql+psycopg://"
    )


# --- the TLS parameter -----------------------------------------------------

def test_sslmode_becomes_ssl_for_asyncpg():
    out = normalise_database_url(NEON, driver="asyncpg")
    assert "ssl=require" in out
    assert "sslmode" not in out, "asyncpg raises TypeError on sslmode"


def test_ssl_becomes_sslmode_for_psycopg():
    out = normalise_database_url(ALREADY_ASYNC, driver="psycopg")
    assert "sslmode=require" in out
    assert "ssl=require" not in out.replace("sslmode=require", "")


@pytest.mark.parametrize(
    "given,expected",
    [("true", "require"), ("1", "require"), ("on", "require"),
     ("false", "disable"), ("0", "disable"),
     ("require", "require"), ("verify-full", "verify-full")],
)
def test_boolean_and_libpq_ssl_values(given, expected):
    url = f"postgresql://u:p@h/db?ssl={given}"
    assert f"sslmode={expected}" in normalise_database_url(url, driver="psycopg")


def test_a_url_without_tls_is_left_without_tls():
    """Local development has no sslmode, and must not acquire one."""
    plain = "postgresql://clarix:secret@localhost:5433/clarix_db"
    assert "ssl" not in normalise_database_url(plain, driver="asyncpg")


# --- parameters asyncpg cannot take ---------------------------------------

def test_channel_binding_is_dropped_for_asyncpg():
    """
    Neon appends it by default and asyncpg rejects it. Dropping is safe:
    `ssl` already carries the security-relevant setting.
    """
    out = normalise_database_url(NEON, driver="asyncpg")
    assert "channel_binding" not in out


def test_channel_binding_is_kept_for_psycopg():
    """libpq understands it, so psycopg should still receive it."""
    assert "channel_binding" in normalise_database_url(NEON, driver="psycopg")


def test_host_database_and_credentials_survive():
    out = normalise_database_url(NEON, driver="asyncpg")
    assert "u:p@ep-x.ap-southeast-1.aws.neon.tech" in out
    assert out.endswith("ssl=require")
    assert "/db?" in out


# --- robustness ------------------------------------------------------------

def test_an_empty_url_is_returned_unchanged():
    assert normalise_database_url("", driver="asyncpg") == ""


def test_a_non_postgres_url_is_not_mangled():
    sqlite = "sqlite+aiosqlite:///./local.db"
    assert normalise_database_url(sqlite, driver="asyncpg") == sqlite


def test_an_unknown_driver_is_refused():
    with pytest.raises(ValueError, match="unknown driver"):
        normalise_database_url(NEON, driver="psycopg2")


def test_normalisation_is_idempotent():
    once = normalise_database_url(NEON, driver="asyncpg")
    assert normalise_database_url(once, driver="asyncpg") == once


# --- redaction -------------------------------------------------------------

def test_redact_removes_the_password():
    """Connection strings reach deploy logs; passwords must not."""
    out = redact(NEON)
    assert ":p@" not in out
    assert "***" in out
    assert "ep-x.ap-southeast-1.aws.neon.tech" in out, "the host stays visible"


def test_redact_tolerates_a_url_without_credentials():
    assert redact("postgresql://localhost/db") == "postgresql://localhost/db"
