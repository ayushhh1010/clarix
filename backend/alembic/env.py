"""
Alembic environment.

The URL comes from application settings rather than alembic.ini, so there is
exactly one source of truth for which database is in play. asyncpg cannot
drive Alembic's synchronous migration context, so the async URL is rewritten
to psycopg for migrations only -- the application keeps using asyncpg.

`ALEMBIC_DATABASE_URL` overrides, which is how the test harness points at an
ephemeral embedded Postgres.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import create_engine, pool

from alembic import context

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.models  # noqa: E402,F401  (register v1 tables on Base)
import app.models_v2  # noqa: E402,F401  (register v2 tables on Base)
from app.database import Base  # noqa: E402

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _sync_url() -> str:
    override = os.environ.get("ALEMBIC_DATABASE_URL")
    if override:
        return override

    from app.config import get_settings

    url = get_settings().database_url
    # Alembic runs synchronously; swap the async driver for a sync one.
    for async_driver, sync_driver in (
        ("postgresql+asyncpg", "postgresql+psycopg"),
        ("postgres://", "postgresql+psycopg://"),
    ):
        if url.startswith(async_driver):
            return _translate_ssl(url.replace(async_driver, sync_driver, 1))
    return url


# asyncpg spells it `ssl`, libpq (and therefore psycopg) spells it
# `sslmode`, and the values are not quite the same vocabulary either.
_SSL_ALIASES = {
    "true": "require", "1": "require", "yes": "require", "on": "require",
    "false": "disable", "0": "disable", "no": "disable", "off": "disable",
}


def _translate_ssl(url: str) -> str:
    """
    Carry the TLS setting across the driver swap.

    Swapping only the driver name leaves an asyncpg `?ssl=require` in place,
    and psycopg rejects it outright:

        invalid connection option "ssl"

    Every managed provider hands out a URL that needs TLS -- Neon and
    Supabase both -- so without this `alembic upgrade head` cannot reach
    any of them, which is exactly the step a first deployment runs.

    An explicit `sslmode` already present wins; it was chosen deliberately.
    """
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    parts = urlsplit(url)
    if not parts.query:
        return url

    params = parse_qsl(parts.query, keep_blank_values=True)
    kept, ssl_value = [], None
    for key, value in params:
        if key == "ssl":
            ssl_value = value
        else:
            kept.append((key, value))

    if ssl_value is not None and not any(k == "sslmode" for k, _ in kept):
        kept.append(("sslmode", _SSL_ALIASES.get(ssl_value.lower(), ssl_value)))

    return urlunsplit(parts._replace(query=urlencode(kept)))


def run_migrations_offline() -> None:
    context.configure(
        url=_sync_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_sync_url(), poolclass=pool.NullPool, future=True)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            # pgvector and pg_trgm create objects Alembic does not model;
            # without this, autogenerate proposes dropping them.
            include_object=_include_object,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


def _include_object(obj, name, type_, reflected, compare_to) -> bool:
    """Keep extension-owned objects out of autogenerate diffs."""
    if type_ == "table" and name in {"spatial_ref_sys"}:
        return False
    return True


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
