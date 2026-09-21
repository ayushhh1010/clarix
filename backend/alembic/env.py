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
from app.db_url import normalise_database_url

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
    # Alembic runs synchronously and cannot use asyncpg. The same helper
    # the application uses handles the driver swap and the TLS parameter,
    # so the two cannot disagree about what a provider URL means.
    return normalise_database_url(url, driver="psycopg")


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
