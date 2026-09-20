"""
Async SQLAlchemy database engine, session management, and Base model.

Schema is owned by Alembic. `init_db` verifies, it does not create.
"""

import logging

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Ensure we use asyncpg driver even if Railway/PaaS injects standard postgres:// url
db_url = settings.database_url
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
elif db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(
    db_url,
    echo=settings.app_env == "development",
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=300,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields an async DB session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    Verify the schema is at the expected migration, and say so if it is not.

    This function no longer creates or alters anything. It used to call
    `Base.metadata.create_all` and then apply a series of hand-written
    `ALTER TABLE ... IF NOT EXISTS` statements, each wrapped in a bare
    `except Exception: pass`, under a comment reading "Since this project
    doesn't use Alembic". The project uses Alembic now, and two schema
    managers is how a schema drifts: `create_all` would happily recreate a
    table Alembic had deliberately altered, and the silent excepts meant a
    failed migration looked exactly like a successful one.

    Schema changes go through `alembic upgrade head`. Startup only checks.
    """
    from sqlalchemy import inspect

    try:
        async with engine.connect() as conn:
            revision = await conn.run_sync(_current_revision)
            tables = await conn.run_sync(
                lambda sync_conn: set(inspect(sync_conn).get_table_names())
            )
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        logger.error("could not reach the database at startup: %s", exc)
        raise

    if "alembic_version" not in tables:
        logger.error(
            "database has no alembic_version table. Run `alembic upgrade head` "
            "before starting the application."
        )
        raise RuntimeError("database schema is not managed by Alembic")

    missing = {"chunks", "embedding_cache", "ingest_jobs", "provider_usage"} - tables
    if missing:
        logger.error(
            "schema is behind: missing %s. Run `alembic upgrade head`.",
            ", ".join(sorted(missing)),
        )
        raise RuntimeError(f"database schema is missing tables: {sorted(missing)}")

    logger.info("database schema at revision %s", revision or "unknown")


def _current_revision(sync_conn) -> str | None:
    from sqlalchemy import inspect
    from sqlalchemy import text as sa_text

    if "alembic_version" not in inspect(sync_conn).get_table_names():
        return None
    row = sync_conn.execute(sa_text("SELECT version_num FROM alembic_version")).first()
    return row[0] if row else None


async def close_db() -> None:
    """Dispose the connection pool on shutdown."""
    await engine.dispose()
