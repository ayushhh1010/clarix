"""
Async SQLAlchemy database engine, session management, and Base model.
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_url,
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
    """Create all tables on startup and run lightweight migrations."""
    async with engine.begin() as conn:
        from app import models  # noqa: F401 — ensure models are imported

        await conn.run_sync(Base.metadata.create_all)

    # ── Lightweight migrations (add columns if missing) ──────
    # Since this project doesn't use Alembic, we handle simple column additions here.
    async with engine.begin() as conn:
        # Add user_id to repositories
        try:
            await conn.execute(
                text(
                    "ALTER TABLE repositories ADD COLUMN IF NOT EXISTS "
                    "user_id UUID REFERENCES users(id) ON DELETE CASCADE"
                )
            )
        except Exception:
            pass  # Column already exists or DB doesn't support IF NOT EXISTS

        # Add user_id to conversations
        try:
            await conn.execute(
                text(
                    "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS "
                    "user_id UUID REFERENCES users(id) ON DELETE CASCADE"
                )
            )
        except Exception:
            pass

        # Add password reset columns to users
        try:
            await conn.execute(
                text(
                    "ALTER TABLE users ADD COLUMN IF NOT EXISTS "
                    "password_reset_token VARCHAR(255)"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE users ADD COLUMN IF NOT EXISTS "
                    "reset_token_expires TIMESTAMP"
                )
            )
        except Exception:
            pass

        # Create indexes if missing
        try:
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_repositories_user_id ON repositories(user_id)"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_conversations_user_id ON conversations(user_id)"
                )
            )
        except Exception:
            pass

        # Add ingestion progress columns to repositories
        for col_name in (
            "ingestion_progress",
            "ingestion_total_chunks",
            "ingestion_cached_chunks",
        ):
            try:
                await conn.execute(
                    text(
                        f"ALTER TABLE repositories ADD COLUMN IF NOT EXISTS "
                        f"{col_name} INTEGER DEFAULT 0"
                    )
                )
            except Exception:
                pass

        # Add ingestion_phase column
        try:
            await conn.execute(
                text(
                    "ALTER TABLE repositories ADD COLUMN IF NOT EXISTS "
                    "ingestion_phase VARCHAR(20) DEFAULT 'clone'"
                )
            )
        except Exception:
            pass


async def close_db() -> None:
    """Dispose the connection pool on shutdown."""
    await engine.dispose()
