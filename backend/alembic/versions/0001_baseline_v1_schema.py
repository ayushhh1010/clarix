"""Baseline: the v1 schema as created by Base.metadata.create_all.

v1 had no migrations -- `database.py` called `create_all`, which can add
tables but can never alter a column. This revision captures the schema that
produced, so later revisions have a known starting point.

It is idempotent by inspection rather than requiring `alembic stamp`. An
existing deployment can run `alembic upgrade head` directly: tables already
present are skipped and logged. A fresh database gets them created. The
alternative (documenting a stamp step) fails silently and confusingly when
someone forgets it, and this schema is small enough that inspection is cheap.

Revision ID: 0001
Revises:
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")


def _existing_tables() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def upgrade() -> None:
    present = _existing_tables()

    def skip(name: str) -> bool:
        if name in present:
            logger.info("baseline: %s already exists, skipping", name)
            return True
        return False

    if not skip("users"):
        op.create_table(
            "users",
            sa.Column("id", UUID(as_uuid=False), primary_key=True),
            sa.Column("email", sa.String(320), nullable=False, unique=True),
            sa.Column("name", sa.String(255), nullable=False, server_default=""),
            sa.Column("hashed_password", sa.String(255), nullable=True),
            sa.Column("oauth_provider", sa.String(50), nullable=True),
            sa.Column("oauth_id", sa.String(255), nullable=True),
            sa.Column("avatar_url", sa.String(1024), nullable=True),
            sa.Column("password_reset_token", sa.String(255), nullable=True),
            sa.Column("reset_token_expires", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
        )
        op.create_index("ix_users_email", "users", ["email"])

    if not skip("repositories"):
        op.create_table(
            "repositories",
            sa.Column("id", UUID(as_uuid=False), primary_key=True),
            sa.Column(
                "user_id",
                UUID(as_uuid=False),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=True,
            ),
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("url", sa.String(1024), nullable=True),
            sa.Column("local_path", sa.String(1024), nullable=False),
            sa.Column(
                "status",
                sa.Enum("pending", "ingesting", "ready", "failed", name="repo_status"),
                nullable=False,
                server_default="pending",
            ),
            sa.Column("file_count", sa.Integer(), server_default="0"),
            sa.Column("chunk_count", sa.Integer(), server_default="0"),
            sa.Column("ingestion_progress", sa.Integer(), server_default="0"),
            sa.Column("ingestion_total_chunks", sa.Integer(), server_default="0"),
            sa.Column("ingestion_cached_chunks", sa.Integer(), server_default="0"),
            sa.Column("ingestion_phase", sa.String(20), server_default="clone"),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
        )
        op.create_index("ix_repositories_user_id", "repositories", ["user_id"])

    if not skip("conversations"):
        op.create_table(
            "conversations",
            sa.Column("id", UUID(as_uuid=False), primary_key=True),
            sa.Column(
                "user_id",
                UUID(as_uuid=False),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=True,
            ),
            sa.Column(
                "repo_id",
                UUID(as_uuid=False),
                sa.ForeignKey("repositories.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("title", sa.String(512), server_default="New Conversation"),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
        )
        op.create_index("ix_conversations_user_id", "conversations", ["user_id"])

    if not skip("messages"):
        op.create_table(
            "messages",
            sa.Column("id", UUID(as_uuid=False), primary_key=True),
            sa.Column(
                "conversation_id",
                UUID(as_uuid=False),
                sa.ForeignKey("conversations.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "role",
                sa.Enum("user", "assistant", "system", "tool", name="message_role"),
                nullable=False,
            ),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("metadata_json", sa.Text(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
        )
        op.create_index("ix_messages_conversation", "messages", ["conversation_id"])


def downgrade() -> None:
    for table in ("messages", "conversations", "repositories", "users"):
        op.drop_table(table)
    sa.Enum(name="message_role").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="repo_status").drop(op.get_bind(), checkfirst=True)
