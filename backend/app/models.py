"""
SQLAlchemy ORM models for the application.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_new_uuid)
    email = Column(String(320), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False, default="")
    hashed_password = Column(String(255), nullable=True)  # null for OAuth-only users
    oauth_provider = Column(String(50), nullable=True)  # "github" | "google"
    oauth_id = Column(String(255), nullable=True)
    avatar_url = Column(String(1024), nullable=True)
    # Password reset fields
    password_reset_token = Column(String(255), nullable=True)
    reset_token_expires = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    # Relationships — user owns repos and conversations
    repositories = relationship(
        "Repository", back_populates="owner", cascade="all, delete-orphan"
    )
    conversations = relationship(
        "Conversation", back_populates="owner", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User {self.email}>"


class Repository(Base):
    __tablename__ = "repositories"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_new_uuid)
    user_id = Column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    name = Column(String(255), nullable=False)
    url = Column(String(1024), nullable=True)
    local_path = Column(String(1024), nullable=False)
    status = Column(
        SAEnum("pending", "ingesting", "ready", "failed", name="repo_status"),
        default="pending",
        nullable=False,
    )
    file_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    ingestion_progress = Column(Integer, default=0, server_default="0")
    ingestion_total_chunks = Column(Integer, default=0, server_default="0")
    ingestion_cached_chunks = Column(Integer, default=0, server_default="0")
    ingestion_phase = Column(
        String(20), default="clone", server_default="clone"
    )  # clone, parse, embed, store
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    owner = relationship("User", back_populates="repositories")
    repo_conversations = relationship(
        "Conversation",
        back_populates="repository",
        cascade="all, delete-orphan",
        foreign_keys="[Conversation.repo_id]",
    )

    def __repr__(self) -> str:
        return f"<Repository {self.name} [{self.status}]>"


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_new_uuid)
    user_id = Column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    repo_id = Column(
        UUID(as_uuid=False),
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
    )
    title = Column(String(512), default="New Conversation")
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    owner = relationship("User", back_populates="conversations")
    repository = relationship(
        "Repository", back_populates="repo_conversations", foreign_keys=[repo_id]
    )
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    def __repr__(self) -> str:
        return f"<Conversation {self.id[:8]} — {self.title}>"


class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_new_uuid)
    conversation_id = Column(
        UUID(as_uuid=False),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    role = Column(
        SAEnum("user", "assistant", "system", "tool", name="message_role"),
        nullable=False,
    )
    content = Column(Text, nullable=False)
    metadata_json = Column(
        Text, nullable=True
    )  # stringified JSON for tool calls, sources, etc.
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message {self.role} {self.id[:8]}>"
