"""
Short-term memory — conversation history from PostgreSQL.
"""

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Conversation, Message

logger = logging.getLogger(__name__)


async def get_conversation_history(
    db: AsyncSession,
    conversation_id: str,
    limit: int = 20,
) -> list[dict]:
    """
    Load the last N messages from a conversation as role/content dicts.
    These are suitable for injection into an LLM prompt.
    """
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    messages = result.scalars().all()

    # Reverse to chronological order
    messages = list(reversed(messages))

    return [
        {"role": msg.role, "content": msg.content}
        for msg in messages
    ]


async def save_message(
    db: AsyncSession,
    conversation_id: str,
    role: str,
    content: str,
    metadata_json: Optional[str] = None,
) -> Message:
    """Save a message to the conversation history."""
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        metadata_json=metadata_json,
    )
    db.add(msg)
    await db.flush()
    logger.debug("Saved %s message to conversation %s", role, conversation_id[:8])
    return msg


async def get_or_create_conversation(
    db: AsyncSession,
    repo_id: str,
    conversation_id: Optional[str] = None,
    title: str = "New Conversation",
) -> Conversation:
    """Get an existing conversation or create a new one."""
    if conversation_id:
        conv = await db.get(Conversation, conversation_id)
        if conv:
            return conv

    conv = Conversation(repo_id=repo_id, title=title)
    db.add(conv)
    await db.flush()
    logger.info("Created new conversation %s for repo %s", conv.id, repo_id)
    return conv
