"""
Conversation persistence.

Replaces `app/memory/`, which was three modules wrapping two stores. The
audit found the Redis half held exactly one thing -- a facts set that no
code path ever wrote to -- so `recall_facts()` returned `[]` on every
request, the `if facts:` branch never fired, and every chat request paid a
Redis round trip plus a 20-connection pool to receive an empty list.

What remained useful was already Postgres. That is all this module is.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Conversation, Message

logger = logging.getLogger(__name__)

# Turns of history sent back to the model. Deliberately small: history
# competes with retrieved code for a ~4,000-token budget, and on free-tier
# limits the code is worth more than the third-last exchange.
DEFAULT_HISTORY_TURNS = 6


async def get_or_create_conversation(
    db: AsyncSession,
    repo_id: str,
    user_id: str | None,
    conversation_id: str | None = None,
    title: str = "New Conversation",
) -> Conversation:
    if conversation_id:
        conv = await db.get(Conversation, conversation_id)
        if conv and conv.repo_id == repo_id:
            if conv.user_id is not None and user_id is not None and conv.user_id != user_id:
                raise PermissionError("conversation belongs to another user")
            return conv

    conv = Conversation(
        id=str(uuid.uuid4()),
        repo_id=repo_id,
        user_id=user_id,
        title=title[:512] or "New Conversation",
    )
    db.add(conv)
    await db.flush()
    return conv


async def save_message(
    db: AsyncSession,
    conversation_id: str,
    role: str,
    content: str,
    metadata: dict | None = None,
) -> Message:
    message = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role=role,
        content=content,
        metadata_json=json.dumps(metadata) if metadata else None,
    )
    db.add(message)
    # Keep the conversation's ordering key fresh so the sidebar sorts by
    # actual activity rather than by creation.
    conv = await db.get(Conversation, conversation_id)
    if conv:
        conv.updated_at = datetime.now(UTC)
    await db.flush()
    return message


async def load_history(
    db: AsyncSession, conversation_id: str, turns: int = DEFAULT_HISTORY_TURNS
) -> list[Message]:
    """
    Return the most recent messages in chronological order.

    Fetched newest-first with a LIMIT and then reversed: ordering ascending
    and slicing in Python would read the entire conversation to return six
    rows of it.
    """
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(turns * 2)
    )
    return list(reversed(result.scalars().all()))


async def count_messages(db: AsyncSession, conversation_id: str) -> int:
    return (
        await db.execute(
            select(func.count(Message.id)).where(
                Message.conversation_id == conversation_id
            )
        )
    ).scalar() or 0
