"""
Memory Manager — unified interface for both short-term and long-term memory.
"""

import json
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.memory.short_term import (
    get_conversation_history,
    save_message,
    get_or_create_conversation,
)
from app.memory.long_term import (
    store_memory,
    get_memory,
    add_fact,
    get_facts,
)

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Unified memory interface combining:
    - Short-term: PostgreSQL conversation history
    - Long-term: Redis key-value store + fact accumulation
    """

    def __init__(self, db: AsyncSession, repo_id: str):
        self.db = db
        self.repo_id = repo_id

    # ── Short-term (conversation) ────────────────────────

    async def load_conversation(
        self,
        conversation_id: str,
        limit: int = 20,
    ) -> list[dict]:
        """Load recent conversation messages."""
        return await get_conversation_history(self.db, conversation_id, limit)

    async def save_user_message(self, conversation_id: str, content: str):
        """Save a user message to the conversation."""
        return await save_message(self.db, conversation_id, "user", content)

    async def save_assistant_message(
        self,
        conversation_id: str,
        content: str,
        sources: list[dict] | None = None,
    ):
        """Save an assistant response with optional source metadata."""
        metadata = json.dumps({"sources": sources}) if sources else None
        return await save_message(self.db, conversation_id, "assistant", content, metadata)

    async def get_or_create_conv(
        self,
        conversation_id: Optional[str] = None,
        title: str = "New Conversation",
        user_id: Optional[str] = None,
    ):
        """Get existing or create new conversation."""
        return await get_or_create_conversation(
            self.db, self.repo_id, conversation_id, title, user_id=user_id
        )

    # ── Long-term (Redis) ────────────────────────────────

    async def remember(self, key: str, value, ttl: int = 86400):
        """Store something in long-term memory."""
        await store_memory(self.repo_id, key, value, ttl)

    async def recall(self, key: str):
        """Recall something from long-term memory."""
        return await get_memory(self.repo_id, key)

    async def learn_fact(self, fact: str):
        """Learn a new fact about this repo."""
        await add_fact(self.repo_id, fact)

    async def recall_facts(self) -> list[str]:
        """Recall all learned facts about this repo."""
        return await get_facts(self.repo_id)

    # ── Combined context building ────────────────────────

    async def build_context(
        self,
        conversation_id: Optional[str] = None,
    ) -> dict:
        """
        Build a combined memory context with conversation history and facts.
        Used to enrich agent prompts.
        """
        context: dict = {"conversation_history": [], "facts": []}

        if conversation_id:
            context["conversation_history"] = await self.load_conversation(
                conversation_id, limit=20
            )

        context["facts"] = await self.recall_facts()

        return context
