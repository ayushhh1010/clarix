"""
Long-term memory — Redis-backed persistent memory for agents.
Stores important facts, user preferences, and previously explored code areas.
"""

import json
import logging
from typing import Any, Optional

import redis.asyncio as aioredis

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_redis_client: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    """Get or create the async Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=20,
        )
        # Test connection
        await _redis_client.ping()
        logger.info("Redis connected at %s", settings.redis_url)
    return _redis_client


async def close_redis() -> None:
    """Close the Redis connection pool."""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis connection closed")


# ── Key Schemas ─────────────────────────────────────────────

def _memory_key(repo_id: str, key: str) -> str:
    return f"clarix:memory:{repo_id}:{key}"


def _facts_key(repo_id: str) -> str:
    return f"clarix:facts:{repo_id}"


def _session_key(session_id: str) -> str:
    return f"clarix:session:{session_id}"


# ── Memory Operations ──────────────────────────────────────

async def store_memory(repo_id: str, key: str, value: Any, ttl: int = 86400) -> None:
    """
    Store a key-value memory entry for a repo.
    TTL defaults to 24 hours.
    """
    r = await get_redis()
    full_key = _memory_key(repo_id, key)
    serialized = json.dumps(value) if not isinstance(value, str) else value
    await r.setex(full_key, ttl, serialized)
    logger.debug("Stored memory: %s (TTL=%ds)", full_key, ttl)


async def get_memory(repo_id: str, key: str) -> Optional[Any]:
    """Retrieve a memory entry for a repo."""
    r = await get_redis()
    full_key = _memory_key(repo_id, key)
    value = await r.get(full_key)
    if value is None:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


async def delete_memory(repo_id: str, key: str) -> None:
    """Delete a memory entry."""
    r = await get_redis()
    await r.delete(_memory_key(repo_id, key))


# ── Facts (accumulated knowledge about a repo) ─────────────

async def add_fact(repo_id: str, fact: str) -> None:
    """Add a learned fact about a repository (stored as a Redis set)."""
    r = await get_redis()
    await r.sadd(_facts_key(repo_id), fact)
    logger.debug("Added fact for repo %s: %.80s...", repo_id, fact)


async def get_facts(repo_id: str) -> list[str]:
    """Get all learned facts about a repository."""
    r = await get_redis()
    facts = await r.smembers(_facts_key(repo_id))
    return list(facts) if facts else []


async def clear_facts(repo_id: str) -> None:
    """Clear all facts for a repository."""
    r = await get_redis()
    await r.delete(_facts_key(repo_id))


# ── Session Cache ───────────────────────────────────────────

async def cache_session_data(session_id: str, data: dict, ttl: int = 3600) -> None:
    """Cache session-specific data (e.g. active conversations, user prefs)."""
    r = await get_redis()
    await r.setex(_session_key(session_id), ttl, json.dumps(data))


async def get_session_data(session_id: str) -> Optional[dict]:
    """Retrieve cached session data."""
    r = await get_redis()
    value = await r.get(_session_key(session_id))
    if value:
        return json.loads(value)
    return None
