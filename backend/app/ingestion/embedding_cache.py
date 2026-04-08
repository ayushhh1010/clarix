"""
Embedding cache — content-hash based file cache for embeddings.
Avoids re-embedding unchanged chunks on re-ingestion.

Cache is stored as a JSON file per repo: data/embedding_cache/{repo_id}.json
Mapping: content_hash (SHA-256, 16-char hex) → embedding vector (list[float])
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

CACHE_DIR = Path(settings.chroma_persist_dir).parent / "embedding_cache"


def _cache_path(repo_id: str) -> Path:
    """Get the cache file path for a given repo."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{repo_id}.json"


def content_hash(text: str) -> str:
    """Compute a 16-char SHA-256 hex digest of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_cache(repo_id: str) -> dict[str, list[float]]:
    """Load the embedding cache for a repo. Returns empty dict if not found."""
    path = _cache_path(repo_id)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        logger.info("Loaded embedding cache for repo %s: %d entries", repo_id, len(cache))
        return cache
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load embedding cache for %s: %s", repo_id, exc)
        return {}


def save_cache(repo_id: str, cache: dict[str, list[float]]) -> None:
    """Save the embedding cache for a repo."""
    path = _cache_path(repo_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        logger.info("Saved embedding cache for repo %s: %d entries", repo_id, len(cache))
    except OSError as exc:
        logger.warning("Failed to save embedding cache for %s: %s", repo_id, exc)


def delete_cache(repo_id: str) -> None:
    """Delete the embedding cache file for a repo."""
    path = _cache_path(repo_id)
    if path.exists():
        try:
            path.unlink()
            logger.info("Deleted embedding cache for repo %s", repo_id)
        except OSError as exc:
            logger.warning("Failed to delete embedding cache for %s: %s", repo_id, exc)
