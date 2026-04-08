"""
Embedding generator — HuggingFace Inference API (free, serverless).
No local model loading, no GPU/RAM needed. Works on free-tier servers.
Model: BAAI/bge-small-en-v1.5 (384-dim)

Performance optimizations:
- Async concurrent HTTP calls (10 in-flight at once)
- Batch size 32 (up from 8)
- Content-hash based caching — skips re-embedding unchanged chunks
"""

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional, Sequence, Union

import httpx

from app.config import get_settings
from app.ingestion.chunker import CodeChunk
from app.ingestion.embedding_cache import (
    content_hash,
    load_cache,
    save_cache,
)

logger = logging.getLogger(__name__)
settings = get_settings()

HF_API_URL = (
    f"https://router.huggingface.co/hf-inference/models/{settings.embedding_model}"
)
BATCH_SIZE = 32
CONCURRENT_REQUESTS = 10
MAX_RETRIES = 5
RETRY_DELAY = 10  # seconds

# Type alias for progress callback: (embedded_so_far, total_to_embed, cache_hits)
# Supports both sync and async callbacks
ProgressCallback = Optional[Callable[[int, int, int], Union[None, Awaitable[None]]]]


def _prepare_text(chunk: CodeChunk) -> str:
    """
    Produce an embedding-friendly text representation of the chunk.
    Includes metadata preamble so the vector captures file context.
    """
    header = (
        f"File: {chunk.file_path}\n"
        f"Language: {chunk.language}\n"
        f"Lines: {chunk.start_line}-{chunk.end_line}\n"
    )
    if chunk.name:
        header += f"Name: {chunk.name}\n"
    header += f"Type: {chunk.chunk_type}\n---\n"
    return header + chunk.content


async def _call_hf_api_async(
    client: httpx.AsyncClient,
    texts: list[str],
    headers: dict[str, str],
) -> list[list[float]]:
    """Call HuggingFace Inference API with retry logic for cold starts (async)."""
    payload = {"inputs": texts, "options": {"wait_for_model": True}}

    for attempt in range(MAX_RETRIES):
        try:
            response = await client.post(
                HF_API_URL,
                json=payload,
                headers=headers,
                timeout=120.0,
            )

            if response.status_code == 503:
                # Model is loading (cold start on HF serverless)
                body = response.json()
                wait_time = body.get("estimated_time", RETRY_DELAY)
                logger.info(
                    "HF model loading, waiting %.1fs (attempt %d/%d)",
                    wait_time,
                    attempt + 1,
                    MAX_RETRIES,
                )
                await asyncio.sleep(min(wait_time, 60))
                continue

            if response.status_code == 429:
                logger.warning("HF rate limited, waiting %ds", RETRY_DELAY)
                await asyncio.sleep(RETRY_DELAY)
                continue

            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException:
            logger.warning("HF API timeout (attempt %d/%d)", attempt + 1, MAX_RETRIES)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
                continue
            raise

    raise RuntimeError(f"HF Inference API failed after {MAX_RETRIES} retries")


async def embed_chunks(
    chunks: Sequence[CodeChunk],
    repo_id: str,
    batch_size: int = BATCH_SIZE,
    progress_callback: ProgressCallback = None,
) -> list[list[float]]:
    """
    Generate embeddings for code chunks via HuggingFace Inference API.

    Optimizations:
    1. Content-hash cache — reuse embeddings for unchanged chunks
    2. Concurrent API calls — up to CONCURRENT_REQUESTS in-flight
    3. Larger batch size — 32 texts per API call

    Args:
        chunks: Code chunks to embed
        repo_id: Repository ID for cache lookup
        batch_size: Number of texts per API call
        progress_callback: Optional callback(embedded_so_far, total, cache_hits)
    """
    token = settings.huggingface_api_token
    if not token:
        raise RuntimeError(
            "HUGGINGFACE_API_TOKEN is not set. Add it to your environment variables."
        )

    # Prepare texts and compute content hashes
    texts = [_prepare_text(c) for c in chunks]
    hashes = [content_hash(t) for t in texts]
    total = len(texts)

    # Load cache and identify misses
    cache = load_cache(repo_id)
    embeddings: list[Optional[list[float]]] = [None] * total
    miss_indices: list[int] = []

    for i, h in enumerate(hashes):
        if h in cache:
            embeddings[i] = cache[h]
        else:
            miss_indices.append(i)

    cache_hits = total - len(miss_indices)
    logger.info(
        "Embedding cache: %d/%d hits (%.0f%%), %d chunks need fresh embedding",
        cache_hits,
        total,
        (cache_hits / total * 100) if total else 0,
        len(miss_indices),
    )

    if progress_callback:
        await progress_callback(cache_hits, total, cache_hits)

    # If all cached, return immediately
    if not miss_indices:
        logger.info("✅ All %d embeddings served from cache", total)
        return embeddings  # type: ignore

    # Batch the misses and embed concurrently
    miss_texts = [texts[i] for i in miss_indices]
    miss_hashes = [hashes[i] for i in miss_indices]
    miss_embeddings: list[Optional[list[float]]] = [None] * len(miss_indices)

    headers = {"Authorization": f"Bearer {token}"}
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    embedded_count = cache_hits  # start from cache hits for progress

    async def _embed_batch(batch_idx: int, batch_texts: list[str]) -> list[list[float]]:
        nonlocal embedded_count
        async with semaphore:
            result = await _call_hf_api_async(client, batch_texts, headers)
            embedded_count += len(batch_texts)
            batch_num = batch_idx + 1
            total_batches = (len(miss_texts) + batch_size - 1) // batch_size
            logger.info(
                "Embedding batch %d/%d (%d texts) — overall %d/%d done",
                batch_num,
                total_batches,
                len(batch_texts),
                embedded_count,
                total,
            )
            if progress_callback:
                await progress_callback(embedded_count, total, cache_hits)
            return result

    async with httpx.AsyncClient() as client:
        tasks = []
        for batch_idx, start in enumerate(range(0, len(miss_texts), batch_size)):
            batch = miss_texts[start : start + batch_size]
            tasks.append(_embed_batch(batch_idx, batch))

        results = await asyncio.gather(*tasks)

    # Flatten results and place them in the correct positions
    flat_idx = 0
    for result_batch in results:
        for vec in result_batch:
            miss_embeddings[flat_idx] = vec
            flat_idx += 1

    # Map miss results back to the full embeddings list and update cache
    for i, miss_i in enumerate(miss_indices):
        embeddings[miss_i] = miss_embeddings[i]
        cache[miss_hashes[i]] = miss_embeddings[i]  # type: ignore

    # Prune old cache entries no longer in this repo and save
    current_hashes = set(hashes)
    pruned = {k: v for k, v in cache.items() if k in current_hashes}
    save_cache(repo_id, pruned)

    logger.info(
        "✅ Generated %d embeddings total (%d cached, %d fresh)",
        total,
        cache_hits,
        len(miss_indices),
    )
    return embeddings  # type: ignore


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string via HF API.
    bge models benefit from a query prefix for retrieval tasks.
    """
    token = settings.huggingface_api_token
    if not token:
        raise RuntimeError("HUGGINGFACE_API_TOKEN is not set.")

    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": [prefixed], "options": {"wait_for_model": True}}

    for attempt in range(MAX_RETRIES):
        try:
            response = httpx.post(
                HF_API_URL,
                json=payload,
                headers=headers,
                timeout=120.0,
            )
            if response.status_code == 503:
                body = response.json()
                wait_time = body.get("estimated_time", RETRY_DELAY)
                time.sleep(min(wait_time, 60))
                continue
            if response.status_code == 429:
                time.sleep(RETRY_DELAY)
                continue
            response.raise_for_status()
            return response.json()[0]
        except httpx.TimeoutException:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            raise

    raise RuntimeError(f"HF Inference API failed after {MAX_RETRIES} retries")
