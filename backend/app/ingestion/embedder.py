"""
Embedding generator — HuggingFace Inference API (free, serverless).
No local model loading, no GPU/RAM needed. Works on free-tier servers.
Model: BAAI/bge-small-en-v1.5 (384-dim)
"""

import logging
import time
from typing import Sequence

import httpx

from app.config import get_settings
from app.ingestion.chunker import CodeChunk

logger = logging.getLogger(__name__)
settings = get_settings()

HF_API_URL = f"https://api-inference.huggingface.co/models/{settings.embedding_model}"
BATCH_SIZE = 8
MAX_RETRIES = 5
RETRY_DELAY = 10  # seconds


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


def _call_hf_api(texts: list[str]) -> list[list[float]]:
    """Call HuggingFace Inference API with retry logic for cold starts."""
    token = settings.huggingface_api_token
    if not token:
        raise RuntimeError("HUGGINGFACE_API_TOKEN is not set. Add it to your environment variables.")
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": texts, "options": {"wait_for_model": True}}

    for attempt in range(MAX_RETRIES):
        try:
            response = httpx.post(
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
                    wait_time, attempt + 1, MAX_RETRIES,
                )
                time.sleep(min(wait_time, 60))
                continue

            if response.status_code == 429:
                logger.warning("HF rate limited, waiting %ds", RETRY_DELAY)
                time.sleep(RETRY_DELAY)
                continue

            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException:
            logger.warning("HF API timeout (attempt %d/%d)", attempt + 1, MAX_RETRIES)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            raise

    raise RuntimeError(f"HF Inference API failed after {MAX_RETRIES} retries")


def embed_chunks(chunks: Sequence[CodeChunk], batch_size: int = BATCH_SIZE) -> list[list[float]]:
    """Generate embeddings for code chunks via HuggingFace Inference API."""
    texts = [_prepare_text(c) for c in chunks]
    all_embeddings: list[list[float]] = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size

        logger.info(
            "Embedding batch %d/%d (%d–%d of %d texts)",
            batch_num, total_batches, i + 1, i + len(batch), total,
        )

        batch_embeddings = _call_hf_api(batch)
        all_embeddings.extend(batch_embeddings)

    logger.info("✅ Generated %d embeddings total", len(all_embeddings))
    return all_embeddings


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string via HF API.
    bge models benefit from a query prefix for retrieval tasks.
    """
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    result = _call_hf_api([prefixed])
    return result[0]