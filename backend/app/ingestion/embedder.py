"""
Embedding generator — HuggingFace sentence-transformers running locally on CPU.
No API key required, no rate limits, works on free-tier servers.
Model: BAAI/bge-small-en-v1.5 (384-dim, ~33MB, ~80MB RAM)
Chosen for free-tier compatibility — better quality/RAM ratio than MiniLM.
"""

import logging
from typing import Sequence

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings
from app.ingestion.chunker import CodeChunk

logger = logging.getLogger(__name__)
settings = get_settings()

_embeddings_model: HuggingFaceEmbeddings | None = None

# ── Batch config ──────────────────────────────────────────────
# Kept small (8) to stay within 512MB free tier RAM limit
BATCH_SIZE = 8


def _get_embeddings_model() -> HuggingFaceEmbeddings:
    global _embeddings_model
    if _embeddings_model is None:
        logger.info("Loading HuggingFace embedding model: %s", settings.embedding_model)
        _embeddings_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={
                "device": "cpu",
            },
            encode_kwargs={
                "normalize_embeddings": True,  # required for bge models
                "batch_size": 8,
            },
        )
        logger.info("✅ Embedding model loaded: %s", settings.embedding_model)
    return _embeddings_model


def _prepare_text(chunk: CodeChunk) -> str:
    """
    Produce an embedding-friendly text representation of the chunk.
    Includes metadata preamble so the vector captures file context.
    bge models benefit from query prefix — handled in embed_query.
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


def embed_chunks(chunks: Sequence[CodeChunk], batch_size: int = BATCH_SIZE) -> list[list[float]]:
    """
    Generate embeddings for a sequence of code chunks.
    Returns a list of embedding vectors in the same order.
    """
    model = _get_embeddings_model()
    texts = [_prepare_text(c) for c in chunks]

    all_embeddings: list[list[float]] = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i: i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size

        logger.info(
            "Embedding batch %d/%d (%d–%d of %d texts)",
            batch_num, total_batches, i + 1, i + len(batch), total,
        )

        batch_embeddings = model.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    logger.info("✅ Generated %d embeddings total", len(all_embeddings))
    return all_embeddings


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.
    bge models recommend a query prefix for better retrieval accuracy.
    """
    model = _get_embeddings_model()
    # bge-small benefits from this prefix for retrieval tasks
    prefixed_query = f"Represent this sentence for searching relevant passages: {query}"
    return model.embed_query(prefixed_query)