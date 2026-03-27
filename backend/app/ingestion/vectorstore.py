"""
Vector store — ChromaDB integration for storing and querying code embeddings.
"""

import logging
from typing import Optional, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.ingestion.chunker import CodeChunk

logger = logging.getLogger(__name__)
settings = get_settings()

_chroma_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB client initialized at %s", settings.chroma_path)
    return _chroma_client


def _collection_name(repo_id: str) -> str:
    """Sanitize repo_id into a valid ChromaDB collection name."""
    # ChromaDB requires: 3-63 chars, starts/ends with alphanum, only [a-zA-Z0-9._-]
    name = f"repo_{repo_id.replace('-', '_')}"
    return name[:63]


def store_chunks(
    repo_id: str,
    chunks: Sequence[CodeChunk],
    embeddings: Sequence[list[float]],
) -> int:
    """
    Store code chunks and their embeddings in ChromaDB.
    Returns the number of chunks stored.
    """
    client = _get_client()
    col_name = _collection_name(repo_id)

    # Delete existing collection if re-ingesting
    try:
        client.delete_collection(col_name)
        logger.info("Deleted existing collection %s", col_name)
    except ValueError:
        pass

    collection = client.create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB batch upsert
    batch_size = 500
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_embeds = embeddings[i : i + batch_size]

        ids = [c.chunk_id for c in batch_chunks]
        documents = [c.content for c in batch_chunks]
        metadatas = [c.metadata for c in batch_chunks]
        embedding_list = [list(e) for e in batch_embeds]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embedding_list,
        )
        total += len(batch_chunks)
        logger.info("Stored batch %d–%d / %d", i, i + len(batch_chunks), len(chunks))

    logger.info("Stored %d chunks in collection %s", total, col_name)
    return total


def search(
    query_embedding: list[float],
    repo_id: str,
    top_k: int = 10,
    filter_metadata: Optional[dict] = None,
) -> list[dict]:
    """
    Search the vector store for chunks similar to the query embedding.

    Returns a list of dicts with keys: content, metadata, distance.
    """
    client = _get_client()
    col_name = _collection_name(repo_id)

    try:
        collection = client.get_collection(col_name)
    except ValueError:
        logger.warning("Collection %s not found", col_name)
        return []

    where = filter_metadata if filter_metadata else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict] = []
    if results and results["documents"]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "content": doc,
                "metadata": meta,
                "distance": dist,
            })

    logger.debug("Vector search returned %d results for repo %s", len(hits), repo_id)
    return hits


def delete_collection(repo_id: str) -> None:
    """Remove an entire repo collection from ChromaDB."""
    client = _get_client()
    col_name = _collection_name(repo_id)
    try:
        client.delete_collection(col_name)
        logger.info("Deleted collection %s", col_name)
    except ValueError:
        logger.warning("Collection %s not found for deletion", col_name)
