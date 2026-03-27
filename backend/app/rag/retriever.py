"""
RAG Retriever — embeds a user query and retrieves relevant code chunks
from the vector store.
"""

import logging
from typing import Optional

from app.ingestion.embedder import embed_query
from app.ingestion.vectorstore import search
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RetrievedChunk:
    """A code chunk returned by retrieval, enriched with relevance score."""

    __slots__ = ("content", "file_path", "language", "start_line", "end_line",
                 "chunk_type", "name", "distance", "repo_id")

    def __init__(self, hit: dict):
        meta = hit.get("metadata", {})
        self.content = hit["content"]
        self.file_path = meta.get("file_path", "")
        self.language = meta.get("language", "")
        self.start_line = meta.get("start_line", 0)
        self.end_line = meta.get("end_line", 0)
        self.chunk_type = meta.get("chunk_type", "")
        self.name = meta.get("name", "")
        self.distance = hit.get("distance", 1.0)
        self.repo_id = meta.get("repo_id", "")

    @property
    def relevance_score(self) -> float:
        """Convert cosine distance to similarity score (0–1)."""
        return max(0.0, 1.0 - self.distance)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "relevance_score": round(self.relevance_score, 4),
        }


def retrieve_context(
    query: str,
    repo_id: str,
    top_k: Optional[int] = None,
    filter_metadata: Optional[dict] = None,
) -> list[RetrievedChunk]:
    """
    Retrieve the most relevant code chunks for a given natural language query.

    1. Embed the query
    2. Search the vector DB
    3. Return ranked RetrievedChunk objects
    """
    k = top_k or settings.rag_top_k

    logger.info("Retrieving context for query: %.80s... (repo=%s, top_k=%d)", query, repo_id, k)

    query_embedding = embed_query(query)
    hits = search(query_embedding, repo_id, top_k=k, filter_metadata=filter_metadata)

    chunks = [RetrievedChunk(h) for h in hits]
    chunks.sort(key=lambda c: c.distance)

    logger.info("Retrieved %d chunks (best score: %.4f)",
                len(chunks),
                chunks[0].relevance_score if chunks else 0.0)

    return chunks
