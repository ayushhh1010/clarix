"""
Retrieval Agent — fetches relevant code chunks from the vector database using RAG.
"""

import logging
from typing import Any

from app.agents.state import AgentState
from app.rag.retriever import retrieve_context

logger = logging.getLogger(__name__)


async def retrieval_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Retrieval Agent.
    Embeds the user query and retrieves the most relevant code chunks.
    """
    query = state["user_query"]
    repo_id = state["repo_id"]

    logger.info("[Retrieval] Searching for: %.100s... (repo=%s)", query, repo_id)

    # Perform semantic retrieval
    chunks = retrieve_context(query, repo_id, top_k=10)

    # Convert to serializable dicts
    context_dicts = []
    for chunk in chunks:
        context_dicts.append({
            "content": chunk.content,
            "file_path": chunk.file_path,
            "language": chunk.language,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "chunk_type": chunk.chunk_type,
            "name": chunk.name,
            "relevance_score": chunk.relevance_score,
        })

    step_log = {
        "agent": "retrieval",
        "chunks_found": len(context_dicts),
        "top_files": list({c["file_path"] for c in context_dicts[:5]}),
    }

    logger.info("[Retrieval] Found %d relevant chunks", len(context_dicts))

    return {
        "retrieved_context": context_dicts,
        "current_step": state.get("current_step", 0) + 1,
        "steps_log": [step_log],
    }
