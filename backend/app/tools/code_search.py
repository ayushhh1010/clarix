"""
Code Search Tool — semantic search over the codebase using the vector database.
"""

from langchain_core.tools import tool, BaseTool
from app.rag.retriever import retrieve_context


def create_code_search_tool(repo_id: str) -> BaseTool:
    """Create a code search tool bound to a specific repo."""

    @tool
    def search_code(query: str) -> str:
        """Search the codebase semantically for code related to a natural language query.
        Use this when you need to find code related to a concept, feature, bug, or pattern.

        Args:
            query: Natural language description of what code to search for.

        Returns:
            Formatted string of relevant code chunks with file paths and line numbers.
        """
        chunks = retrieve_context(query, repo_id, top_k=8)

        if not chunks:
            return "No relevant code found for the query."

        results: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            name_info = f" ({chunk.name})" if chunk.name else ""
            header = (
                f"### Result {i}: {chunk.file_path}{name_info}\n"
                f"Lines {chunk.start_line}–{chunk.end_line} | "
                f"{chunk.chunk_type} | Relevance: {chunk.relevance_score:.2%}"
            )
            results.append(f"{header}\n```{chunk.language}\n{chunk.content}\n```\n")

        return "\n".join(results)

    return search_code
