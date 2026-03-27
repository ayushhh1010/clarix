"""
Prompt builder — constructs LLM prompts with retrieved code context.
Manages context window budget to stay within token limits.
"""

import logging
from typing import Sequence

from app.rag.retriever import RetrievedChunk
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are an expert AI code assistant. You have deep knowledge of software engineering, \
architecture patterns, debugging, and code quality.

You are analyzing a codebase that has been ingested and indexed. Below you are given relevant code \
snippets retrieved from the codebase based on the user's question.

Instructions:
- Answer the user's question thoroughly using the provided code context.
- Reference specific files, functions, and line numbers when relevant.
- If the code context is insufficient, say so honestly and explain what additional information you'd need.
- Provide actionable, production-quality advice.
- Use markdown formatting with code blocks for any code snippets.
- When suggesting code changes, show exact diffs or replacement code.

IMPORTANT: Only use information from the provided code context. Do not hallucinate code that isn't shown."""

CONTEXT_HEADER = "\n\n## Retrieved Code Context\n\n"
CHUNK_TEMPLATE = """### {file_path} (L{start_line}–L{end_line}) [{chunk_type}]{name_part}
```{language}
{content}
```
Relevance: {score}

"""

CONVERSATION_HEADER = "\n\n## Conversation History\n\n"


def _estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token for code."""
    return len(text) // 4


def build_context_string(
    chunks: Sequence[RetrievedChunk],
    max_tokens: int | None = None,
) -> str:
    """Build the code context string from retrieved chunks, respecting token budget."""
    budget = max_tokens or settings.rag_context_max_tokens
    context_parts: list[str] = []
    used_tokens = 0

    for chunk in chunks:
        name_part = f" — {chunk.name}" if chunk.name else ""
        block = CHUNK_TEMPLATE.format(
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            chunk_type=chunk.chunk_type,
            name_part=name_part,
            language=chunk.language,
            content=chunk.content,
            score=f"{chunk.relevance_score:.4f}",
        )
        block_tokens = _estimate_tokens(block)
        if used_tokens + block_tokens > budget:
            logger.info("Context budget reached: %d / %d tokens, %d chunks included",
                        used_tokens, budget, len(context_parts))
            break
        context_parts.append(block)
        used_tokens += block_tokens

    return CONTEXT_HEADER + "".join(context_parts) if context_parts else ""


def build_messages(
    user_query: str,
    chunks: Sequence[RetrievedChunk],
    conversation_history: list[dict] | None = None,
    additional_context: str = "",
) -> list[dict]:
    """
    Build a full message list for the LLM:
    [system, ...history, user_with_context]
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history (trimmed to last 10 exchanges)
    if conversation_history:
        recent = conversation_history[-20:]  # last 10 user+assistant pairs
        messages.extend(recent)

    # Build user message with code context
    context_str = build_context_string(chunks)
    user_content = user_query
    if context_str:
        user_content = context_str + "\n\n## User Question\n\n" + user_query
    if additional_context:
        user_content += "\n\n## Additional Context\n\n" + additional_context

    messages.append({"role": "user", "content": user_content})

    logger.debug("Built %d messages, total ~%d tokens",
                 len(messages),
                 sum(_estimate_tokens(m["content"]) for m in messages))

    return messages
