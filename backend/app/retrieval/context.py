"""
Pack retrieved chunks into a prompt under a hard token budget.

The budget is not a tuning knob here, it is an infrastructure constraint.
Measured free-tier limits: Cerebras caps free-tier context at 8,192 tokens,
and Groq's free `gpt-oss-120b` allows 8,000 tokens per minute -- so a single
over-budget request fails outright rather than costing a little more. v1's
`rag_context_max_tokens` was 12,000, which exceeds both.

Two consequences shape this module:

  Retrieval precision is load-bearing. With room for five or six chunks,
  which five matters more than how many. That is why the router exists and
  why recall@1 was the metric that moved (0.680 -> 0.937 on identifier
  queries, BENCHMARKS.md section 7).

  Counting must be real. v1 estimated `len(text) // 4`, a prose-calibrated
  heuristic that under-counts code, so a "12,000 token" context could
  overflow the actual window. Tokens are counted with the real tokenizer,
  and the count errs high when the tokenizer is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.indexing.tokens import get_token_counter
from app.retrieval.hybrid import RetrievedChunk

logger = logging.getLogger(__name__)

# Total prompt budget. Sits under the tightest free-tier ceiling with room
# for the system prompt, the question and the answer.
DEFAULT_CONTEXT_BUDGET = 4_000

# Reserved for the model's reply; subtracted from the budget before packing.
DEFAULT_OUTPUT_RESERVE = 900

SYSTEM_PROMPT = """You are a code assistant answering questions about one repository.

You are given code retrieved from that repository. Follow these rules:

- Answer only from the provided code. If it does not contain the answer, say
  so plainly and name what would be needed. Do not invent APIs, file paths,
  or behaviour.
- Cite the source of every specific claim as `path:start-end`, using the
  citations given with each excerpt.
- The excerpts are retrieved fragments, not the whole repository. Absence of
  something from the context is not evidence it does not exist.
- Be concise. Prefer showing the relevant code to describing it."""


@dataclass
class PackedContext:
    text: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    used_tokens: int = 0
    budget: int = 0
    dropped: int = 0

    @property
    def citations(self) -> list[dict]:
        return [
            {
                "chunk_id": c.chunk_id,
                "file_path": c.file_path,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "symbol": c.symbol,
                "citation": c.citation,
                "arms": list(c.arms),
            }
            for c in self.chunks
        ]

    @property
    def empty(self) -> bool:
        return not self.chunks


def _excerpt(chunk: RetrievedChunk) -> str:
    where = chunk.citation
    what = f" — {chunk.symbol}" if chunk.symbol else ""
    return f"### {where}{what}\n```{chunk.language}\n{chunk.content}\n```\n"


def pack(
    chunks: list[RetrievedChunk],
    *,
    budget: int = DEFAULT_CONTEXT_BUDGET,
    output_reserve: int = DEFAULT_OUTPUT_RESERVE,
    question_tokens: int = 0,
    history_tokens: int = 0,
) -> PackedContext:
    """
    Fill the remaining budget with the highest-ranked chunks that fit.

    A chunk too large for the remaining room is skipped rather than
    truncated, and the next one is tried: a truncated function is often
    worse than no function, because it looks complete.
    """
    counter = get_token_counter()
    overhead = counter.count(SYSTEM_PROMPT) + question_tokens + history_tokens
    remaining = budget - output_reserve - overhead

    packed = PackedContext(text="", budget=budget)
    if remaining <= 0:
        logger.warning(
            "no context budget left: system+question+history is %d of %d tokens",
            overhead, budget,
        )
        packed.dropped = len(chunks)
        return packed

    parts: list[str] = []
    for chunk in chunks:
        block = _excerpt(chunk)
        cost = counter.count(block)
        if cost > remaining:
            packed.dropped += 1
            continue
        parts.append(block)
        packed.chunks.append(chunk)
        packed.used_tokens += cost
        remaining -= cost

    packed.text = "\n".join(parts)
    logger.info(
        "packed %d/%d chunks, %d tokens of %d budget (%d dropped)",
        len(packed.chunks), len(chunks), packed.used_tokens, budget, packed.dropped,
    )
    return packed


def build_messages(
    question: str,
    packed: PackedContext,
    history: list[tuple[str, str]] | None = None,
) -> list:
    """
    Assemble the message list.

    The system prompt is first and byte-stable across requests, which is
    what makes it cacheable by providers that support prefix caching.
    Volatile content -- retrieved code and the question -- comes after it.
    """
    from app.llm.types import assistant, system, user

    messages = [system(SYSTEM_PROMPT)]
    for role, content in history or []:
        messages.append(user(content) if role == "user" else assistant(content))

    if packed.empty:
        messages.append(
            user(
                f"No code was retrieved for this question.\n\n"
                f"Question: {question}\n\n"
                f"Say that you could not find relevant code and suggest how the "
                f"question might be narrowed. Do not guess at an answer."
            )
        )
    else:
        messages.append(
            user(f"## Retrieved code\n\n{packed.text}\n## Question\n\n{question}")
        )
    return messages
