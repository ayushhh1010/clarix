"""
Token counting.

The v1 pipeline estimated tokens as `len(text) // 4`. That heuristic is
calibrated on English prose; source code tokenizes considerably worse
(punctuation, identifiers with underscores, indentation runs), so the estimate
runs low -- which means a "12,000 token" context can overflow the real budget.

Free-tier inference makes this a correctness issue rather than a tuning issue:
Cerebras caps free-tier context at 8,192 tokens and Groq's free `gpt-oss-120b`
allows 8K tokens/minute, so a single under-estimated request fails outright.

`bench/bench_token_estimate.py` measures the heuristic's error against the real
tokenizer on this repository's own source.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Protocol

logger = logging.getLogger(__name__)

# Tokenizer used for budgeting. This is the embedding model's tokenizer, which
# is what determines whether a chunk fits the encoder's 8,192-token window.
# Generation-side budgets are approximated with the same counter: exactness is
# not required there, only that we never under-count.
DEFAULT_TOKENIZER = "jinaai/jina-embeddings-v2-base-code"

# Encoder context limit for the above model.
ENCODER_MAX_TOKENS = 8192


class TokenCounter(Protocol):
    """Anything that can report a token count for a string."""

    def count(self, text: str) -> int: ...

    @property
    def name(self) -> str: ...


class HeuristicTokenCounter:
    """
    The v1 estimator: four characters per token.

    Retained only so benchmarks can quantify its error. Do not use it for
    budgeting.
    """

    name = "heuristic:len//4"

    def count(self, text: str) -> int:
        return len(text) // 4


class HFTokenCounter:
    """
    Real tokenizer, loaded via `tokenizers` (Rust; no torch, no transformers).

    Only the tokenizer.json is fetched -- a few megabytes, not the model
    weights. Counting is done without special tokens; callers that care about
    the two-token overhead of [CLS]/[SEP] should add it explicitly.
    """

    def __init__(self, model_id: str = DEFAULT_TOKENIZER):
        from tokenizers import Tokenizer  # imported lazily: indexer-only dep

        self._model_id = model_id
        self._tok = Tokenizer.from_pretrained(model_id)
        # No truncation: we need true counts to decide *whether* to split.
        self._tok.no_truncation()
        self._tok.no_padding()

    @property
    def name(self) -> str:
        return f"hf:{self._model_id}"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tok.encode(text, add_special_tokens=False).ids)


@lru_cache(maxsize=4)
def get_token_counter(model_id: str = DEFAULT_TOKENIZER) -> TokenCounter:
    """
    Return the real tokenizer, falling back to the heuristic if it cannot be
    loaded (no network on first run, `tokenizers` absent in the serving image).

    The fallback is logged at WARNING because it silently degrades budget
    accuracy, and a silently wrong budget is how you get 429s in production.
    """
    try:
        return HFTokenCounter(model_id)
    except Exception as exc:  # noqa: BLE001 - any failure means fall back
        logger.warning(
            "Real tokenizer unavailable (%s: %s); falling back to the len//4 "
            "heuristic. Token budgets will be approximate and biased low.",
            type(exc).__name__,
            exc,
        )
        return HeuristicTokenCounter()
