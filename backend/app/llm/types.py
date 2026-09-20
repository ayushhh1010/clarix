"""
Provider-neutral types for chat completion.

Deliberately small. The v1 code reached for LangChain to get message
objects and then bypassed its own abstraction anyway -- `planner.py`
imported the private `_get_llm` directly -- which is a reliable sign the
abstraction was not earning its ~100 MB. These four dataclasses are what
the application actually uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def to_openai(self) -> dict:
        return {"role": self.role.value, "content": self.content}


def system(content: str) -> Message:
    return Message(Role.SYSTEM, content)


def user(content: str) -> Message:
    return Message(Role.USER, content)


def assistant(content: str) -> Message:
    return Message(Role.ASSISTANT, content)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Completion:
    """One model response, plus enough provenance to debug and bill it."""

    text: str
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    # Providers that were tried and failed before this one succeeded. Empty
    # on the happy path; populated on failover so a degraded response is
    # visibly degraded rather than silently so.
    fallbacks: list[str] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        return bool(self.fallbacks)


class LLMError(Exception):
    """Base class for provider failures."""


class RateLimited(LLMError):
    """Provider refused for quota reasons. Retryable elsewhere, not here."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ProviderUnavailable(LLMError):
    """Transport or 5xx failure. Retryable."""


class ContextTooLong(LLMError):
    """
    The request exceeded the model's context window.

    Distinguished from other 400s because it is actionable: the caller can
    drop context and retry, whereas a malformed request cannot be fixed by
    retrying. Free tiers make this common -- Cerebras caps free-tier
    context at 8,192 tokens.
    """


class AllProvidersFailed(LLMError):
    """Every candidate was exhausted. Carries what each one said."""

    def __init__(self, failures: dict[str, str]):
        self.failures = failures
        detail = "; ".join(f"{k}: {v}" for k, v in failures.items())
        super().__init__(f"all providers failed ({detail})")
