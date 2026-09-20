"""Provider-neutral LLM access: quota-aware routing with failover."""

from app.llm.breaker import BreakerRegistry
from app.llm.providers import build_providers
from app.llm.router import LLMRouter, RouteTrace
from app.llm.types import (
    AllProvidersFailed,
    Completion,
    ContextTooLong,
    LLMError,
    Message,
    ProviderUnavailable,
    RateLimited,
    assistant,
    system,
    user,
)

__all__ = [
    "AllProvidersFailed",
    "BreakerRegistry",
    "Completion",
    "ContextTooLong",
    "LLMError",
    "LLMRouter",
    "Message",
    "ProviderUnavailable",
    "RateLimited",
    "RouteTrace",
    "assistant",
    "build_providers",
    "system",
    "user",
]
