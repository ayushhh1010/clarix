"""
Quota-aware routing with failover across free-tier providers.

This is the component that makes a $0 budget an engineering position rather
than a constraint. No single free tier can carry the service -- measured
limits put Groq's `gpt-oss-120b` at 200K tokens/day and Gemini 2.5 Flash at
250 requests/day -- but four of them behind a router that knows each one's
remaining allowance can.

Order of checks per candidate, cheapest first:

  1. circuit breaker   in-process, free, catches providers that are down
  2. quota             one indexed query, catches providers that will 429
  3. the request       the only expensive step

Failure handling distinguishes three cases, because they want different
things:

  RateLimited / quota     try the next provider immediately; this one is
                          fine, just full. Does NOT trip the breaker -- a
                          provider at its daily limit is not broken.
  ProviderUnavailable     try the next provider AND trip the breaker.
  ContextTooLong          no other provider will do better with the same
                          oversized prompt, so it is raised to the caller,
                          who can drop context and retry.

Degradation is explicit. When every provider is exhausted the caller gets
`AllProvidersFailed` carrying each provider's reason, so the API can return
retrieved chunks with citations and no prose rather than a bare 500 -- a
worse answer beats no answer, but only if the caller can tell the
difference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.breaker import BreakerRegistry
from app.llm.budget import check as quota_check
from app.llm.budget import record as record_usage
from app.llm.providers import Provider
from app.llm.types import (
    AllProvidersFailed,
    Completion,
    ContextTooLong,
    LLMError,
    Message,
    ProviderUnavailable,
    RateLimited,
)

logger = logging.getLogger(__name__)

# Output tokens are unknown before the call, so the quota check reserves
# this much on top of the measured input. Reserving nothing would let a
# burst of concurrent requests sail past a limit together.
DEFAULT_OUTPUT_RESERVE = 800

# Rough characters-per-token for the pre-flight estimate only. The real
# tokenizer is an indexer dependency and is not imported by the API
# process; this number is used solely to decide *which* provider to try,
# and the provider's own reported usage is what gets recorded afterwards.
CHARS_PER_TOKEN = 3.5


def estimate_tokens(messages: list[Message]) -> int:
    chars = sum(len(m.content) for m in messages)
    return int(chars / CHARS_PER_TOKEN) + 8 * len(messages)


@dataclass
class RouteAttempt:
    provider: str
    outcome: str  # skipped_breaker | skipped_quota | error | ok
    detail: str = ""
    latency_ms: float = 0.0


@dataclass
class RouteTrace:
    attempts: list[RouteAttempt] = field(default_factory=list)
    estimated_tokens: int = 0

    @property
    def chosen(self) -> str | None:
        for attempt in self.attempts:
            if attempt.outcome == "ok":
                return attempt.provider
        return None

    def summary(self) -> str:
        return " -> ".join(f"{a.provider}:{a.outcome}" for a in self.attempts)


class LLMRouter:
    """
    Tries providers in order until one answers.

    Order is the caller's: put the fastest or highest-quality provider
    first and the rest behind it. There is no scoring here -- a scorer
    would need per-provider quality data we do not have, and ordering a
    short list by hand is honest about that.
    """

    def __init__(
        self,
        providers: list[Provider],
        breakers: BreakerRegistry | None = None,
        output_reserve: int = DEFAULT_OUTPUT_RESERVE,
    ):
        self.providers = providers
        self.breakers = breakers or BreakerRegistry()
        self.output_reserve = output_reserve

    async def complete(
        self,
        db: AsyncSession,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        only: list[str] | None = None,
    ) -> tuple[Completion, RouteTrace]:
        if not self.providers:
            raise AllProvidersFailed({"(none)": "no provider is configured"})

        estimate = estimate_tokens(messages) + min(max_tokens, self.output_reserve)
        trace = RouteTrace(estimated_tokens=estimate)
        failures: dict[str, str] = {}
        candidates = [
            p for p in self.providers if only is None or p.name in only
        ]

        for provider in candidates:
            breaker = self.breakers.get(provider.name)
            if not breaker.allows():
                reason = f"circuit open ({breaker.last_error or 'recent failures'})"
                trace.attempts.append(
                    RouteAttempt(provider.name, "skipped_breaker", reason)
                )
                failures[provider.name] = reason
                continue

            verdict = await quota_check(
                db, provider.name, provider.model, provider.limits, estimate
            )
            if not verdict:
                trace.attempts.append(
                    RouteAttempt(provider.name, "skipped_quota", verdict.reason)
                )
                failures[provider.name] = verdict.reason
                continue

            try:
                completion = await provider.complete(
                    messages, max_tokens=max_tokens, temperature=temperature
                )
            except ContextTooLong:
                # No other provider will do better with the same oversized
                # prompt. Raising lets the caller shrink the context, which
                # is the only thing that actually helps.
                trace.attempts.append(
                    RouteAttempt(provider.name, "error", "context too long")
                )
                await record_usage(db, provider.name, provider.model, errors=1)
                await db.commit()
                raise
            except RateLimited as exc:
                # Full, not broken: do not trip the breaker.
                trace.attempts.append(RouteAttempt(provider.name, "error", str(exc)[:200]))
                failures[provider.name] = str(exc)[:200]
                await record_usage(db, provider.name, provider.model, requests=1, errors=1)
                await db.commit()
                continue
            except (ProviderUnavailable, LLMError) as exc:
                breaker.record_failure(str(exc))
                trace.attempts.append(RouteAttempt(provider.name, "error", str(exc)[:200]))
                failures[provider.name] = str(exc)[:200]
                await record_usage(db, provider.name, provider.model, requests=1, errors=1)
                await db.commit()
                continue

            breaker.record_success()
            # Record the provider's own reported usage, not the estimate.
            await record_usage(
                db, provider.name, provider.model,
                requests=1,
                input_tokens=completion.usage.input_tokens,
                output_tokens=completion.usage.output_tokens,
            )
            await db.commit()

            completion.fallbacks = [
                a.provider for a in trace.attempts if a.outcome != "ok"
            ]
            trace.attempts.append(
                RouteAttempt(provider.name, "ok", latency_ms=completion.latency_ms)
            )
            if completion.degraded:
                logger.warning(
                    "served by %s after %s", provider.name, trace.summary()
                )
            return completion, trace

        logger.error("all providers failed: %s", trace.summary())
        raise AllProvidersFailed(failures)

    def health(self) -> dict:
        return {
            "providers": [
                {"name": p.name, "model": p.model} for p in self.providers
            ],
            "breakers": self.breakers.snapshot(),
        }
