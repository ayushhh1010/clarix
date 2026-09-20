"""
Per-provider quota tracking, so the router can route *away* from a provider
before it starts refusing.

Free tiers bind on tokens, not requests, and the limits are small enough
that a single unbudgeted query can exhaust a day's allowance. Measured from
each provider's published limits (verified 2026-09; they move, so they are
configuration, not constants in code):

    Groq          30 RPM; gpt-oss-120b 1K RPD, 8K TPM, 200K TPD
    Cerebras      1M tokens/day, 30 RPM, ~60K TPM, 8,192-token context cap
    Gemini Flash  10 RPM / 250 RPD; Flash-Lite 15 RPM / 1,000 RPD
    OpenRouter    20 RPM, 50 RPD below $10 lifetime spend

Counters live in Postgres rather than Redis. At the traffic these limits
permit -- a few hundred requests a day across every provider -- an atomic
`INSERT ... ON CONFLICT DO UPDATE` is orders of magnitude faster than
required, and it removes a service. The audit that preceded this found v1's
Redis held only a facts set nothing ever wrote to.

Reservations are approximate on purpose. Input tokens are known before the
call; output tokens are not, so a conservative estimate is reserved up
front and reconciled with the provider's reported usage afterwards.
Reserving nothing would let a burst of concurrent requests sail past the
limit together.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Limits:
    """Published free-tier limits for one provider/model pair."""

    requests_per_minute: int | None = None
    requests_per_day: int | None = None
    tokens_per_minute: int | None = None
    tokens_per_day: int | None = None
    # Hard ceiling on a single request's context. Cerebras caps free-tier
    # context at 8,192 regardless of the model's nominal window.
    max_context_tokens: int | None = None

    def headroom_fraction(self, used_day_tokens: int) -> float:
        if not self.tokens_per_day:
            return 1.0
        return max(0.0, 1.0 - used_day_tokens / self.tokens_per_day)


def minute_window(now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    return now.replace(second=0, microsecond=0)


def day_window(now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


_UPSERT = text(
    """
    INSERT INTO provider_usage (
        provider, model, window_kind, window_start,
        requests, input_tokens, output_tokens, errors
    )
    VALUES (:provider, :model, :window_kind, :window_start,
            :requests, :input_tokens, :output_tokens, :errors)
    ON CONFLICT (provider, model, window_kind, window_start) DO UPDATE SET
        requests = provider_usage.requests + EXCLUDED.requests,
        input_tokens = provider_usage.input_tokens + EXCLUDED.input_tokens,
        output_tokens = provider_usage.output_tokens + EXCLUDED.output_tokens,
        errors = provider_usage.errors + EXCLUDED.errors,
        updated_at = now()
    RETURNING requests, input_tokens + output_tokens AS tokens
    """
)


@dataclass
class Usage:
    minute_requests: int = 0
    minute_tokens: int = 0
    day_requests: int = 0
    day_tokens: int = 0


async def record(
    db: AsyncSession,
    provider: str,
    model: str,
    *,
    requests: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    errors: int = 0,
    now: datetime | None = None,
) -> None:
    """Add to both the minute and day counters."""
    for kind, start in (("minute", minute_window(now)), ("day", day_window(now))):
        await db.execute(
            _UPSERT,
            {
                "provider": provider, "model": model,
                "window_kind": kind, "window_start": start,
                "requests": requests, "input_tokens": input_tokens,
                "output_tokens": output_tokens, "errors": errors,
            },
        )


async def current_usage(
    db: AsyncSession, provider: str, model: str, now: datetime | None = None
) -> Usage:
    rows = await db.execute(
        text(
            "SELECT window_kind, requests, input_tokens + output_tokens AS tokens "
            "FROM provider_usage "
            "WHERE provider = :p AND model = :m "
            "  AND ((window_kind = 'minute' AND window_start = :minute) "
            "    OR (window_kind = 'day' AND window_start = :day))"
        ),
        {"p": provider, "m": model,
         "minute": minute_window(now), "day": day_window(now)},
    )
    usage = Usage()
    for row in rows:
        if row.window_kind == "minute":
            usage.minute_requests, usage.minute_tokens = row.requests, row.tokens
        else:
            usage.day_requests, usage.day_tokens = row.requests, row.tokens
    return usage


@dataclass
class Verdict:
    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


async def check(
    db: AsyncSession,
    provider: str,
    model: str,
    limits: Limits,
    estimated_tokens: int,
    now: datetime | None = None,
) -> Verdict:
    """
    Whether a request of roughly `estimated_tokens` fits the remaining
    allowance.

    The estimate is included in the comparison rather than checked after
    the fact: a provider with 500 tokens left should refuse a 4,000-token
    request, not accept it and 429 mid-stream.
    """
    if limits.max_context_tokens and estimated_tokens > limits.max_context_tokens:
        return Verdict(
            False,
            f"request is ~{estimated_tokens} tokens, over {provider}'s "
            f"{limits.max_context_tokens}-token free-tier context cap",
        )

    usage = await current_usage(db, provider, model, now)

    if limits.requests_per_minute and usage.minute_requests >= limits.requests_per_minute:
        return Verdict(False, f"{provider}: {usage.minute_requests} requests this minute")
    if limits.requests_per_day and usage.day_requests >= limits.requests_per_day:
        return Verdict(False, f"{provider}: {usage.day_requests} requests today")
    if limits.tokens_per_minute and usage.minute_tokens + estimated_tokens > limits.tokens_per_minute:
        return Verdict(
            False,
            f"{provider}: {usage.minute_tokens} + ~{estimated_tokens} tokens "
            f"exceeds {limits.tokens_per_minute}/min",
        )
    if limits.tokens_per_day and usage.day_tokens + estimated_tokens > limits.tokens_per_day:
        return Verdict(
            False,
            f"{provider}: {usage.day_tokens} + ~{estimated_tokens} tokens "
            f"exceeds {limits.tokens_per_day}/day",
        )
    return Verdict(True)


async def prune(db: AsyncSession, older_than: timedelta = timedelta(days=8)) -> int:
    """
    Drop counter rows outside the retention window.

    A minute-granularity counter accumulates 1,440 rows per provider per
    day; unpruned, the table outgrows the data it exists to protect.
    """
    result = await db.execute(
        text(
            "DELETE FROM provider_usage "
            "WHERE window_start < now() - make_interval(secs => :secs) "
            "RETURNING provider"
        ),
        {"secs": float(older_than.total_seconds())},
    )
    return len(result.fetchall())
