"""
Router tests: quota awareness, failover, and honest degradation.

This is what makes a $0 budget workable. No single free tier can carry the
service -- measured limits put Groq's free `gpt-oss-120b` at 200K
tokens/day and Gemini 2.5 Flash at 250 requests/day -- but four behind a
router that tracks each one's remaining allowance can.
"""

from __future__ import annotations

import httpx
import pytest

from app.llm.breaker import BreakerRegistry, State
from app.llm.budget import Limits, current_usage, record
from app.llm.providers import Gemini, OpenAICompatible
from app.llm.router import LLMRouter, estimate_tokens
from app.llm.types import (
    AllProvidersFailed,
    Completion,
    ContextTooLong,
    ProviderUnavailable,
    RateLimited,
    Usage,
    system,
    user,
)

MESSAGES = [system("You are a code assistant."), user("How does auth work?")]


class FakeProvider:
    """Scripted provider: each call pops the next outcome."""

    def __init__(self, name, outcomes, model="m", limits=None):
        self.name = name
        self.model = model
        self.limits = limits or Limits()
        self.outcomes = list(outcomes)
        self.calls = 0

    async def complete(self, messages, *, max_tokens=1024, temperature=0.1):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def ok(name, inp=100, out=50) -> Completion:
    return Completion(
        text="answer", provider=name, model="m",
        usage=Usage(input_tokens=inp, output_tokens=out), latency_ms=12.0,
    )


# --- happy path ------------------------------------------------------------

async def test_first_healthy_provider_answers(async_session):
    a = FakeProvider("cerebras", [ok("cerebras")])
    b = FakeProvider("groq", [ok("groq")])
    completion, trace = await LLMRouter([a, b]).complete(async_session, MESSAGES)

    assert completion.provider == "cerebras"
    assert not completion.degraded
    assert b.calls == 0, "the second provider must not be called"
    assert trace.chosen == "cerebras"


async def test_usage_is_recorded_from_the_providers_own_numbers(async_session):
    a = FakeProvider("groq", [ok("groq", inp=321, out=123)])
    await LLMRouter([a]).complete(async_session, MESSAGES)

    usage = await current_usage(async_session, "groq", "m")
    assert usage.day_requests == 1
    assert usage.day_tokens == 321 + 123, "record measured usage, not the estimate"


# --- failover --------------------------------------------------------------

async def test_failover_on_rate_limit(async_session):
    a = FakeProvider("cerebras", [RateLimited("cerebras: 429")])
    b = FakeProvider("groq", [ok("groq")])
    completion, trace = await LLMRouter([a, b]).complete(async_session, MESSAGES)

    assert completion.provider == "groq"
    assert completion.degraded
    assert completion.fallbacks == ["cerebras"]
    assert "cerebras:error" in trace.summary()


async def test_a_rate_limited_provider_does_not_trip_its_breaker(async_session):
    """Full is not broken. Tripping here would lock out a healthy provider."""
    breakers = BreakerRegistry(threshold=1)
    a = FakeProvider("cerebras", [RateLimited("429")])
    b = FakeProvider("groq", [ok("groq")])
    await LLMRouter([a, b], breakers).complete(async_session, MESSAGES)

    assert breakers.get("cerebras").state() is State.CLOSED


async def test_an_unavailable_provider_does_trip_its_breaker(async_session):
    breakers = BreakerRegistry(threshold=1)
    a = FakeProvider("cerebras", [ProviderUnavailable("503")])
    b = FakeProvider("groq", [ok("groq")])
    await LLMRouter([a, b], breakers).complete(async_session, MESSAGES)

    assert breakers.get("cerebras").state() is not State.CLOSED


async def test_an_open_breaker_skips_the_provider_without_calling_it(async_session):
    breakers = BreakerRegistry(threshold=1, cooldown=999)
    breakers.get("cerebras").record_failure("down")

    a = FakeProvider("cerebras", [ok("cerebras")])
    b = FakeProvider("groq", [ok("groq")])
    completion, trace = await LLMRouter([a, b], breakers).complete(async_session, MESSAGES)

    assert a.calls == 0, "an open circuit must cost nothing"
    assert completion.provider == "groq"
    assert trace.attempts[0].outcome == "skipped_breaker"


# --- quota awareness -------------------------------------------------------

async def test_a_provider_over_its_daily_token_budget_is_skipped(async_session):
    """Route away before the 429, not after it."""
    limits = Limits(tokens_per_day=1_000)
    await record(async_session, "groq", "m", requests=1, input_tokens=990)
    await async_session.commit()

    a = FakeProvider("groq", [ok("groq")], limits=limits)
    b = FakeProvider("gemini", [ok("gemini")])
    completion, trace = await LLMRouter([a, b]).complete(async_session, MESSAGES)

    assert a.calls == 0
    assert completion.provider == "gemini"
    assert trace.attempts[0].outcome == "skipped_quota"
    assert "tokens" in trace.attempts[0].detail


async def test_request_per_minute_limit_is_respected(async_session):
    limits = Limits(requests_per_minute=2)
    for _ in range(2):
        await record(async_session, "groq", "m", requests=1)
    await async_session.commit()

    a = FakeProvider("groq", [ok("groq")], limits=limits)
    b = FakeProvider("gemini", [ok("gemini")])
    completion, _ = await LLMRouter([a, b]).complete(async_session, MESSAGES)
    assert completion.provider == "gemini"
    assert a.calls == 0


async def test_free_tier_context_cap_is_enforced_before_the_call(async_session):
    """Cerebras caps free-tier context at 8,192 whatever the model's window."""
    limits = Limits(max_context_tokens=100)
    a = FakeProvider("cerebras", [ok("cerebras")], limits=limits)
    b = FakeProvider("groq", [ok("groq")])

    completion, trace = await LLMRouter([a, b]).complete(
        async_session, [user("x" * 20_000)]
    )

    assert a.calls == 0
    assert completion.provider == "groq"
    assert "context cap" in trace.attempts[0].detail


# --- non-retryable ---------------------------------------------------------

async def test_context_too_long_is_raised_not_failed_over(async_session):
    """
    No other provider does better with the same oversized prompt. Failing
    over would burn every provider's quota on a request that cannot work.
    """
    a = FakeProvider("cerebras", [ContextTooLong("too long")])
    b = FakeProvider("groq", [ok("groq")])

    with pytest.raises(ContextTooLong):
        await LLMRouter([a, b]).complete(async_session, MESSAGES)
    assert b.calls == 0


# --- exhaustion ------------------------------------------------------------

async def test_all_providers_failing_reports_every_reason(async_session):
    a = FakeProvider("cerebras", [ProviderUnavailable("503")])
    b = FakeProvider("groq", [RateLimited("429 daily")])

    with pytest.raises(AllProvidersFailed) as exc:
        await LLMRouter([a, b]).complete(async_session, MESSAGES)

    assert set(exc.value.failures) == {"cerebras", "groq"}
    assert "503" in exc.value.failures["cerebras"]
    assert "429" in exc.value.failures["groq"]


async def test_no_configured_provider_is_an_explicit_failure(async_session):
    with pytest.raises(AllProvidersFailed, match="no provider"):
        await LLMRouter([]).complete(async_session, MESSAGES)


async def test_only_filter_restricts_candidates(async_session):
    a = FakeProvider("cerebras", [ok("cerebras")])
    b = FakeProvider("groq", [ok("groq")])
    completion, _ = await LLMRouter([a, b]).complete(
        async_session, MESSAGES, only=["groq"]
    )
    assert completion.provider == "groq"
    assert a.calls == 0


# --- estimation ------------------------------------------------------------

def test_estimate_grows_with_content():
    assert 0 < estimate_tokens([user("hi")]) < estimate_tokens([user("word " * 500)])


# --- provider wire format --------------------------------------------------

def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_openai_compatible_parses_a_normal_response():
    async def handler(request):
        return httpx.Response(200, json={
            "model": "m",
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        })

    p = OpenAICompatible(name="groq", model="m", base_url="https://x/v1",
                         api_key="k", client=_client(handler))
    got = await p.complete(MESSAGES)
    assert got.text == "hello"
    assert got.usage.input_tokens == 11
    assert got.usage.output_tokens == 3


@pytest.mark.parametrize("status,body,expected", [
    (429, "rate limit exceeded", RateLimited),
    (400, "maximum context length is 8192 tokens", ContextTooLong),
    (413, "string too long", ContextTooLong),
    (500, "internal error", ProviderUnavailable),
    (503, "overloaded", ProviderUnavailable),
    (401, "invalid api key", ProviderUnavailable),
    (400, "malformed json", ProviderUnavailable),
])
async def test_openai_compatible_error_mapping(status, body, expected):
    async def handler(request):
        return httpx.Response(status, text=body)

    p = OpenAICompatible(name="groq", model="m", base_url="https://x/v1",
                         api_key="k", client=_client(handler))
    with pytest.raises(expected):
        await p.complete(MESSAGES)


async def test_rate_limited_carries_retry_after():
    async def handler(request):
        return httpx.Response(429, text="slow down", headers={"retry-after": "42"})

    p = OpenAICompatible(name="groq", model="m", base_url="https://x/v1",
                         api_key="k", client=_client(handler))
    with pytest.raises(RateLimited) as exc:
        await p.complete(MESSAGES)
    assert exc.value.retry_after == 42.0


async def test_malformed_response_shape_is_an_error_not_a_crash():
    async def handler(request):
        return httpx.Response(200, json={"unexpected": True})

    p = OpenAICompatible(name="groq", model="m", base_url="https://x/v1",
                         api_key="k", client=_client(handler))
    with pytest.raises(ProviderUnavailable, match="unexpected response shape"):
        await p.complete(MESSAGES)


async def test_gemini_sends_the_system_prompt_as_a_separate_field():
    captured: dict = {}

    async def handler(request):
        import json

        captured.update(json.loads(request.read()))
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "hi"}]},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2},
        })

    got = await Gemini(api_key="k", client=_client(handler)).complete(MESSAGES)
    assert got.text == "hi"
    assert "systemInstruction" in captured
    assert all(c["role"] in {"user", "model"} for c in captured["contents"])
    assert got.usage.input_tokens == 7


async def test_gemini_blocked_prompt_is_an_error_not_an_empty_answer():
    """A safety block returns 200 with no candidates; "" would look like silence."""

    async def handler(request):
        return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})

    with pytest.raises(ProviderUnavailable, match="SAFETY"):
        await Gemini(api_key="k", client=_client(handler)).complete(MESSAGES)
