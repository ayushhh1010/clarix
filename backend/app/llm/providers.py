"""
Provider clients.

Three of the four free providers speak the OpenAI chat-completions shape
(Groq, Cerebras, OpenRouter), so they share one client that differs only in
base URL, key and limits. Gemini has its own request and response shape and
gets its own.

Written against `httpx` directly rather than each vendor's SDK. Four SDKs
would be four dependency trees in the serving image to send the same JSON,
and the failure mapping below -- which is the part that actually matters --
has to be written per-provider regardless.

Error mapping is the point of this module. A 429 and a 500 need different
handling (route elsewhere now vs. trip the breaker), and a context-length
400 is actionable where other 400s are not. Providers signal these
inconsistently, so the mapping is explicit and tested.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from app.llm.budget import Limits
from app.llm.types import (
    Completion,
    ContextTooLong,
    Message,
    ProviderUnavailable,
    RateLimited,
    Usage,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60.0

# Substrings that identify a context-length rejection. Providers report it
# as an ordinary 400 with prose, so there is nothing structured to match on.
_CONTEXT_MARKERS = (
    "context length", "context_length", "maximum context", "too long",
    "reduce the length", "tokens per request", "context window",
    "string too long", "exceeds the maximum",
)


class Provider(Protocol):
    name: str
    model: str
    limits: Limits

    async def complete(
        self, messages: list[Message], *, max_tokens: int, temperature: float
    ) -> Completion: ...


def _is_context_error(body: str) -> bool:
    lowered = body.lower()
    return any(marker in lowered for marker in _CONTEXT_MARKERS)


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


@dataclass
class OpenAICompatible:
    """
    Groq, Cerebras and OpenRouter.

    They differ in base URL, key and published limits; the wire format is
    the same, so duplicating three near-identical clients would only
    triplicate the error mapping.
    """

    name: str
    model: str
    base_url: str
    api_key: str
    limits: Limits = field(default_factory=Limits)
    timeout: float = DEFAULT_TIMEOUT
    extra_headers: dict[str, str] = field(default_factory=dict)
    client: httpx.AsyncClient | None = None

    async def complete(
        self, messages: list[Message], *, max_tokens: int = 1024, temperature: float = 0.1
    ) -> Completion:
        payload = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        owns = self.client is None
        client = self.client or httpx.AsyncClient(timeout=self.timeout)
        started = time.perf_counter()
        try:
            response = await client.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                json=payload, headers=headers,
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"{self.name}: {type(exc).__name__}: {exc}") from exc
        finally:
            if owns:
                await client.aclose()

        elapsed = (time.perf_counter() - started) * 1000
        self._raise_for_status(response)

        data = response.json()
        try:
            choice = data["choices"][0]
            text_out = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderUnavailable(
                f"{self.name}: unexpected response shape: {str(data)[:200]}"
            ) from exc

        raw_usage = data.get("usage") or {}
        return Completion(
            text=text_out,
            provider=self.name,
            model=data.get("model", self.model),
            usage=Usage(
                input_tokens=int(raw_usage.get("prompt_tokens", 0)),
                output_tokens=int(raw_usage.get("completion_tokens", 0)),
            ),
            latency_ms=elapsed,
            finish_reason=choice.get("finish_reason") or "stop",
        )

    def _raise_for_status(self, response: httpx.Response) -> None:
        if 200 <= response.status_code < 300:
            return
        body = response.text[:600]
        if response.status_code == 429:
            raise RateLimited(f"{self.name}: rate limited: {body}", _retry_after(response))
        if response.status_code in (400, 413, 422) and _is_context_error(body):
            raise ContextTooLong(f"{self.name}: {body}")
        if response.status_code in (401, 403):
            # Not retryable and not the next provider's problem, but it must
            # not look like an outage either -- a bad key should be obvious.
            raise ProviderUnavailable(f"{self.name}: authentication failed ({body})")
        if response.status_code >= 500:
            raise ProviderUnavailable(f"{self.name}: upstream {response.status_code}: {body}")
        raise ProviderUnavailable(f"{self.name}: HTTP {response.status_code}: {body}")


@dataclass
class Gemini:
    """
    Google AI Studio (`generativelanguage.googleapis.com`).

    Different request shape: the system prompt is a separate field rather
    than a message, roles are `user`/`model`, and content is a list of
    parts.
    """

    name: str = "gemini"
    model: str = "gemini-2.5-flash-lite"
    api_key: str = ""
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    limits: Limits = field(default_factory=Limits)
    timeout: float = DEFAULT_TIMEOUT
    client: httpx.AsyncClient | None = None

    async def complete(
        self, messages: list[Message], *, max_tokens: int = 1024, temperature: float = 0.1
    ) -> Completion:
        system_parts = [m.content for m in messages if m.role.value == "system"]
        contents = [
            {
                "role": "model" if m.role.value == "assistant" else "user",
                "parts": [{"text": m.content}],
            }
            for m in messages
            if m.role.value != "system"
        ]

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

        owns = self.client is None
        client = self.client or httpx.AsyncClient(timeout=self.timeout)
        started = time.perf_counter()
        try:
            response = await client.post(
                f"{self.base_url}/models/{self.model}:generateContent",
                json=payload,
                headers={"Content-Type": "application/json",
                         "x-goog-api-key": self.api_key},
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"{self.name}: {type(exc).__name__}: {exc}") from exc
        finally:
            if owns:
                await client.aclose()

        elapsed = (time.perf_counter() - started) * 1000
        self._raise_for_status(response)

        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            # A prompt blocked by safety filters returns 200 with no
            # candidates. Silently returning "" would look like a model
            # that had nothing to say.
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise ProviderUnavailable(f"{self.name}: empty response ({reason})")

        parts = (candidates[0].get("content") or {}).get("parts") or []
        text_out = "".join(p.get("text", "") for p in parts)
        raw_usage = data.get("usageMetadata") or {}

        return Completion(
            text=text_out,
            provider=self.name,
            model=self.model,
            usage=Usage(
                input_tokens=int(raw_usage.get("promptTokenCount", 0)),
                output_tokens=int(raw_usage.get("candidatesTokenCount", 0)),
            ),
            latency_ms=elapsed,
            finish_reason=candidates[0].get("finishReason", "STOP"),
        )

    def _raise_for_status(self, response: httpx.Response) -> None:
        if 200 <= response.status_code < 300:
            return
        body = response.text[:600]
        if response.status_code == 429:
            raise RateLimited(f"{self.name}: rate limited: {body}", _retry_after(response))
        if response.status_code == 400 and _is_context_error(body):
            raise ContextTooLong(f"{self.name}: {body}")
        if response.status_code in (401, 403):
            raise ProviderUnavailable(f"{self.name}: authentication failed ({body})")
        raise ProviderUnavailable(f"{self.name}: HTTP {response.status_code}: {body}")


# --- published free-tier limits (verified 2026-09) -------------------------
#
# Configuration, not constants: providers change these without notice, and
# a stale limit here means either wasted allowance or avoidable 429s.

GROQ_LIMITS = Limits(
    requests_per_minute=30, requests_per_day=1_000,
    tokens_per_minute=8_000, tokens_per_day=200_000,
)
CEREBRAS_LIMITS = Limits(
    requests_per_minute=30, tokens_per_minute=60_000,
    tokens_per_day=1_000_000,
    # Free tier caps context at 8,192 regardless of the model's window.
    max_context_tokens=8_192,
)
GEMINI_FLASH_LITE_LIMITS = Limits(
    requests_per_minute=15, requests_per_day=1_000, tokens_per_minute=250_000,
)
OPENROUTER_FREE_LIMITS = Limits(
    requests_per_minute=20, requests_per_day=50,
)


def build_providers(settings) -> list[Provider]:
    """
    Construct every provider that has a key configured.

    A provider without a key is omitted rather than constructed and left to
    fail: an absent provider is a smaller candidate list, while a broken
    one is a wasted request and a tripped breaker on every query.
    """
    providers: list[Provider] = []

    if getattr(settings, "cerebras_api_key", ""):
        providers.append(OpenAICompatible(
            name="cerebras",
            model=getattr(settings, "cerebras_model", "llama-3.3-70b"),
            base_url="https://api.cerebras.ai/v1",
            api_key=settings.cerebras_api_key,
            limits=CEREBRAS_LIMITS,
        ))
    if getattr(settings, "groq_api_key", ""):
        providers.append(OpenAICompatible(
            name="groq",
            model=getattr(settings, "groq_model", "openai/gpt-oss-120b"),
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.groq_api_key,
            limits=GROQ_LIMITS,
        ))
    if getattr(settings, "gemini_api_key", ""):
        providers.append(Gemini(
            model=getattr(settings, "gemini_model", "gemini-2.5-flash-lite"),
            api_key=settings.gemini_api_key,
            limits=GEMINI_FLASH_LITE_LIMITS,
        ))
    if getattr(settings, "openrouter_api_key", ""):
        providers.append(OpenAICompatible(
            name="openrouter",
            model=getattr(settings, "openrouter_model", "deepseek/deepseek-chat:free"),
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
            limits=OPENROUTER_FREE_LIMITS,
            extra_headers={"HTTP-Referer": getattr(settings, "frontend_url", "")},
        ))

    if not providers:
        logger.warning(
            "no LLM provider keys configured; retrieval will work but "
            "generation will fail"
        )
    return providers
