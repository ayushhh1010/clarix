"""
The wake exists to break a deadlock: a stopped indexer is never started by
the job that needs it, because the only traffic it ever receives is a query
against a repository that is not indexed yet.

So what matters is that queuing work causes a request to the indexer, that
the request goes to the right place whichever form the address was given in,
and that a failed wake never propagates -- the job is already committed, and
an upload must not fail because the indexer was slow to answer.
"""

from __future__ import annotations

import httpx
import pytest

from app.indexing import wake


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        # Render's `fromService` injects a bare origin.
        ("https://clarix-indexer.onrender.com", "https://clarix-indexer.onrender.com/health"),
        # A trailing slash is the same address.
        ("https://clarix-indexer.onrender.com/", "https://clarix-indexer.onrender.com/health"),
        # The same service named by its full embed URL.
        ("https://clarix-indexer.onrender.com/embed", "https://clarix-indexer.onrender.com/health"),
        ("http://localhost:8081/embed", "http://localhost:8081/health"),
        # Nothing configured, and nothing that could be a URL.
        ("", ""),
        ("   ", ""),
        ("not-a-url", ""),
    ],
)
def test_health_url_from_every_configured_form(configured, expected):
    assert wake.health_url(configured) == expected


async def test_wake_requests_the_indexer(monkeypatch):
    called = {}

    def handler(request: httpx.Request) -> httpx.Response:
        called["url"] = str(request.url)
        return httpx.Response(200, json={"status": "healthy"})

    _use(monkeypatch, handler)
    assert await wake.wake_indexer("https://indexer.invalid/embed") is True
    assert called["url"] == "https://indexer.invalid/health"


async def test_unreachable_indexer_does_not_raise(monkeypatch):
    """An upload has already been committed by the time we wake anything."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("service is asleep and refusing connections")

    _use(monkeypatch, handler)
    assert await wake.wake_indexer("https://indexer.invalid/embed") is False


async def test_no_endpoint_configured_makes_no_request(monkeypatch):
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("tried to wake an indexer that has no address")

    monkeypatch.setattr(wake.httpx, "AsyncClient", explode)
    assert await wake.wake_indexer("") is False


def _use(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(wake.httpx, "AsyncClient", factory)
