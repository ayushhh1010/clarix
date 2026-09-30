"""
The MCP server's job is shaping: turn an API response into what a model can
act on, and turn an API error into a message rather than a traceback. That is
what these check, against a mocked transport -- no network, no server.
"""

from __future__ import annotations

import httpx
import pytest

from app import mcp_server


@pytest.fixture
def api(monkeypatch):
    """Route every request the module makes to a handler the test supplies."""

    def install(handler):
        transport = httpx.MockTransport(handler)
        original = httpx.AsyncClient

        def factory(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(mcp_server.httpx, "AsyncClient", factory)

    monkeypatch.setenv("CLARIX_TOKEN", "test-token")
    return install


async def test_token_is_sent_and_repos_are_flattened(api):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "r1",
                        "name": "nanoserve",
                        "status": "ready",
                        "chunk_count": 1385,
                        "url": "https://example.invalid/x",
                    }
                ],
                "total": 1,
                "page": 1,
                "per_page": 100,
                "has_more": False,
            },
        )

    api(handler)
    repos = await mcp_server.list_repos()

    assert seen["auth"] == "Bearer test-token"
    assert repos == [
        {"repo_id": "r1", "name": "nanoserve", "status": "ready", "chunks": 1385}
    ]


async def test_ask_returns_answer_with_citable_sources(api):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "conversation_id": "c1",
                "message_id": "m1",
                "content": "It batches ready requests together.",
                "sources": [
                    {
                        "file_path": "engine/scheduler.py",
                        "start_line": 335,
                        "end_line": 344,
                        "symbol": "_record_finished",
                    }
                ],
            },
        )

    api(handler)
    result = await mcp_server.ask("r1", "how does the scheduler work?")

    assert result["answer"] == "It batches ready requests together."
    assert result["sources"] == [
        {"file": "engine/scheduler.py", "lines": "335-344", "symbol": "_record_finished"}
    ]


async def test_api_error_surfaces_the_detail_not_the_body(api):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409, json={"detail": "Repository is not ready (status: ingesting)"}
        )

    api(handler)
    with pytest.raises(RuntimeError, match="not ready"):
        await mcp_server.ask("r1", "anything")


async def test_missing_token_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("CLARIX_TOKEN", raising=False)

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("a request was made without a token")

    monkeypatch.setattr(mcp_server.httpx, "AsyncClient", explode)
    with pytest.raises(RuntimeError, match="CLARIX_TOKEN"):
        await mcp_server.list_repos()
