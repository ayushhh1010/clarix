"""
MCP server for Clarix.

Exposes an indexed repository to any MCP host (Claude Code, Claude Desktop,
Cursor) as three tools.

It talks to the deployed HTTP API, not to the database. The API already owns
authentication, per-user ownership checks and the whole retrieval pipeline,
so going through it means this process needs none of the indexer's
dependencies -- no onnxruntime, no tree-sitter, no numpy. The MCP extra is
`mcp` and nothing else; httpx is already in the base set.

Configure with two environment variables:

    CLARIX_URL     base URL, e.g. https://clarix-api-z26m.onrender.com
    CLARIX_TOKEN   a JWT from POST /api/auth/login

Run it directly for stdio (what every host launches):

    python -m app.mcp_server
"""

from __future__ import annotations

import os

import httpx
from mcp.server import MCPServer

BASE_URL = os.environ.get("CLARIX_URL", "http://localhost:8000").rstrip("/")

# Free-tier instances sleep after 15 minutes idle and take ~50 s to wake, on
# top of ~15 s for a chat round trip. A default 5 s timeout would fail every
# first call of the day.
TIMEOUT = 120.0

mcp = MCPServer("Clarix")


async def _call(method: str, path: str, **kwargs) -> dict:
    """One authenticated request. Raises with the API's own message."""
    token = os.environ.get("CLARIX_TOKEN", "")
    if not token:
        raise RuntimeError(
            "CLARIX_TOKEN is not set. Get one from POST /api/auth/login."
        )

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT) as client:
        response = await client.request(
            method, path, headers={"Authorization": f"Bearer {token}"}, **kwargs
        )

    if response.is_error:
        # FastAPI puts the actionable part in `detail`. Surfacing the raw body
        # instead would hand the model an HTML error page to reason about.
        detail = response.text
        if response.headers.get("content-type", "").startswith("application/json"):
            detail = response.json().get("detail", detail)
        raise RuntimeError(f"clarix {response.status_code}: {detail}")

    return response.json()


@mcp.tool()
async def list_repos() -> list[dict]:
    """
    List the indexed repositories on this Clarix account.

    Call this first: every other tool needs a repo_id, and only repositories
    with status "ready" can be queried.
    """
    data = await _call("GET", "/api/repo/", params={"per_page": 100})
    return [
        {
            "repo_id": repo["id"],
            "name": repo["name"],
            "status": repo["status"],
            "chunks": repo["chunk_count"],
        }
        for repo in data["items"]
    ]


@mcp.tool()
async def ask(repo_id: str, question: str) -> dict:
    """
    Ask a natural-language question about an indexed repository.

    Returns an answer grounded in the repository's own code, plus the exact
    file paths and line ranges it was drawn from. Prefer this over guessing
    at a codebase you cannot see.
    """
    data = await _call(
        "POST", "/api/chat", json={"repo_id": repo_id, "message": question}
    )
    return {
        "answer": data["content"],
        "sources": [
            {
                "file": source.get("file_path"),
                "lines": f"{source.get('start_line')}-{source.get('end_line')}",
                "symbol": source.get("symbol"),
            }
            for source in data.get("sources", [])
        ],
    }


@mcp.tool()
async def read_file(repo_id: str, path: str) -> str:
    """
    Read one file at the exact commit that was indexed.

    `path` must be a repository-relative path the index knows about -- the
    `file` values returned by `ask` are always valid here.
    """
    data = await _call(
        "GET", f"/api/repo/{repo_id}/file-content", params={"path": path}
    )
    return data["content"]


if __name__ == "__main__":
    mcp.run()
