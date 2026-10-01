"""
Wake the indexer when work is queued for it.

The two services talk through the database: the API inserts a row into
`ingest_jobs`, and the indexer's worker polls for it. That is fine while both
processes are running, and deadlocks when they are not.

Free-tier hosts stop a service that has had no inbound HTTP for ~15 minutes.
The indexer receives inbound HTTP for exactly one reason -- the API asking it
to embed a *query* -- and a query can only be asked of a repository that is
already indexed. So a stopped indexer is never woken by the thing that needs
it, and a newly queued job waits forever rather than slowly.

One HTTP request to the indexer breaks the cycle, because the request itself
is what the host watches for. It is deliberately fire-and-forget: the job is
already committed, the worker will find it whenever it next polls, and a
failed wake must never turn a successful upload into an error.
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# Long enough to cover a cold start we are not waiting on anyway, short
# enough that a wedged indexer cannot pile up requests indefinitely.
WAKE_TIMEOUT = 30.0


def health_url(endpoint: str) -> str:
    """
    The indexer's health URL, from whatever form its address was configured in.

    `EMBEDDING_ENDPOINT` may be a bare origin (what Render's `fromService`
    injects) or a full `/embed` URL. Both describe the same service, and
    either way the cheapest thing to ask it for is `/health`.
    """
    cleaned = endpoint.strip().rstrip("/")
    if not cleaned:
        return ""
    parsed = urlparse(cleaned)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}/health"


async def wake_indexer(endpoint: str) -> bool:
    """
    Ask the indexer for its health, to cause the traffic that starts it.

    Returns whether the request got a response, for tests and logging. Never
    raises: every caller has already committed the job it is waking for.
    """
    url = health_url(endpoint)
    if not url:
        logger.debug("no embedding endpoint configured; not waking the indexer")
        return False

    try:
        async with httpx.AsyncClient(timeout=WAKE_TIMEOUT) as client:
            response = await client.get(url)
        logger.info("woke indexer at %s (%s)", url, response.status_code)
        return True
    except Exception as exc:  # noqa: BLE001 - a failed wake is not a failed upload
        logger.warning("could not wake indexer at %s: %s", url, exc)
        return False


def schedule_wake(endpoint: str) -> None:
    """
    Fire the wake off in the background.

    The caller is answering a user who is waiting on a 202. Waiting out a
    cold start before replying would make the queue's whole point -- that
    indexing is asynchronous -- invisible to them.
    """
    task = asyncio.create_task(wake_indexer(endpoint))
    # Hold a reference, or the event loop may collect the task mid-flight.
    _PENDING.add(task)
    task.add_done_callback(_PENDING.discard)


_PENDING: set[asyncio.Task] = set()
