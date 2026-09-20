"""
Durable job queue on Postgres, claimed with `FOR UPDATE SKIP LOCKED`.

Replaces FastAPI `BackgroundTasks`, which has no durability semantics: the
v1 ingestion ran in-process, so a spin-down mid-ingest left the repository
permanently in `status='ingesting'` with no retry and nothing to notice it.

Why a table rather than Redis
-----------------------------
Enqueueing a job and updating the repository row happen in one transaction,
so a job can never reference a repository that was rolled back, and a
repository can never be marked queued without a job existing. That
guarantee is the whole point; it is not available across two stores.

The audit that preceded this found v1's Redis held exactly one thing -- a
facts set that no code path ever wrote to -- so removing it cost nothing.

Claiming
--------
`FOR UPDATE SKIP LOCKED` is the standard Postgres queue primitive: each
worker locks a distinct row and concurrent workers skip rather than block.
Combined with a heartbeat and a reaper, a worker that dies mid-job has its
row reclaimed rather than stranding it.
"""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# A worker is presumed dead if its heartbeat goes quiet for this long. Must
# comfortably exceed the heartbeat interval, or healthy workers get their
# own jobs stolen mid-run.
STALE_LOCK_TIMEOUT = timedelta(minutes=10)
HEARTBEAT_INTERVAL = timedelta(seconds=30)

# Retry backoff, indexed by attempt number. Cloning a large repository can
# fail transiently; hammering it immediately helps nobody.
RETRY_BACKOFF = [timedelta(seconds=30), timedelta(minutes=5), timedelta(minutes=30)]


def worker_id() -> str:
    """Stable-enough identity for lock attribution and debugging."""
    return f"{socket.gethostname()}:{os.getpid()}"


@dataclass
class Job:
    id: int
    repo_id: str
    kind: str
    payload: dict[str, Any]
    attempts: int
    max_attempts: int


async def enqueue(
    db: AsyncSession,
    repo_id: str,
    kind: str,
    payload: dict[str, Any] | None = None,
    *,
    dedupe: bool = True,
) -> int | None:
    """
    Queue a job. Returns its id, or None if deduplicated.

    Does NOT commit: the caller decides the transaction boundary, which is
    what makes "create the repository row and queue its indexing job
    atomically" possible.

    With `dedupe`, an existing queued or running job of the same kind for
    the same repository suppresses the new one. Without it, a user clicking
    "reindex" five times queues five full reindexes.
    """
    if dedupe:
        existing = await db.execute(
            text(
                "SELECT id FROM ingest_jobs "
                "WHERE repo_id = :repo_id AND kind = :kind "
                "  AND status IN ('queued', 'running') LIMIT 1"
            ),
            {"repo_id": repo_id, "kind": kind},
        )
        row = existing.first()
        if row:
            logger.info("job %s already pending for repo %s; not duplicating",
                        kind, repo_id[:8])
            return None

    import json

    result = await db.execute(
        text(
            "INSERT INTO ingest_jobs (repo_id, kind, payload) "
            "VALUES (:repo_id, :kind, CAST(:payload AS jsonb)) RETURNING id"
        ),
        {"repo_id": repo_id, "kind": kind, "payload": json.dumps(payload or {})},
    )
    return int(result.scalar_one())


async def claim(db: AsyncSession, kinds: list[str] | None = None) -> Job | None:
    """
    Atomically claim one runnable job, or None.

    The `SKIP LOCKED` is what makes this safe under concurrency: two workers
    racing take different rows instead of one blocking on the other's lock.
    Commits, because holding the claim open until the job finishes would
    keep a transaction alive for the entire indexing run -- and a long-lived
    transaction is exactly the subtransaction/bloat hazard documented in
    app/retrieval/hybrid.py.
    """
    # The only interpolated value is this module-local literal; the kind
    # list itself is a bind parameter.
    kind_filter = "AND kind = ANY(:kinds)" if kinds else ""  # noqa: S608
    result = await db.execute(
        text(
            f"""
            WITH claimed AS (
                SELECT id FROM ingest_jobs
                WHERE status = 'queued' AND run_after <= now() {kind_filter}
                ORDER BY run_after
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            UPDATE ingest_jobs j
               SET status = 'running',
                   locked_by = :worker,
                   locked_at = now(),
                   heartbeat_at = now(),
                   attempts = j.attempts + 1,
                   updated_at = now()
              FROM claimed
             WHERE j.id = claimed.id
         RETURNING j.id, j.repo_id, j.kind, j.payload, j.attempts, j.max_attempts
            """
        ),
        {"worker": worker_id(), **({"kinds": kinds} if kinds else {})},
    )
    row = result.first()
    await db.commit()
    if row is None:
        return None
    return Job(
        # asyncpg returns a uuid.UUID for a uuid column; the rest of the
        # application passes repo ids around as strings, and a silent type
        # split here would surface as a confusing mismatch far downstream.
        id=int(row.id),
        repo_id=str(row.repo_id),
        kind=row.kind,
        payload=row.payload or {},
        attempts=int(row.attempts),
        max_attempts=int(row.max_attempts),
    )


async def heartbeat(db: AsyncSession, job_id: int) -> None:
    """Signal liveness so the reaper does not reclaim this job."""
    await db.execute(
        text("UPDATE ingest_jobs SET heartbeat_at = now() WHERE id = :id"),
        {"id": job_id},
    )
    await db.commit()


async def complete(db: AsyncSession, job_id: int) -> None:
    await db.execute(
        text(
            "UPDATE ingest_jobs SET status = 'done', locked_by = NULL, "
            "locked_at = NULL, heartbeat_at = NULL, updated_at = now() "
            "WHERE id = :id"
        ),
        {"id": job_id},
    )
    await db.commit()


async def fail(
    db: AsyncSession, job: Job, error: str, *, permanent: bool = False
) -> str:
    """
    Record a failure and either schedule a retry or dead-letter the job.

    Returns the resulting status. Dead-lettered jobs stay in the table:
    a silently vanished job is indistinguishable from one that never ran.

    `permanent` dead-letters immediately, for failures that cannot succeed
    on a retry -- a URL with a forbidden scheme, a job kind with no handler.
    Without it, `worker.py` was retrying unsafe URLs three times with
    backoff despite a comment claiming they were not retryable, which
    delayed the error the user actually needs to see and did nothing else.
    """
    exhausted = permanent or job.attempts >= job.max_attempts
    status = "dead" if exhausted else "queued"
    delay = RETRY_BACKOFF[min(job.attempts - 1, len(RETRY_BACKOFF) - 1)]

    await db.execute(
        text(
            # Intervals go through make_interval(secs => float) rather than
            # a bound `interval` parameter. Binding a string fails under
            # asyncpg ("'str' object has no attribute 'days'"), and binding
            # a timedelta leaves the operator ambiguous
            # ("operator does not exist: timestamp with time zone <
            # interval"). A numeric argument to make_interval has neither
            # problem and reads the same under psycopg.
            "UPDATE ingest_jobs SET status = :status, last_error = :err, "
            "run_after = now() + make_interval(secs => :delay_seconds), "
            "locked_by = NULL, locked_at = NULL, heartbeat_at = NULL, "
            "updated_at = now() WHERE id = :id"
        ),
        {
            "id": job.id, "status": status, "err": error[:4000],
            "delay_seconds": float(delay.total_seconds()),
        },
    )
    await db.commit()
    logger.warning(
        "job %s (%s, attempt %d/%d) failed -> %s: %s",
        job.id, job.kind, job.attempts, job.max_attempts, status, error[:200],
    )
    return status


async def reap_stale(
    db: AsyncSession, stale_after: timedelta = STALE_LOCK_TIMEOUT
) -> int:
    """
    Return jobs whose worker stopped heartbeating to the queued state.

    Without this a worker killed mid-job strands its row in `running`
    forever -- the exact v1 failure mode, where a spin-down left
    repositories permanently `ingesting`.
    """
    result = await db.execute(
        text(
            "UPDATE ingest_jobs SET status = 'queued', locked_by = NULL, "
            "locked_at = NULL, heartbeat_at = NULL, "
            "last_error = 'reclaimed: worker heartbeat timed out', "
            "updated_at = now() "
            "WHERE status = 'running' "
            "  AND heartbeat_at < now() - make_interval(secs => :timeout_seconds) "
            "RETURNING id"
        ),
        {"timeout_seconds": float(stale_after.total_seconds())},
    )
    ids = [r.id for r in result]
    await db.commit()
    if ids:
        logger.warning("reclaimed %d stale job(s): %s", len(ids), ids)
    return len(ids)


async def queue_depth(db: AsyncSession) -> dict[str, int]:
    """Counts by status, for health checks and metrics."""
    result = await db.execute(
        text("SELECT status, count(*) AS n FROM ingest_jobs GROUP BY status")
    )
    return {row.status: row.n for row in result}


def _utcnow() -> datetime:
    return datetime.now(UTC)
