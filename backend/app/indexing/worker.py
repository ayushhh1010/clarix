"""
Indexing worker: claim a job, do the work, report the outcome.

Runs out of process from the API. That split is the point of the whole
design: the API container holds only FastAPI and a Postgres driver, while
tree-sitter, onnxruntime and the model weights live here, where a burst of
memory costs nothing that a request path has to survive.

The loop is deliberately boring -- reap, claim, dispatch, heartbeat,
complete or fail. Everything interesting about durability lives in
`queue.py`, and everything interesting about memory lives in `pipeline.py`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import signal
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.indexing import queue
from app.indexing.chunker import ASTChunker
from app.indexing.pipeline import index_repository
from app.indexing.source import (
    SourceError,
    UnsafeSourceError,
    clone,
    remote_head,
    validate_url,
)

logger = logging.getLogger(__name__)

IDLE_SLEEP_SECONDS = 2.0
REAP_EVERY_ITERATIONS = 30


@dataclass
class WorkerConfig:
    workdir: Path
    model_id: str
    index_version: int = 1
    batch_size: int = 64
    poll_kinds: list[str] | None = None


class JobFailure(Exception):
    """A job failed for a reason the user should see."""


async def _repo_row(db: AsyncSession, repo_id: str):
    return (
        await db.execute(
            text("SELECT id, name, url, indexed_commit_sha, index_version "
                 "FROM repositories WHERE id = :i"),
            {"i": repo_id},
        )
    ).first()


async def handle_full_index(
    db: AsyncSession,
    job: queue.Job,
    config: WorkerConfig,
    embedder,
    chunker: ASTChunker,
) -> dict:
    """Clone the repository and index it."""
    repo = await _repo_row(db, job.repo_id)
    if repo is None:
        # The repository was deleted while the job sat in the queue. Not an
        # error: the work is simply no longer wanted.
        logger.info("repo %s no longer exists; job %s is a no-op", job.repo_id[:8], job.id)
        return {"skipped": "repository deleted"}
    if not repo.url:
        raise JobFailure("repository has no URL to clone")

    force = bool(job.payload.get("force"))
    # Cheapest possible incremental check: if the remote head already
    # matches what we indexed at the current index version, there is
    # nothing to do and no clone to pay for.
    if not force and repo.indexed_commit_sha and repo.index_version == config.index_version:
        try:
            head = remote_head(repo.url)
        except SourceError as exc:
            logger.info("head check failed (%s); cloning anyway", exc)
        else:
            if head == repo.indexed_commit_sha:
                logger.info("repo %s already at %s; nothing to do",
                            job.repo_id[:8], head[:8])
                await db.execute(
                    text("UPDATE repositories SET status = 'ready', "
                         "last_indexed_at = now(), updated_at = now() WHERE id = :i"),
                    {"i": job.repo_id},
                )
                await db.commit()
                return {"skipped": "already current", "commit": head}

    # Validate before allocating anything. An unsafe URL must not cause a
    # temp directory to be created, and the failure the user sees must be
    # "scheme not permitted", not a filesystem error from the cleanup path.
    try:
        validate_url(repo.url)
    except UnsafeSourceError as exc:
        raise JobFailure(str(exc)) from exc

    config.workdir.mkdir(parents=True, exist_ok=True)
    dest = Path(tempfile.mkdtemp(prefix="clarix_src_", dir=str(config.workdir)))
    try:
        try:
            checkout = clone(repo.url, dest / "repo", ref=job.payload.get("ref"))
        except UnsafeSourceError as exc:
            # Not retryable: the URL will still be unsafe next time.
            raise JobFailure(str(exc)) from exc
        except SourceError as exc:
            raise JobFailure(f"clone failed: {exc}") from exc

        await db.execute(
            text("UPDATE repositories SET name = COALESCE(NULLIF(name, ''), :n), "
                 "head_commit_sha = :sha, default_branch = :branch, "
                 "updated_at = now() WHERE id = :i"),
            {"i": job.repo_id, "n": checkout.name,
             "sha": checkout.commit_sha, "branch": checkout.default_branch},
        )
        await db.commit()

        async def beat() -> None:
            await queue.heartbeat(db, job.id)

        stats = await index_repository(
            db, job.repo_id, checkout.path, embedder, chunker,
            model_id=config.model_id,
            index_version=config.index_version,
            batch_size=config.batch_size,
            commit_id=checkout.commit_sha,
            heartbeat=beat,
        )
        return {
            "commit": checkout.commit_sha,
            "chunks": stats.chunks_written,
            "deleted": stats.chunks_deleted,
            "cache_hit_rate": round(stats.cache_hit_rate, 4),
            "seconds": round(stats.total_seconds, 1),
        }
    finally:
        # Always remove the checkout. A worker that leaks clones fills its
        # disk and then fails every subsequent job for an unrelated reason.
        shutil.rmtree(dest, ignore_errors=True)


async def handle_delete(db: AsyncSession, job: queue.Job, *_args) -> dict:
    """Drop a repository's chunks. The row itself cascades."""
    result = await db.execute(
        text("DELETE FROM chunks WHERE repo_id = :i RETURNING chunk_id"),
        {"i": job.repo_id},
    )
    n = len(result.fetchall())
    await db.commit()
    return {"chunks_deleted": n}


HANDLERS: dict[str, Callable] = {
    "full_index": handle_full_index,
    "incremental": handle_full_index,  # the head check makes these the same path
    "delete": handle_delete,
}


async def run_once(
    db: AsyncSession, config: WorkerConfig, embedder, chunker: ASTChunker
) -> bool:
    """
    Claim and run at most one job. Returns True if one was processed.

    Exceptions are converted to queue failures rather than propagated: a
    worker that dies on a bad repository stops processing every other
    repository too.
    """
    job = await queue.claim(db, config.poll_kinds)
    if job is None:
        return False

    logger.info("job %s: %s for repo %s (attempt %d/%d)",
                job.id, job.kind, job.repo_id[:8], job.attempts, job.max_attempts)

    handler = HANDLERS.get(job.kind)
    if handler is None:
        await queue.fail(db, job, f"no handler for job kind {job.kind!r}")
        return True

    try:
        result = await handler(db, job, config, embedder, chunker)
    except JobFailure as exc:
        await db.rollback()
        await queue.fail(db, job, str(exc))
    except Exception as exc:  # noqa: BLE001 - one bad job must not stop the worker
        await db.rollback()
        logger.exception("job %s raised", job.id)
        await queue.fail(db, job, f"{type(exc).__name__}: {exc}")
    else:
        await queue.complete(db, job.id)
        logger.info("job %s done: %s", job.id, result)
    return True


async def run_worker(
    session_factory,
    config: WorkerConfig,
    embedder,
    chunker: ASTChunker,
    *,
    stop: asyncio.Event | None = None,
    max_jobs: int | None = None,
) -> int:
    """
    Poll until stopped. Returns the number of jobs processed.

    `max_jobs` bounds a run, which is what makes this testable and also what
    a serverless invocation wants: do a bounded amount of work, exit, let
    the platform scale to zero.
    """
    stop = stop or asyncio.Event()
    config.workdir.mkdir(parents=True, exist_ok=True)
    processed = 0
    iterations = 0

    logger.info("worker started (model=%s, batch=%d)", config.model_id, config.batch_size)
    while not stop.is_set():
        iterations += 1
        async with session_factory() as db:
            if iterations % REAP_EVERY_ITERATIONS == 1:
                try:
                    await queue.reap_stale(db)
                except Exception:  # noqa: BLE001 - reaping is best-effort
                    logger.exception("reaper failed")

            try:
                did_work = await run_once(db, config, embedder, chunker)
            except Exception:  # noqa: BLE001 - never exit the loop on error
                logger.exception("worker iteration failed")
                did_work = False

        if did_work:
            processed += 1
            if max_jobs is not None and processed >= max_jobs:
                break
            continue

        # Idle sleep that wakes immediately if asked to stop, so a deploy
        # does not wait out the poll interval before shutting down.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=IDLE_SLEEP_SECONDS)

    logger.info("worker stopped after %d job(s)", processed)
    return processed


def install_signal_handlers(stop: asyncio.Event) -> None:
    """
    Finish the current job on SIGINT/SIGTERM rather than abandoning it.

    Without this, a deploy mid-index leaves the row `running` until the
    reaper reclaims it minutes later -- correct, but slower than simply
    finishing.
    """
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            # Windows does not support add_signal_handler for these.
            signal.signal(sig, lambda *_: stop.set())
