"""
Tests for the Postgres job queue.

These exist because the v1 failure mode was silent: `BackgroundTasks` died
with the process and left repositories stuck in `status='ingesting'`
forever, with no retry and nothing to observe. Every property below is one
that absence caused.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import text

from app.indexing.queue import (
    claim,
    complete,
    enqueue,
    fail,
    heartbeat,
    queue_depth,
    reap_stale,
)

REPO = "aaaa0000-0000-4000-8000-000000000001"


@pytest.fixture
async def repo(async_session, conn):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO repositories (id, name, local_path, status) "
            "VALUES (%s, 'q', '/tmp/q', 'pending')",
            (REPO,),
        )
    return REPO


# --- basic lifecycle -------------------------------------------------------

async def test_enqueue_then_claim_then_complete(async_session, repo):
    job_id = await enqueue(async_session, repo, "full_index", {"ref": "main"})
    await async_session.commit()
    assert job_id

    job = await claim(async_session)
    assert job is not None
    assert job.repo_id == repo
    assert job.kind == "full_index"
    assert job.payload == {"ref": "main"}
    assert job.attempts == 1

    await complete(async_session, job.id)
    assert await claim(async_session) is None


async def test_claiming_an_empty_queue_returns_none(async_session, repo):
    assert await claim(async_session) is None


async def test_claim_marks_the_row_running_and_records_the_worker(async_session, repo):
    await enqueue(async_session, repo, "full_index")
    await async_session.commit()
    job = await claim(async_session)

    row = (
        await async_session.execute(
            text("SELECT status, locked_by, heartbeat_at FROM ingest_jobs WHERE id = :i"),
            {"i": job.id},
        )
    ).one()
    assert row.status == "running"
    assert row.locked_by
    assert row.heartbeat_at is not None


async def test_kind_filter_only_claims_matching_jobs(async_session, repo):
    await enqueue(async_session, repo, "full_index")
    await enqueue(async_session, repo, "delete", dedupe=False)
    await async_session.commit()

    job = await claim(async_session, kinds=["delete"])
    assert job is not None and job.kind == "delete"


# --- concurrency -----------------------------------------------------------

async def test_two_workers_never_claim_the_same_job(async_session, migrated, repo):
    """
    The property `FOR UPDATE SKIP LOCKED` exists for. Two independent
    sessions claim concurrently; each must get a different row.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    for i in range(4):
        await enqueue(async_session, repo, f"k{i}", dedupe=False)
    await async_session.commit()

    url = migrated.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
    engines = [create_async_engine(url) for _ in range(3)]
    claimed = []
    try:
        for eng in engines:
            async with async_sessionmaker(eng, expire_on_commit=False)() as s:
                job = await claim(s)
                if job:
                    claimed.append(job.id)
    finally:
        for eng in engines:
            await eng.dispose()

    assert len(claimed) == len(set(claimed)), f"same job claimed twice: {claimed}"
    assert len(claimed) == 3


# --- deduplication ---------------------------------------------------------

async def test_duplicate_enqueue_is_suppressed(async_session, repo):
    """A user clicking reindex five times must not queue five reindexes."""
    first = await enqueue(async_session, repo, "full_index")
    second = await enqueue(async_session, repo, "full_index")
    await async_session.commit()
    assert first is not None
    assert second is None


async def test_dedupe_can_be_disabled(async_session, repo):
    a = await enqueue(async_session, repo, "full_index", dedupe=False)
    b = await enqueue(async_session, repo, "full_index", dedupe=False)
    await async_session.commit()
    assert a != b


async def test_a_completed_job_does_not_block_requeueing(async_session, repo):
    first = await enqueue(async_session, repo, "full_index")
    await async_session.commit()
    job = await claim(async_session)
    await complete(async_session, job.id)

    again = await enqueue(async_session, repo, "full_index")
    await async_session.commit()
    assert again is not None and again != first


# --- failure handling ------------------------------------------------------

async def test_failure_requeues_with_backoff(async_session, repo):
    await enqueue(async_session, repo, "full_index")
    await async_session.commit()
    job = await claim(async_session)

    assert await fail(async_session, job, "clone timed out") == "queued"

    row = (
        await async_session.execute(
            text("SELECT status, last_error, run_after > now() AS deferred "
                 "FROM ingest_jobs WHERE id = :i"),
            {"i": job.id},
        )
    ).one()
    assert row.status == "queued"
    assert "clone timed out" in row.last_error
    assert row.deferred, "retry should be deferred, not immediate"

    # Deferred means not immediately claimable.
    assert await claim(async_session) is None


async def test_permanent_failure_dead_letters_on_the_first_attempt(
    async_session, repo
):
    """
    Some failures cannot be fixed by waiting: a forbidden URL scheme, a job
    kind with no handler. Retrying those three times with backoff only
    delays the error the user needs to see.
    """
    await enqueue(async_session, repo, "full_index")
    await async_session.commit()
    job = await claim(async_session)
    assert job.attempts == 1
    assert job.attempts < job.max_attempts, "attempts are not yet exhausted"

    status = await fail(async_session, job, "scheme not permitted", permanent=True)
    assert status == "dead"

    row = (
        await async_session.execute(
            text("SELECT status, last_error FROM ingest_jobs WHERE id = :i"),
            {"i": job.id},
        )
    ).one()
    assert row.status == "dead"
    assert "not permitted" in row.last_error
    # A dead job must not be claimable again.
    assert await claim(async_session) is None


async def test_retryable_failure_still_retries_when_attempts_remain(
    async_session, repo
):
    """The permanent flag must not change the default path."""
    await enqueue(async_session, repo, "full_index")
    await async_session.commit()
    job = await claim(async_session)
    assert await fail(async_session, job, "network blip", permanent=False) == "queued"


async def test_job_is_dead_lettered_after_max_attempts(async_session, repo):
    await enqueue(async_session, repo, "full_index")
    await async_session.commit()
    await async_session.execute(
        text("UPDATE ingest_jobs SET max_attempts = 2 WHERE repo_id = :r"),
        {"r": repo},
    )
    await async_session.commit()

    statuses = []
    for _ in range(2):
        await async_session.execute(text("UPDATE ingest_jobs SET run_after = now()"))
        await async_session.commit()
        job = await claim(async_session)
        assert job is not None
        statuses.append(await fail(async_session, job, "boom"))

    assert statuses == ["queued", "dead"]
    await async_session.execute(text("UPDATE ingest_jobs SET run_after = now()"))
    await async_session.commit()
    assert await claim(async_session) is None, "a dead job must not be re-claimed"


async def test_dead_jobs_remain_visible(async_session, repo):
    """A vanished job is indistinguishable from one that never ran."""
    await enqueue(async_session, repo, "full_index")
    await async_session.commit()
    await async_session.execute(text("UPDATE ingest_jobs SET max_attempts = 1"))
    await async_session.commit()
    job = await claim(async_session)
    await fail(async_session, job, "fatal")

    assert (await queue_depth(async_session)).get("dead") == 1


# --- stale lock recovery ---------------------------------------------------

async def test_reaper_reclaims_a_job_whose_worker_died(async_session, repo):
    """
    The exact v1 failure: a worker dies mid-job. Without a reaper the row
    stays `running` forever and the repository never recovers.
    """
    await enqueue(async_session, repo, "full_index")
    await async_session.commit()
    job = await claim(async_session)

    await async_session.execute(
        text("UPDATE ingest_jobs SET heartbeat_at = now() - interval '1 hour' "
             "WHERE id = :i"),
        {"i": job.id},
    )
    await async_session.commit()

    assert await reap_stale(async_session, timedelta(minutes=10)) == 1
    reclaimed = await claim(async_session)
    assert reclaimed is not None and reclaimed.id == job.id
    assert reclaimed.attempts == 2


async def test_reaper_leaves_a_heartbeating_job_alone(async_session, repo):
    await enqueue(async_session, repo, "full_index")
    await async_session.commit()
    job = await claim(async_session)
    await heartbeat(async_session, job.id)

    assert await reap_stale(async_session, timedelta(minutes=10)) == 0
    assert await claim(async_session) is None


# --- transactional coupling ------------------------------------------------

async def test_enqueue_rolls_back_with_its_transaction(async_session, conn):
    """
    The reason this is a table and not Redis: a job cannot outlive the
    transaction that created its repository.
    """
    rid = "bbbb0000-0000-4000-8000-000000000002"
    await async_session.execute(
        text("INSERT INTO repositories (id, name, local_path, status) "
             "VALUES (:i, 'rb', '/tmp/rb', 'pending')"),
        {"i": rid},
    )
    await enqueue(async_session, rid, "full_index")
    await async_session.rollback()

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM ingest_jobs WHERE repo_id = %s", (rid,))
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM repositories WHERE id = %s", (rid,))
        assert cur.fetchone()[0] == 0


async def test_deleting_a_repository_removes_its_jobs(async_session, repo):
    await enqueue(async_session, repo, "full_index")
    await async_session.commit()

    await async_session.execute(
        text("DELETE FROM repositories WHERE id = :i"), {"i": repo}
    )
    await async_session.commit()

    assert (await queue_depth(async_session)) == {}
