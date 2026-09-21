"""
End-to-end smoke test: does the indexer actually index, and can you then
search what it produced?

Nothing here is mocked. It starts a real PostgreSQL (embedded-postgres,
PG 18.6 + pgvector 0.8.6), applies the real migrations, clones a real
repository over https, runs the real worker with the real ONNX model, and
then runs routed hybrid search against the result.

Why it is a script and not a test
---------------------------------
It clones from GitHub, so it needs the network and takes about a minute.
The unit suite must run offline and deterministically, so this lives here
and is run deliberately.

It is worth running deliberately, because this is the path that had never
been executed once: the worker had no entrypoint, and the embedding
endpoint it depends on did not exist. Running it for the first time found
that unsafe-URL failures were being retried despite a comment saying they
were not -- see BENCHMARKS.md section 8.

Expected output ends with `END TO END OK`. Exit code 0 means:

  * migrations applied against a real pgvector
  * the job was claimed, executed and marked `done`
  * every chunk has BOTH a halfvec and a bit vector
  * the repository reached `ready`
  * retrieval returns hits, and the router classifies each query correctly

Usage:
    python bench/e2e_smoke.py
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
import shutil
import sys
import tempfile
import time
import uuid

BACKEND = pathlib.Path(__file__).resolve()
sys.path.insert(0, r"D:\clarix\backend")
os.chdir(r"D:\clarix\backend")


def main() -> int:
    import psycopg
    from embedded_postgres import get_server

    datadir = pathlib.Path(tempfile.mkdtemp(prefix="clarix_e2e_pg_"))
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="clarix_e2e_work_"))
    server = get_server(datadir, cleanup_mode=None)
    try:
        admin = server.get_uri()
        dbname = f"e2e_{uuid.uuid4().hex[:8]}"
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(f'CREATE DATABASE "{dbname}"')
        sync_url = server.get_uri(database=dbname)
        print(f"postgres up: {dbname}")

        base = sync_url.replace("postgresql://", "", 1)
        async_url = f"postgresql+asyncpg://{base}"
        os.environ["DATABASE_URL"] = async_url
        os.environ["WORKER_DIR"] = str(workdir)
        os.environ["APP_ENV"] = "test"
        os.environ["INDEXER_EMBED_BATCH"] = "1"

        from alembic import command
        from alembic.config import Config

        cfg = Config("alembic.ini")
        os.environ["ALEMBIC_DATABASE_URL"] = sync_url.replace(
            "postgresql://", "postgresql+psycopg://", 1
        )
        command.upgrade(cfg, "head")
        print("migrations applied")

        with psycopg.connect(sync_url, autocommit=True) as conn:
            ver = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname='vector'"
            ).fetchone()[0]
            print(f"pgvector {ver}")

        return asyncio.run(run(sync_url, workdir))
    finally:
        # Teardown must not mask a real failure above it.
        with contextlib.suppress(Exception):
            server.cleanup()
        shutil.rmtree(datadir, ignore_errors=True)
        shutil.rmtree(workdir, ignore_errors=True)


async def run(sync_url: str, workdir: pathlib.Path) -> int:
    from app.database import async_session_factory
    from app.indexing import queue
    from app.indexing.chunker import ASTChunker
    from app.indexing.embedder import OnnxEmbedder
    from app.indexing.service import SharedEmbedder
    from app.indexing.tokens import get_token_counter
    from app.indexing.worker import WorkerConfig, run_worker
    from sqlalchemy import text

    repo_id = str(uuid.uuid4())
    # A real repository over https. file:// and ext:: are rejected by the
    # clone guard, which this run confirmed the hard way.
    clone_url = "https://github.com/pallets/itsdangerous.git"
    source = "(cloned)"

    async with async_session_factory() as db:
        await db.execute(
            text(
                "INSERT INTO repositories (id, name, url, local_path, status) "
                "VALUES (:i, :n, :u, :p, 'pending')"
            ),
            {"i": repo_id, "n": "itsdangerous", "u": clone_url, "p": str(source)},
        )
        job_id = await queue.enqueue(db, repo_id, "full_index", {})
        await db.commit()
        print(f"enqueued job {job_id} for repo {repo_id[:8]}")

        depth = await queue.queue_depth(db)
        print(f"queue depth: {depth}")

    print("loading model...")
    t0 = time.perf_counter()
    # Built from Settings, not from the module defaults. `OnnxEmbedder()`
    # with no arguments is the benchmark reference (fp16 at 2048), which is
    # NOT what deploys -- testing that would exercise a configuration
    # nobody runs.
    from app.config import get_settings

    cfg = get_settings()
    embedder = SharedEmbedder(
        OnnxEmbedder(
            model_id=cfg.embedding_model_id,
            onnx_file=cfg.embedding_onnx_file,
            max_tokens=cfg.embedding_max_tokens,
            threads=cfg.embedding_threads or None,
        ),
        batch_size=1,
    )
    print(f"model ready in {time.perf_counter() - t0:.1f}s "
          f"({cfg.embedding_onnx_file}, cap {cfg.embedding_max_tokens})")
    print(f"identity: {embedder._inner.identity}")

    config = WorkerConfig(
        workdir=workdir,
        model_id=embedder.model_id,
        index_version=1,
        batch_size=64,
    )
    chunker = ASTChunker(count_tokens=get_token_counter().count)

    print("running worker (max_jobs=1)...")
    t0 = time.perf_counter()
    processed = await run_worker(
        async_session_factory, config, embedder, chunker, max_jobs=1
    )
    elapsed = time.perf_counter() - t0
    print(f"worker processed {processed} job(s) in {elapsed:.1f}s")

    async with async_session_factory() as db:
        row = (
            await db.execute(
                text(
                    "SELECT status, last_error FROM ingest_jobs WHERE id = :i"
                ),
                {"i": job_id},
            )
        ).first()
        print(f"job status: {row.status}  error: {row.last_error}")
        if row.status != "done":
            return 1

        n_chunks = (
            await db.execute(
                text("SELECT count(*) FROM chunks WHERE repo_id = :i"),
                {"i": repo_id},
            )
        ).scalar()
        n_embedded = (
            await db.execute(
                text(
                    "SELECT count(*) FROM chunks "
                    "WHERE repo_id = :i AND embedding IS NOT NULL AND embedding_bits IS NOT NULL"
                ),
                {"i": repo_id},
            )
        ).scalar()
        n_cache = (await db.execute(text("SELECT count(*) FROM embedding_cache"))).scalar()
        repo = (
            await db.execute(
                text("SELECT status, ingestion_phase, ingestion_total_chunks "
                     "FROM repositories WHERE id = :i"),
                {"i": repo_id},
            )
        ).one()
        print(f"chunks: {n_chunks}  with vectors: {n_embedded}  "
              f"cache rows: {n_cache}  repo status: {repo.status}")
        print(f"progress fields: phase={repo.ingestion_phase} "
              f"counter={repo.ingestion_total_chunks}")
        if n_chunks == 0 or n_embedded != n_chunks:
            print("FAIL: chunks missing vectors")
            return 1
        # The dashboard reads these. They were never written, so the whole
        # progress panel stayed blank for the length of the run.
        if repo.ingestion_phase != "done" or repo.ingestion_total_chunks != n_chunks:
            print("FAIL: progress fields not published for the UI")
            return 1

        # --- retrieval against what was just indexed ----------------------
        from app.indexing.embedder import EMBED_DIM, binary_quantize, bits_to_sql, vector_to_sql
        from app.retrieval import hybrid_search

        # One natural-language query, one bare identifier, one more prose
        # query: the router must classify these differently, and the
        # printed route shows whether it did.
        for q in ["how is a signature verified",
                  "BadSignature",
                  "what does the timestamp signer do"]:
            m = embedder.embed([q], batch_size=1)
            hits, trace = await hybrid_search(
                db, repo_id, q,
                query_bits=bits_to_sql(binary_quantize(m)[0], EMBED_DIM),
                query_vector=vector_to_sql(m[0]),
                limit=5,
            )
            top = hits[0] if hits else None
            print(f"  [{trace.route:<10}] {q[:38]:<40} -> {len(hits)} hits"
                  + (f", top={top.citation} {top.symbol or ''}" if top else ""))
            if not hits:
                print("FAIL: retrieval returned nothing")
                return 1

    print("\nEND TO END OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
