"""
Peak memory of a REAL indexing run, sampled while it happens.

Why this exists, and why the two benchmarks before it were not enough
--------------------------------------------------------------------
This number has now been wrong twice, and each time the flaw was in what
was measured rather than in the arithmetic.

  bench_colocated_rss.py measured on Windows, with the model cached, with
  no worker. Render killed the service with
  "Out of memory (used over 512Mi)".

  bench_service_memory.py measured on Linux with a cold download and
  `INDEXER_RUN_WORKER=true` -- but pointed the worker at a database that
  does not exist. The thread started, imported the chunker, failed to
  connect, and idled. That measured the worker's *imports*, not the
  worker *working*: no clone, no parse trees, no batch of chunks held in
  memory, no embedding loop. Render killed it again, mid-index.

So this one runs the thing itself: a real PostgreSQL, a real clone, the
real worker, indexing a real repository end to end, with a sampler thread
reading VmRSS every 100 ms and reporting the true high-water mark.

The lesson worth keeping is narrow and general at once: a memory
measurement is only valid if the process was doing the work whose memory
you are trying to bound. Every shortcut taken to make the measurement
convenient removed exactly the thing that consumed the memory.

Usage (Linux, or WSL on a Windows host):
    python bench/bench_index_memory.py --repo https://github.com/psf/requests
    python bench/bench_index_memory.py --repo <url> --batch 16
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"

CHILD = r'''
import asyncio, json, os, shutil, sys, tempfile, threading, time, uuid
sys.path.insert(0, BACKEND_PATH)

PEAK = {"rss": 0.0, "at": ""}
TRACE = []          # (seconds, rss_mb) -- the shape matters, not just the max
STOP = threading.Event()
T0 = time.time()

def _rss():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024
    return 0.0

def sampler():
    # 100 ms: fine enough to catch a transient spike during a parse or an
    # embed, cheap enough not to perturb what it is measuring.
    last = 0.0
    while not STOP.is_set():
        r = _rss()
        if r > PEAK["rss"]:
            PEAK["rss"] = r
            PEAK["at"] = STAGE[0]
        now = time.time() - T0
        if now - last >= 2.0:          # one sample every 2 s is enough shape
            TRACE.append((round(now, 1), round(r, 1)))
            last = now
        time.sleep(0.1)

STAGE = ["startup"]
threading.Thread(target=sampler, daemon=True).start()

import psycopg
from embedded_postgres import get_server

datadir = Path_ = tempfile.mkdtemp(prefix="idxmem_pg_")
workdir = tempfile.mkdtemp(prefix="idxmem_work_")
server = get_server(datadir, cleanup_mode=None)
admin = server.get_uri()
dbname = "m" + uuid.uuid4().hex[:8]
with psycopg.connect(admin, autocommit=True) as c:
    c.execute('CREATE DATABASE "%s"' % dbname)
sync_url = server.get_uri(database=dbname)
os.environ["DATABASE_URL"] = "postgresql+asyncpg://" + sync_url.split("://", 1)[1]
os.environ["ALEMBIC_DATABASE_URL"] = "postgresql+psycopg://" + sync_url.split("://", 1)[1]
os.environ["WORKER_DIR"] = workdir
os.environ["APP_ENV"] = "development"
os.environ["INDEX_BATCH_SIZE"] = str(BATCH)
os.environ["EMBEDDING_MEM_ARENA"] = ARENA

os.chdir(BACKEND_PATH)
from alembic import command
from alembic.config import Config
command.upgrade(Config("alembic.ini"), "head")
STAGE[0] = "migrated"

from sqlalchemy import text

from app.config import get_settings
from app.database import async_session_factory
from app.indexing import queue
from app.indexing.chunker import ASTChunker
from app.indexing.embedder import OnnxEmbedder
from app.indexing.service import SharedEmbedder
from app.indexing.tokens import get_token_counter
from app.indexing.worker import WorkerConfig, run_worker

cfg = get_settings()
STAGE[0] = "loading model"
inner = OnnxEmbedder(
    model_id=cfg.embedding_model_id,
    onnx_file=cfg.embedding_onnx_file,
    max_tokens=cfg.embedding_max_tokens,
    threads=cfg.embedding_threads or None,
    enable_mem_arena=cfg.embedding_mem_arena,
)
embedder = SharedEmbedder(inner, batch_size=cfg.indexer_embed_batch)
after_model = _rss()
STAGE[0] = "model loaded"

repo_id = str(uuid.uuid4())

async def main():
    async with async_session_factory() as db:
        await db.execute(text(
            "INSERT INTO repositories (id, name, url, local_path, status) "
            "VALUES (:i,:n,:u,:p,'pending')"),
            {"i": repo_id, "n": "target", "u": REPO_URL, "p": "(cloned)"})
        await queue.enqueue(db, repo_id, "full_index", {})
        await db.commit()

    STAGE[0] = "indexing"
    t0 = time.perf_counter()
    await run_worker(
        async_session_factory,
        WorkerConfig(workdir=Path(workdir), model_id=embedder.identity,
                     index_version=cfg.index_version, batch_size=BATCH),
        embedder, ASTChunker(count_tokens=get_token_counter().count),
        max_jobs=1,
    )
    elapsed = time.perf_counter() - t0
    STAGE[0] = "done"

    async with async_session_factory() as db:
        n = (await db.execute(
            text("SELECT count(*) FROM chunks WHERE repo_id=:i"),
            {"i": repo_id})).scalar()
        st = (await db.execute(
            text("SELECT status FROM repositories WHERE id=:i"),
            {"i": repo_id})).scalar()
        job = (await db.execute(
            text("SELECT status, last_error FROM ingest_jobs WHERE repo_id=:i"),
            {"i": repo_id})).first()
    return n, st, elapsed, job

from pathlib import Path
try:
    chunks, status, elapsed, job = asyncio.run(main())
finally:
    STOP.set()
    time.sleep(0.2)
    try:
        server.cleanup()
    except Exception:
        pass
    shutil.rmtree(datadir, ignore_errors=True)
    shutil.rmtree(workdir, ignore_errors=True)

print(json.dumps({
    "repo": REPO_URL, "batch": BATCH,
    "chunks": chunks, "repo_status": status,
    "job_status": job.status if job else None,
    "job_error": (job.last_error if job else None),
    "seconds": round(elapsed, 1),
    "after_model_mb": round(after_model, 1),
    "peak_rss_mb": round(PEAK["rss"], 1),
    "peak_during": PEAK["at"],
    "arena": ARENA,
    "trace": TRACE,
}))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--repo", default="https://github.com/psf/requests")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--arena", choices=["true", "false"], default="false")
    ap.add_argument("--limit-mb", type=float, default=537.0)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    if not Path("/proc/self/status").exists():
        print("Reads /proc; run on Linux (or `wsl` on a Windows host).")
        return 2

    source = (
        f"BACKEND_PATH = {str(BACKEND)!r}\n"
        f"REPO_URL = {args.repo!r}\n"
        f"BATCH = {args.batch}\n"
        f"ARENA = {args.arena!r}\n" + CHILD
    )
    proc = subprocess.run(  # noqa: S603
        [args.python, "-c", source], capture_output=True, text=True,
        cwd=str(BACKEND),
    )
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:], file=sys.stderr)
        return 1

    row = json.loads(proc.stdout.strip().splitlines()[-1])
    head = args.limit_mb - row["peak_rss_mb"]
    print(f"repo          {row['repo']}")
    print(f"batch size    {row['batch']}")
    print(f"chunks        {row['chunks']}  ({row['job_status']}, {row['seconds']}s)")
    if row["job_error"]:
        print(f"job error     {row['job_error'][:200]}")
    print(f"after model   {row['after_model_mb']:.0f} MB")
    print(f"PEAK RSS      {row['peak_rss_mb']:.0f} MB   during: {row['peak_during']}")
    print(f"headroom      {head:.0f} MB against {args.limit_mb:.0f} MB "
          f"-> {'FITS' if head > 0 else 'DOES NOT FIT'}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(row, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
