"""
Peak memory of the indexer service, measured on Linux, where it deploys.

Why this replaces the earlier measurement
-----------------------------------------
`bench_colocated_rss.py` measured peak RSS with psutil on Windows, with
the model already cached, and concluded the indexer fitted a 512 MiB
instance with 73 MB to spare. Render then killed it:

    ==> Out of memory (used over 512Mi)

Host RSS on one operating system is not a prediction of a cgroup limit on
another. This measures the thing that actually matters: peak RSS of the
real ASGI application, on Linux, reading `VmHWM` from /proc -- the
kernel's own high-water mark -- with the option to force a cold download
so the first-boot path is included.

What it found
-------------
fp16 cannot fit at any sequence length: 915 MB just to *load*. CPUs have
no native fp16 kernels, so ONNX Runtime upcasts every weight to fp32 --
306 MB on disk becomes ~612 MB resident plus overhead. That also explains,
independently, why fp16 measured *slower* than fp32 in section 6.

The rest of the peak is attention, which is O(sequence^2), so the
truncation cap is the second lever.

Usage (from Linux, or WSL on a Windows host):
    python bench/bench_service_memory.py
    python bench/bench_service_memory.py --cold      # force a fresh download
    python bench/bench_service_memory.py --json-out bench/results/memory_linux.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"

# One fresh interpreter per configuration: a previous session's allocator
# growth would otherwise be attributed to whichever variant ran after it.
CHILD = r'''
import asyncio, json, os, sys
sys.path.insert(0, BACKEND_PATH)
os.environ["INDEXER_RUN_WORKER"] = RUN_WORKER
os.environ["APP_ENV"] = "development"
os.environ["EMBEDDING_API_KEY"] = ""
os.environ["DATABASE_URL"] = "postgresql+asyncpg://u:p@localhost/x"
os.environ["EMBEDDING_ONNX_FILE"] = ONNX_FILE
os.environ["EMBEDDING_MAX_TOKENS"] = str(MAX_TOKENS)
os.environ["EMBEDDING_THREADS"] = str(THREADS)

def _proc(field):
    for line in open("/proc/self/status"):
        if line.startswith(field):
            return int(line.split()[1]) / 1024
    return -1.0

hwm = lambda: _proc("VmHWM")   # noqa: E731 - kernel peak RSS
rss = lambda: _proc("VmRSS")   # noqa: E731

import httpx
from app.config import get_settings
from app.indexing.service import build_app

get_settings.cache_clear()
settings = get_settings()
app = build_app(settings)
out = {"onnx_file": settings.embedding_onnx_file,
       "max_tokens": settings.embedding_max_tokens,
       "worker": RUN_WORKER == "true", "threads": THREADS}

async def main():
    async with app.router.lifespan_context(app):
        out["after_load_mb"] = round(rss(), 1)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://x") as c:
            await c.post("/embed", json={"texts": ["how is a signature verified"]})
            out["after_query_mb"] = round(rss(), 1)
            # A chunk at the chunker's cap, which is the worst case the
            # worker hands the model.
            chunk = ("def handler(request, context):\n"
                     + "\n".join(f"    step_{i} = transform(payload[{i}], mode)"
                                 for i in range(120)))
            for _ in range(3):
                await c.post("/embed", json={"texts": [chunk]})
            out["after_long_chunk_mb"] = round(rss(), 1)
    out["peak_mb"] = round(hwm(), 1)
asyncio.run(main())
print(json.dumps(out))
'''

# The worker runs in production, and it loads the tree-sitter chunker and
# the database stack on top of the model. Measuring without it is how the
# first estimate came out optimistic, so the shipped configuration is
# measured both ways and the deploy decision uses the worker-on number.
VARIANTS = [
    ("onnx/model_quantized.onnx", 512, "true", 1),
    ("onnx/model_quantized.onnx", 512, "true", 0),
    ("onnx/model_quantized.onnx", 384, "true", 1),
    ("onnx/model_quantized.onnx", 1024, "true", 1),
    ("onnx/model_fp16.onnx", 512, "true", 1),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--limit-mb", type=float, default=537.0,
                    help="512 MiB, Render's free instance cap")
    ap.add_argument("--cold", action="store_true",
                    help="force a fresh model download, as an ephemeral "
                         "filesystem does on every cold start")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    if not Path("/proc/self/status").exists():
        print("This benchmark reads /proc and must run on Linux.")
        print("On a Windows host: wsl python3 bench/bench_service_memory.py")
        return 2

    results = []
    for onnx_file, max_tokens, run_worker, threads in VARIANTS:
        env = dict(os.environ)
        cache = None
        if args.cold:
            cache = tempfile.mkdtemp(prefix="hf_cold_")
            env["HF_HOME"] = cache

        source = (
            f"BACKEND_PATH = {str(BACKEND)!r}\n"
            f"ONNX_FILE = {onnx_file!r}\n"
            f"MAX_TOKENS = {max_tokens}\n"
            f"RUN_WORKER = {run_worker!r}\n"
            f"THREADS = {threads}\n" + CHILD
        )
        proc = subprocess.run(  # noqa: S603
            [args.python, "-c", source],
            capture_output=True, text=True, cwd=str(BACKEND), env=env,
        )
        if proc.returncode != 0:
            print(proc.stdout[-1500:])
            print(proc.stderr[-1500:], file=sys.stderr)
            return 1
        row = json.loads(proc.stdout.strip().splitlines()[-1])
        row["cold_download"] = bool(args.cold)
        results.append(row)

    label = "cold (download included)" if args.cold else "warm (model cached)"
    print(f"peak RSS on Linux, {label}, limit {args.limit_mb:.0f} MB "
          f"(= 512 MiB)\n")
    print(f"{'model':<8} {'cap':>5} {'wrk':>5} {'thr':>4} {'load':>8} "
          f"{'peak':>8} {'headroom':>9}  fits")
    print("-" * 62)
    for r in results:
        name = "int8" if "quantized" in r["onnx_file"] else "fp16"
        fits = "yes" if r["peak_mb"] < args.limit_mb else "NO"
        head = args.limit_mb - r["peak_mb"]
        print(f"{name:<8} {r['max_tokens']:>5} {str(r['worker'])[0]:>5} "
              f"{r['threads']:>4} {r['after_load_mb']:>7.0f}M "
              f"{r['peak_mb']:>7.0f}M {head:>8.0f}M  {fits}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"limit_mb": args.limit_mb, "platform": "linux", "results": results},
            indent=2,
        ))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
