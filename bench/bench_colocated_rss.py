"""
The memory measurements the deployment topology rests on.

The decision this settles
-------------------------
`app/retrieval/query_embedder.py` embeds the query by calling an HTTP
endpoint, because the API container is supposed to stay thin. Nothing
served that endpoint, so the dense arm was disabled in any real deployment
and the retrieval work in section 7 was unreachable.

Three ways to fix it:

  A. API loads the model itself. No network hop, simplest code.
  B. Scale-to-zero indexer serves /embed. Free, but a cold start loads the
     weights while a user waits.
  C. Always-on indexer serves /embed and runs the worker. One extra
     instance, but the worker already holds the model, so serving queries
     from it costs nothing extra.

A is only viable if the API plus the model fits the instance with room to
serve requests. That is a measurement, so this measures it -- and, since
ONNX Runtime's memory arena turned out to dominate the process, measures
the arena and the model precision too.

Everything is measured on real chunks from the benchmark corpus, because
sequence length drives both the time and the peak.

Usage:
    python bench_colocated_rss.py --python bench/venvs/proposed/Scripts/python.exe
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Each variant runs in a fresh interpreter. Measuring several in one process
# attributes the first one's allocator growth to whoever runs after it, which
# is exactly the confound that inverted an earlier fp16 claim.
CHILD = r'''
import json, sys, time
from pathlib import Path
import psutil
_P = psutil.Process()
def rss(): return _P.memory_info().rss / (1024 * 1024)

model_file = sys.argv[1]
arena = sys.argv[2] == "on"
with_api = sys.argv[3] == "yes"
n_chunks = int(sys.argv[4])

out = {"model": model_file, "arena": arena, "with_api": with_api}
baseline = rss()

if with_api:
    import app.main  # noqa: F401
    out["api_only_mb"] = round(rss() - baseline, 1)

from app.indexing.chunker import ASTChunker
from app.indexing.embedder import OnnxEmbedder
from app.indexing.tokens import get_token_counter

chunker = ASTChunker(count_tokens=get_token_counter().count)
texts = []
corpus = Path("../bench/corpus")
if not corpus.exists():
    corpus = Path("bench/corpus")
for repo in sorted(corpus.iterdir()):
    if not repo.is_dir():
        continue
    for path in sorted(repo.rglob("*.py")):
        if len(texts) >= n_chunks:
            break
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for c in chunker.chunk_file(repo.name, path.relative_to(repo).as_posix(), src):
            texts.append(c.content)
            if len(texts) >= n_chunks:
                break
    if len(texts) >= n_chunks:
        break
texts = texts[:n_chunks]

t0 = time.perf_counter()
emb = OnnxEmbedder(onnx_file=model_file, enable_mem_arena=arena)
out["load_seconds"] = round(time.perf_counter() - t0, 2)
out["after_load_mb"] = round(rss(), 1)

emb.embed(texts[:4], batch_size=4)          # warm the graph before timing
out["after_warm_mb"] = round(rss(), 1)

t0 = time.perf_counter()
emb.embed(texts, batch_size=16)
dt = time.perf_counter() - t0
out["chunks"] = len(texts)
out["chunks_per_sec"] = round(len(texts) / dt, 2)
out["after_batch_mb"] = round(rss(), 1)

qs = ["how does the retry budget work", "what validates the clone url",
      "where is the job queue reaped", "how are chunks deduplicated"]
t0 = time.perf_counter()
for _ in range(5):
    for q in qs:
        emb.embed([q], batch_size=1)
out["query_ms"] = round((time.perf_counter() - t0) / (5 * len(qs)) * 1000, 1)
out["peak_mb"] = round(rss(), 1)
print(json.dumps(out))
'''

VARIANTS = [
    ("onnx/model_fp16.onnx", "off", "no"),
    ("onnx/model_fp16.onnx", "on", "no"),
    ("onnx/model.onnx", "off", "no"),
    ("onnx/model_fp16.onnx", "off", "yes"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--chunks", type=int, default=200)
    ap.add_argument("--budget", type=float, default=512.0)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    python = str(Path(args.python).resolve())
    results = []
    for model, arena, with_api in VARIANTS:
        label = f"{'fp16' if 'fp16' in model else 'fp32'}, arena {arena}" + (
            " + API" if with_api == "yes" else ""
        )
        print(f"-- {label}", flush=True)
        proc = subprocess.run(  # noqa: S603
            [python, "-c", CHILD, model, arena, with_api, str(args.chunks)],
            capture_output=True, text=True, cwd=str(ROOT / "backend"),
        )
        if proc.returncode != 0:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:], file=sys.stderr)
            return 1
        row = json.loads(proc.stdout.strip().splitlines()[-1])
        row["label"] = label
        results.append(row)
        print(f"   peak {row['peak_mb']:.0f} MB   {row['chunks_per_sec']} chunks/s   "
              f"{row['query_ms']} ms/query", flush=True)

    print(f"\n{'configuration':<26} {'peak RSS':>9} {'chunks/s':>9} {'query':>8} "
          f"{'fits 512MB':>11}")
    print("-" * 68)
    for r in results:
        fits = "yes" if r["peak_mb"] < args.budget else "NO"
        print(f"{r['label']:<26} {r['peak_mb']:>8.0f}M {r['chunks_per_sec']:>9} "
              f"{r['query_ms']:>7}ms {fits:>11}")

    api_only = next((r["api_only_mb"] for r in results if "api_only_mb" in r), None)
    if api_only is not None:
        print(f"\nAPI import alone: {api_only:.1f} MB")

    combined = next((r for r in results if r["with_api"]), None)
    shipped = next(r for r in results if not r["with_api"]
                   and r["arena"] is False and "fp16" in r["model"])
    if combined:
        head = args.budget - combined["peak_mb"]
        print(f"colocating API + model peaks at {combined['peak_mb']:.0f} MB "
              f"({head:.0f} MB headroom) -- "
              f"{'viable' if head > 100 else 'TOO TIGHT to serve under load'}")
    print(f"shipped indexer config (fp16, arena off): {shipped['peak_mb']:.0f} MB, "
          f"{args.budget - shipped['peak_mb']:.0f} MB headroom")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"budget_mb": args.budget, "chunks": args.chunks, "results": results},
            indent=2,
        ))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
