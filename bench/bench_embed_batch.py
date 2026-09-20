"""
Batch size against throughput and lock-hold time, for the shared embedder.

Why this matters
----------------
The indexer serves query embeddings and runs the indexing worker in one
process, sharing one model behind a lock (app/indexing/service.py). The
worker holds that lock for one sub-batch at a time, so the batch size sets
the worst case a query can wait.

The assumption worth testing is that bigger batches are faster. With the
ONNX memory arena disabled -- which is mandatory, see
bench_colocated_rss.py -- they are not:

    batch 1     2.45 chunks/s    median hold 0.25s    p95 1.11s
    batch 16    2.41 chunks/s    median hold 4.84s    p95 16.17s

Throughput is flat and hold time scales linearly, so batching costs query
latency and returns nothing. The mechanism is padding: every sequence in a
batch is padded to the longest in it, and a batch of one has no padding at
all. Length-sorting recovers most of that within a batch (2.75x, see
section 6) but cannot beat not padding in the first place.

`INDEXER_EMBED_BATCH` defaults to 1 because of this.

Usage:
    python bench_embed_batch.py --chunks 200 --reps 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).parent
sys.path.insert(0, str(BENCH_DIR.parent / "backend"))


def collect(n: int) -> list[str]:
    from app.indexing.chunker import ASTChunker
    from app.indexing.tokens import get_token_counter

    chunker = ASTChunker(count_tokens=get_token_counter().count)
    texts: list[str] = []
    for repo in sorted((BENCH_DIR / "corpus").iterdir()):
        if not repo.is_dir():
            continue
        for path in sorted(repo.rglob("*.py")):
            if len(texts) >= n:
                break
            try:
                src = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for chunk in chunker.chunk_file(
                repo.name, path.relative_to(repo).as_posix(), src
            ):
                texts.append(chunk.content)
                if len(texts) >= n:
                    break
        if len(texts) >= n:
            break
    return texts[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=200)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    from app.indexing.embedder import OnnxEmbedder

    texts = collect(args.chunks)
    print(f"{len(texts)} real chunks, fp16, arena off (the shipped config)")

    embedder = OnnxEmbedder()
    embedder.embed(texts[:4], batch_size=4)  # warm the graph before timing

    results = []
    print(f"\n{'batch':>6} {'chunks/s':>9} {'median hold':>12} {'p95 hold':>9} "
          f"{'max hold':>9}")
    print("-" * 50)
    for batch in args.batches:
        rates, holds = [], []
        for _ in range(args.reps):
            # Mirror SharedEmbedder: sort once, then time each sub-batch.
            order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            t_all = time.perf_counter()
            for start in range(0, len(order), batch):
                idx = order[start : start + batch]
                t0 = time.perf_counter()
                embedder.embed(
                    [texts[i] for i in idx],
                    batch_size=len(idx),
                    sort_by_length=False,
                )
                holds.append(time.perf_counter() - t0)
            rates.append(len(texts) / (time.perf_counter() - t_all))

        holds.sort()
        row = {
            "batch": batch,
            "chunks_per_sec": round(statistics.median(rates), 2),
            "rates": [round(r, 2) for r in rates],
            "median_hold_s": round(statistics.median(holds), 2),
            "p95_hold_s": round(holds[int(len(holds) * 0.95)], 2),
            "max_hold_s": round(holds[-1], 2),
        }
        results.append(row)
        print(f"{batch:>6} {row['chunks_per_sec']:>9.2f} "
              f"{row['median_hold_s']:>11.2f}s {row['p95_hold_s']:>8.2f}s "
              f"{row['max_hold_s']:>8.2f}s")

    best = max(results, key=lambda r: r["chunks_per_sec"])
    spread = best["chunks_per_sec"] - min(r["chunks_per_sec"] for r in results)
    print(f"\nthroughput spread across batch sizes: {spread:.2f} chunks/s "
          f"({spread / best['chunks_per_sec']:.0%})")
    print(f"hold time at batch {results[-1]['batch']} is "
          f"{results[-1]['median_hold_s'] / results[0]['median_hold_s']:.0f}x "
          f"that of batch {results[0]['batch']}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"chunks": len(texts), "reps": args.reps, "results": results}, indent=2
        ))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
