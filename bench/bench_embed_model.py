"""
Compare ONNX model variants for the embedder: throughput vs retrieval quality.

Why
---
bench_quantization.py measured fp32 embedding throughput at 1.1 chunks/sec on
this 2-core CPU. At that rate a 3,601-chunk repository takes 53 minutes to
index, which is not a viable ingestion time -- and Modal bills per second, so
it is also the dominant cost of the whole pipeline.

`jinaai/jina-embeddings-v2-base-code` ships three ONNX exports: fp32, fp16 and
an int8-quantised build. The question is whether the faster ones cost
retrieval quality, and that has to be measured rather than assumed -- int8
weight quantisation is not the same operation as the binary quantisation of
*output vectors* measured elsewhere, and there is no reason to expect the two
to behave alike.

Method
------
fp32 output is ground truth. For each variant: embed the same chunks, time it,
then score the variant's vectors as a retrieval index against fp32's top-10.
A variant that is fast but reorders results is not a win.

Usage:
    python bench_embed_model.py --chunks 600
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).parent
sys.path.insert(0, str(BENCH_DIR.parent / "backend"))

VARIANTS = {
    "fp32": "onnx/model.onnx",
    "fp16": "onnx/model_fp16.onnx",
    "int8": "onnx/model_quantized.onnx",
}


def exact_topk(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    sims = queries @ corpus.T
    idx = np.argpartition(-sims, kth=k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sims, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)


def recall_at_k(truth: np.ndarray, got: np.ndarray, k: int) -> float:
    return sum(len(set(t[:k]) & set(g[:k])) for t, g in zip(truth, got)) / (len(truth) * k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=600)
    ap.add_argument("--queries", type=int, default=150)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    from bench_quantization import collect_chunks

    from app.indexing.embedder import OnnxEmbedder

    chunks = collect_chunks(args.chunks)
    texts = [c.content for c in chunks][: args.chunks]
    print(f"{len(texts)} chunks, batch={args.batch}, "
          f"threads={args.threads or 'default'}", flush=True)

    results: dict[str, dict] = {}
    vectors: dict[str, np.ndarray] = {}

    for name, path in VARIANTS.items():
        print(f"\n-- {name} ({path})", flush=True)
        try:
            emb = OnnxEmbedder(onnx_file=path, threads=args.threads or None)
        except Exception as exc:  # noqa: BLE001 - a missing export is data
            print(f"   unavailable: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            results[name] = {"error": f"{type(exc).__name__}"}
            continue

        t0 = time.perf_counter()
        vecs = emb.embed(texts, batch_size=args.batch)
        dt = time.perf_counter() - t0
        vectors[name] = vecs
        results[name] = {
            "seconds": round(dt, 1),
            "chunks_per_sec": round(len(texts) / dt, 2),
        }
        print(f"   {dt:.1f}s  ->  {len(texts) / dt:.2f} chunks/sec", flush=True)

    if "fp32" not in vectors:
        print("\nfp32 unavailable; cannot score quality")
        return 1

    rng = np.random.default_rng(11)
    q_idx = rng.choice(len(texts), size=min(args.queries, len(texts) // 2), replace=False)
    base = vectors["fp32"]
    truth = exact_topk(base, base[q_idx], args.k)

    for name, vecs in vectors.items():
        got = exact_topk(vecs, vecs[q_idx], args.k)
        results[name]["recall_vs_fp32"] = round(recall_at_k(truth, got, args.k), 4)
        # Cosine agreement between the two encoders on the same text: a
        # direct read on how much the weights moved.
        results[name]["mean_cosine_to_fp32"] = round(
            float((vecs * base).sum(axis=1).mean()), 4
        )

    print(f"\n{'variant':<10} {'chunks/sec':>11} {'speedup':>9} "
          f"{'recall@10':>10} {'cos(fp32)':>10}")
    print("-" * 54)
    fp32_rate = results["fp32"]["chunks_per_sec"]
    for name in VARIANTS:
        r = results.get(name, {})
        if "error" in r:
            print(f"{name:<10} {'unavailable':>11}")
            continue
        print(f"{name:<10} {r['chunks_per_sec']:>11.2f} "
              f"{r['chunks_per_sec'] / fp32_rate:>8.2f}x "
              f"{r['recall_vs_fp32']:>9.1%} {r['mean_cosine_to_fp32']:>10.4f}")

    print("\nprojected time to index 3,601 chunks:")
    for name in VARIANTS:
        r = results.get(name, {})
        if "error" not in r:
            print(f"  {name:<6} {3601 / r['chunks_per_sec'] / 60:6.1f} min")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"chunks": len(texts), "batch": args.batch,
             "threads": args.threads, "variants": results}, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
