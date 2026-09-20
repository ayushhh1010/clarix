"""
Measure the recall cost of vector quantisation on REAL code embeddings.

Why this exists
---------------
verify_pgvector.py measured recall@10 of 18.5% for binary quantisation with
rescoring -- on random Gaussian vectors. That number is meaningless: random
vectors in 768 dimensions are near-orthogonal and carry no structure for
quantisation to preserve. It tells us nothing about real embeddings, and
would have been a catastrophic thing to design around either way.

This runs the same question against embeddings the real model produces for
real code from the benchmark corpus.

The storage question it settles
-------------------------------
Measured per-row costs (pgvector 0.8.6, 768 dims):

    halfvec value        1,544 B      halfvec HNSW index   ~2,050 B
    bit value              101 B      bit HNSW index         ~400 B

  A: halfvec + its HNSW index   3,594 B/row -> 719 MB @ 200k chunks  (over 500 MB free tier)
  B: halfvec stored, bit indexed  2,045 B/row -> 409 MB @ 200k chunks  (fits)
  C: bit only, no rescoring         501 B/row -> 100 MB @ 200k chunks  (recall unknown)

Plan B is only viable if two-stage recall is high enough. That is what this
measures, so the decision is made on data rather than on the shape of the
storage table.

Also tested: whether thresholding at the corpus centroid instead of zero
matters. pgvector's `binary_quantize()` thresholds at zero, which is only
correct if each dimension is zero-centred -- encoder outputs are not.

Usage:
    python bench_quantization.py --chunks 8000 --queries 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).parent
CORPUS_DIR = BENCH_DIR / "corpus"
CACHE = BENCH_DIR / "results" / "embeddings_cache.npz"
sys.path.insert(0, str(BENCH_DIR.parent / "backend"))

SKIP_PARTS = {".git", "node_modules", "venv", "__pycache__", "dist", "build", "vendor", "testdata"}
CODE_EXTS = {".py", ".js", ".ts", ".tsx", ".go"}


def collect_chunks(limit: int) -> list:
    from app.indexing.chunker import ASTChunker
    from app.indexing.tokens import get_token_counter

    counter = get_token_counter()
    chunker = ASTChunker(count_tokens=counter.count)

    out = []
    for repo_dir in sorted(CORPUS_DIR.iterdir()):
        if not repo_dir.is_dir():
            continue
        for path in sorted(repo_dir.rglob("*")):
            if len(out) >= limit:
                return out
            if not path.is_file() or path.suffix.lower() not in CODE_EXTS:
                continue
            if any(p in SKIP_PARTS for p in path.parts):
                continue
            try:
                src = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = path.relative_to(repo_dir).as_posix()
            out.extend(chunker.chunk_file(repo_dir.name, rel, src))
    return out[:limit]


def embed_corpus(chunks, batch: int) -> np.ndarray:
    from app.indexing.embedder import OnnxEmbedder

    emb = OnnxEmbedder()
    texts = [c.content for c in chunks]
    t0 = time.perf_counter()
    vecs = emb.embed(texts, batch_size=batch)
    dt = time.perf_counter() - t0
    print(f"  embedded {len(texts):,} chunks in {dt:.1f}s "
          f"({len(texts) / dt:.1f}/s, batch={batch})")
    return vecs


def recall_at_k(truth: np.ndarray, got: np.ndarray, k: int) -> float:
    """Mean overlap between two (queries, k) index matrices."""
    hits = sum(len(set(t[:k]) & set(g[:k])) for t, g in zip(truth, got, strict=True))
    return hits / (len(truth) * k)


def exact_topk(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Cosine top-k. Vectors are L2-normalised, so cosine == dot product."""
    sims = queries @ corpus.T
    idx = np.argpartition(-sims, kth=k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sims, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)


def hamming_topk(cbits: np.ndarray, qbits: np.ndarray, k: int) -> np.ndarray:
    """Top-k by Hamming distance over packed uint8 rows."""
    lut = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
    out = np.empty((len(qbits), k), dtype=np.int64)
    for i, q in enumerate(qbits):
        dist = lut[np.bitwise_xor(cbits, q)].sum(axis=1)
        idx = np.argpartition(dist, kth=k)[:k]
        out[i] = idx[np.argsort(dist[idx])]
    return out


def two_stage(
    corpus: np.ndarray, queries: np.ndarray, cbits: np.ndarray, qbits: np.ndarray,
    fetch: int, k: int,
) -> tuple[np.ndarray, float]:
    """Binary ANN to `fetch` candidates, then exact rescore to top-k."""
    t0 = time.perf_counter()
    cands = hamming_topk(cbits, qbits, fetch)
    out = np.empty((len(queries), k), dtype=np.int64)
    for i, cand in enumerate(cands):
        sims = corpus[cand] @ queries[i]
        top = np.argsort(-sims)[:k]
        out[i] = cand[top]
    return out, (time.perf_counter() - t0) / len(queries) * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=8000)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    if CACHE.exists() and not args.no_cache:
        data = np.load(CACHE)
        vecs = data["vecs"]
        print(f"loaded {len(vecs):,} cached embeddings from {CACHE.name}")
    else:
        print("chunking corpus...")
        chunks = collect_chunks(args.chunks)
        print(f"  {len(chunks):,} chunks")
        print("embedding (real model, ONNX CPU)...")
        vecs = embed_corpus(chunks, args.batch)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, vecs=vecs)

    n_q = min(args.queries, len(vecs) // 4)
    rng = np.random.default_rng(42)
    q_idx = rng.choice(len(vecs), size=n_q, replace=False)
    queries = vecs[q_idx]
    corpus = vecs
    k = args.k

    print(f"\ncorpus {len(corpus):,} x {corpus.shape[1]}  |  {n_q} queries  |  k={k}")

    # Sanity: are these embeddings actually structured, or degenerate?
    sims = (queries[:50] @ corpus.T)
    off = sims.copy()
    off[np.arange(50), q_idx[:50]] = -1
    print(f"similarity of a query to its nearest *other* chunk: "
          f"mean {off.max(axis=1).mean():.3f}   "
          f"(random-vector baseline would be ~0.0)")

    centroid = corpus.mean(axis=0)
    print(f"per-dimension mean: |mu| avg {np.abs(centroid).mean():.4f}, "
          f"max {np.abs(centroid).max():.4f}  "
          f"(zero-thresholding assumes this is ~0)")

    truth = exact_topk(corpus, queries, k)

    results = []

    # --- halfvec precision loss (no ANN involved) ------------------------
    half = corpus.astype(np.float16).astype(np.float32)
    half_q = queries.astype(np.float16).astype(np.float32)
    r = recall_at_k(truth, exact_topk(half, half_q, k), k)
    results.append({"method": "halfvec exact (no ANN)", "fetch": None,
                    "threshold": None, "recall": r, "ms": None})

    # --- binary, zero threshold (what pgvector's binary_quantize does) ---
    for thresh_name, thresh in (("zero", 0.0), ("centroid", centroid)):
        cbits = np.packbits(corpus > thresh, axis=1, bitorder="big")
        qbits = np.packbits(queries > thresh, axis=1, bitorder="big")
        for fetch in (k, 50, 100, 200, 500, 1000, 2000):
            if fetch > len(corpus):
                continue
            got, ms = two_stage(corpus, queries, cbits, qbits, fetch, k)
            results.append({
                "method": f"binary + rescore, {thresh_name} threshold",
                "fetch": fetch, "threshold": thresh_name,
                "recall": recall_at_k(truth, got, k), "ms": round(ms, 2),
            })

    print(f"\n{'method':<44} {'fetch':>6} {'recall@10':>10} {'ms/query':>9}")
    print("-" * 73)
    for r in results:
        f = str(r["fetch"]) if r["fetch"] else "-"
        m = f"{r['ms']:.2f}" if r["ms"] else "-"
        print(f"{r['method']:<44} {f:>6} {r['recall']:>9.1%} {m:>9}")

    best_zero = max((r for r in results if r["threshold"] == "zero"),
                    key=lambda r: r["recall"])
    best_cent = max((r for r in results if r["threshold"] == "centroid"),
                    key=lambda r: r["recall"])
    print(f"\nbest zero-threshold:     {best_zero['recall']:.1%} @ fetch={best_zero['fetch']}")
    print(f"best centroid-threshold: {best_cent['recall']:.1%} @ fetch={best_cent['fetch']}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "corpus": int(len(corpus)), "dim": int(corpus.shape[1]),
            "queries": int(n_q), "k": k,
            "centroid_abs_mean": float(np.abs(centroid).mean()),
            "results": results,
        }, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
