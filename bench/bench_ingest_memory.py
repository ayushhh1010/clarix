"""
Measure peak resident memory of the v1 ingestion pipeline's data structures.

Motivation
----------
bench_import_rss.py showed the v1 dependency set imports in ~125 MB, which is
comfortable inside a 512 MB container. So imports did not cause the OOM that
commit 46925d0 ("replace local embeddings with HF Inference API to fix OOM on
Render free tier") was fighting. Something else did.

`run_ingestion_pipeline` holds four large structures alive simultaneously at
the moment it calls `store_chunks`:

    parsed_files = list(parse_repository(...))   # every file's full text
    chunks       = chunk_repository(...)         # every chunk's text again
    embeddings   = await embed_chunks(chunks)    # list[list[float]]
    store_chunks(repo_id, chunks, embeddings)    # all four still referenced

This measures each, because the representation of `embeddings` in particular
is far more expensive than its nominal size: a Python list of 384 floats is
not 1,536 bytes. Each float is a boxed 24-byte object plus an 8-byte pointer,
so the real cost is roughly 8x the numeric payload.

Usage:
    python bench_ingest_memory.py --chunks 10000 25000 50000
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import string
import sys
from pathlib import Path

import psutil

_PROC = psutil.Process()

EMBED_DIM = 384  # v1: bge-small-en-v1.5
AVG_CHUNK_CHARS = 1400  # ~50 lines of source


def rss_mb() -> float:
    gc.collect()
    return _PROC.memory_info().rss / (1024 * 1024)


def make_text(n_chars: int, rng: random.Random) -> str:
    """A realistic-ish chunk body. Content is irrelevant; size is not."""
    alphabet = string.ascii_letters + string.digits + "  ()[]{}:.,_=\n"
    return "".join(rng.choices(alphabet, k=n_chars))


def measure(n: int) -> dict:
    rng = random.Random(20260919)
    base = rss_mb()
    out: dict[str, float] = {"chunks": n, "baseline_mb": round(base, 1)}

    # --- 1. chunk text, held as Python strings (v1 keeps all of these) ----
    texts = [make_text(AVG_CHUNK_CHARS, rng) for _ in range(n)]
    after_text = rss_mb()
    out["chunk_text_mb"] = round(after_text - base, 1)

    # --- 2. embeddings as list[list[float]] -- what the v1 embedder returns
    embeddings = [[rng.random() for _ in range(EMBED_DIM)] for _ in range(n)]
    after_py = rss_mb()
    out["embeddings_list_of_list_mb"] = round(after_py - after_text, 1)
    out["peak_v1_shape_mb"] = round(after_py - base, 1)

    del embeddings
    after_free = rss_mb()

    # --- 3. the same vectors as a numpy float32 matrix --------------------
    try:
        import numpy as np

        arr = np.asarray(
            np.random.default_rng(0).random((n, EMBED_DIM)), dtype=np.float32
        )
        after_np = rss_mb()
        out["embeddings_numpy_f32_mb"] = round(after_np - after_free, 1)

        half = arr.astype(np.float16)
        out["embeddings_numpy_f16_mb"] = round(half.nbytes / (1024 * 1024), 1)

        # int8, as stored in pgvector `halfvec`-adjacent quantisation
        out["embeddings_int8_mb"] = round(n * EMBED_DIM / (1024 * 1024), 1)

        # 1-bit packed, as stored in pgvector `bit(d)` for the ANN stage
        out["embeddings_binary_mb"] = round(n * EMBED_DIM / 8 / (1024 * 1024), 1)
        del arr, half
    except ImportError:
        out["numpy"] = "unavailable"

    del texts
    gc.collect()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, nargs="+", default=[10_000, 25_000, 50_000])
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    print(f"python {sys.version.split()[0]}  |  embedding dim {EMBED_DIM}  "
          f"|  avg chunk {AVG_CHUNK_CHARS} chars\n")

    results = [measure(n) for n in args.chunks]

    cols = [
        ("chunks", "chunks", "{:,}"),
        ("chunk text", "chunk_text_mb", "{:.0f} MB"),
        ("embeddings list[list[float]]", "embeddings_list_of_list_mb", "{:.0f} MB"),
        ("PEAK (v1 shape)", "peak_v1_shape_mb", "{:.0f} MB"),
        ("embeddings numpy f32", "embeddings_numpy_f32_mb", "{:.0f} MB"),
        ("embeddings f16", "embeddings_numpy_f16_mb", "{:.0f} MB"),
        ("embeddings int8", "embeddings_int8_mb", "{:.1f} MB"),
        ("embeddings binary", "embeddings_binary_mb", "{:.1f} MB"),
    ]

    header = f"{'metric':<30}" + "".join(f"{r['chunks']:>14,}" for r in results)
    print(header)
    print("-" * len(header))
    for label, key, fmt in cols:
        if key == "chunks":
            continue
        row = f"{label:<30}"
        for r in results:
            v = r.get(key)
            row += f"{fmt.format(v) if v is not None else '--':>14}"
        print(row)

    print("\nRatio of list[list[float]] to its numeric payload:")
    for r in results:
        payload = r["chunks"] * EMBED_DIM * 4 / (1024 * 1024)
        actual = r.get("embeddings_list_of_list_mb")
        if actual:
            print(
                f"  {r['chunks']:>7,} chunks: {actual:6.0f} MB actual vs "
                f"{payload:5.1f} MB as f32  ->  {actual / payload:4.1f}x overhead"
            )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
