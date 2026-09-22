"""
Assert that every number quoted in prose matches the artifact it came from.

Why this exists
---------------
The evaluation sets were regenerated once (a path-portability fix changed
chunk ids, and chunk ids seed the split). Six numbers quoted in docstrings
and BENCHMARKS.md silently went stale -- four retrieval metrics that had
simply moved, and two claims that had been rounded into being wrong:
0.9995 reported as "100.0%" and described as "lossless".

Measured claims that drift are worse than no claims, because they still
read as authoritative. This re-derives each quoted number from the JSON
artifact and fails if they disagree, so `python bench/check_documented_numbers.py`
is a pre-commit gate rather than a memory exercise.

Exit code 0 = every documented number matches its artifact.
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
from pathlib import Path

BENCH = Path(__file__).parent
ROOT = BENCH.parent
RESULTS = BENCH / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def metric(artifact: str, config: str, key: str) -> float:
    return load(artifact)["configs"][config][key]["value"]


def colocated(label: str, field: str) -> float:
    for row in load("colocated_rss.json")["results"]:
        if row["label"] == label:
            return row[field]
    raise KeyError(label)


def indexing(artifact: str, field: str) -> float:
    """A figure from a real indexing run, not a loaded-but-idle process."""
    return load(artifact)[field]


def linux_mem(onnx: str, cap: int, field: str) -> float:
    for row in load("memory_linux.json")["results"]:
        if row["onnx_file"].endswith(onnx) and row["max_tokens"] == cap and row["worker"]:
            return row[field]
    raise KeyError((onnx, cap))


def shipped(artifact: str, config: str, metric: str) -> float:
    return load(artifact)["configs"][config][metric]["value"]


def repro(config: str, metric: str) -> float:
    return load("eval_reproducibility.json")["configs"][config][metric]["spread"]


def batch(size: int, field: str) -> float:
    for row in load("embed_batch.json")["results"]:
        if row["batch"] == size:
            return row[field]
    raise KeyError(size)


def quant(fetch, threshold="zero") -> float:
    for r in load("quantization.json")["results"]:
        if r["fetch"] == fetch and (r["threshold"] or "zero") == threshold:
            return r["recall"]
    raise KeyError(f"quantization fetch={fetch} threshold={threshold}")


# (file, literal text that must appear, expected value, tolerance)
# Percentage-formatted claims declare scale=100 so a fraction from the
# artifact is compared against the percent written in the prose.
# The literal is searched verbatim, so a reworded sentence fails loudly
# rather than passing because the number happened to survive elsewhere.
CHECKS: list[tuple[str, str, float, float]] = [
    # --- retrieval, held-out test split (BENCHMARKS section 7) ---------
    ("backend/app/routes/chat.py", "recall@1 0.680 to 0.937",
     metric("symbol_v1_TEST.json", "dense only", "recall@1"), 0.0005),
    ("backend/app/routes/chat.py", "recall@1 0.680 to 0.937",
     metric("symbol_v1_TEST.json", "ROUTED (shipping)", "recall@1"), 0.0005),
    ("backend/app/retrieval/context.py", "(0.680 -> 0.937 on identifier",
     metric("symbol_v1_TEST.json", "ROUTED (shipping)", "recall@1"), 0.0005),

    # --- retrieval, dev split (the tuning numbers in the module header) -
    ("backend/app/retrieval/hybrid.py", "recall@10 0.908   MRR 0.737",
     metric("retrieval_v1_dev.json", "dense only", "mrr"), 0.0005),
    ("backend/app/retrieval/hybrid.py", "recall@10 0.285   MRR 0.234",
     metric("retrieval_v1_dev.json", "symbol only", "mrr"), 0.0005),
    ("backend/app/retrieval/hybrid.py", "recall@10 0.953   MRR 0.948",
     metric("symbol_v1_dev.json", "symbol only", "mrr"), 0.0005),
    ("backend/app/retrieval/hybrid.py", "recall@10 0.990   MRR 0.960",
     metric("symbol_v1_dev.json", "hybrid w=.5,.1,2", "mrr"), 0.0005),
    ("backend/app/retrieval/hybrid.py", "recall@10 0.813 vs 0.908",
     metric("retrieval_v1_dev.json", "hybrid w=1,1,1", "recall@10"), 0.0005),
    ("backend/app/retrieval/query_embedder.py", "MRR 0.948",
     metric("symbol_v1_dev.json", "symbol only", "mrr"), 0.0005),

    # --- quantisation: the two that were rounded into a false claim ----
    ("BENCHMARKS.md", "| halfvec exact, no ANN | - | 99.95% | - |",
     quant(None) * 100, 0.01),
    ("BENCHMARKS.md", "| **binary + rescore** | **100** | **99.95%** | **3.7** |",
     quant(100) * 100, 0.01),
    ("BENCHMARKS.md", "| binary + rescore | 200 | 100.00% | 4.4 |",
     quant(200) * 100, 0.01),
    ("backend/app/indexing/embedder.py", "0.9995-1.0000\n    recall@10",
     quant(100), 0.0001),

    # --- embedder variants ----------------------------------------------
    # These are the corrected figures. The previous ones claimed fp16 was a
    # free 1.36x speedup; re-measurement showed 0.76x. This check is the
    # reason a future re-measurement cannot quietly diverge from the prose
    # again.
    ("backend/app/indexing/embedder.py", "fp32  5.21 chunks/s",
     load("embed_model.json")["variants"]["fp32"]["chunks_per_sec"], 0.005),
    ("backend/app/indexing/embedder.py", "fp16  3.94 chunks/s",
     load("embed_model.json")["variants"]["fp16"]["chunks_per_sec"], 0.005),
    ("backend/app/indexing/embedder.py", "int8  9.70 chunks/s",
     load("embed_model.json")["variants"]["int8"]["chunks_per_sec"], 0.005),
    ("BENCHMARKS.md", "| fp32 | 5.21 | 1.00x | 100.0% | 1.0000 |",
     load("embed_model.json")["variants"]["fp32"]["chunks_per_sec"], 0.005),
    ("BENCHMARKS.md", "| **fp16** | **3.94** | **0.76x** | **100.0%** | **1.0000** |",
     load("embed_model.json")["variants"]["fp16"]["chunks_per_sec"], 0.005),
    ("BENCHMARKS.md", "| int8 | 9.70 | 1.86x | 90.5% | 0.9831 |",
     load("embed_model.json")["variants"]["int8"]["chunks_per_sec"], 0.005),

    # --- deployability: the numbers the topology rests on -----------------
    ("BENCHMARKS.md", "| indexer: fp16, arena off | 439 MB | 2.33 | 99 ms |",
     colocated("fp16, arena off", "peak_mb"), 0.5),
    ("BENCHMARKS.md", "| indexer: fp16, arena off | 439 MB | 2.33 | 99 ms |",
     colocated("fp16, arena off", "chunks_per_sec"), 0.005),
    ("BENCHMARKS.md", "| indexer + API in one process | 485 MB | 2.24 | 98 ms |",
     colocated("fp16, arena off + API", "peak_mb"), 0.5),
    ("BENCHMARKS.md", "| fp16, arena on | 1,809 MB | 3.40 | 55 ms |",
     colocated("fp16, arena on", "chunks_per_sec"), 0.005),
    ("BENCHMARKS.md", "| fp32, arena off | 734 MB | 3.24 | 21 ms |",
     colocated("fp32, arena off", "chunks_per_sec"), 0.005),
    ("BENCHMARKS.md", "| API alone | 59.1 MB | - | - |",
     colocated("fp16, arena off + API", "api_only_mb"), 0.05),
    ("backend/app/indexing/service.py", "app.main (API) alone                 59.1 MB",
     colocated("fp16, arena off + API", "api_only_mb"), 0.05),
    # The corrected arena claim, in the docstring that previously carried
    # the wrong one.
    ("backend/app/indexing/embedder.py",
     "arena ON     430 MB    301 s    fits, 107 MB spare, and faster",
     indexing("index_memory_arena_on.json", "peak_rss_mb"), 1.0),
    ("backend/app/indexing/embedder.py",
     "arena off    574 MB    343 s    killed by the platform",
     indexing("index_memory.json", "peak_rss_mb"), 1.0),

    # --- batch size: the reason INDEXER_EMBED_BATCH is 1 ------------------
    ("BENCHMARKS.md", "| **1** | **2.43** | **0.25 s** | **1.15 s** | 2.18 s |",
     batch(1, "median_hold_s"), 0.005),
    ("BENCHMARKS.md", "| 16 | 2.32 | 4.97 s | 16.74 s | 18.07 s |",
     batch(16, "median_hold_s"), 0.005),
    ("backend/app/config.py", "batch  1   2.45 chunks/s   median hold 0.25s",
     batch(1, "median_hold_s"), 0.005),

    # --- the memory numbers the deployment topology now rests on ----------
    # These replace the Windows figures that predicted a fit and got an OOM.
    ("BENCHMARKS.md", "| fp16 | 512 | 1,135 MB | −598 MB | no |",
     linux_mem("model_fp16.onnx", 512, "peak_mb"), 1.0),
    ("BENCHMARKS.md", "| int8 | 1024 | 619 MB | −82 MB | no |",
     linux_mem("model_quantized.onnx", 1024, "peak_mb"), 1.0),
    ("BENCHMARKS.md", "| int8 | 512 | 469 MB | 68 MB | yes |",
     linux_mem("model_quantized.onnx", 512, "peak_mb"), 1.0),
    ("BENCHMARKS.md", "| **int8** | **384** | **432 MB** | **105 MB** | **yes** |",
     linux_mem("model_quantized.onnx", 384, "peak_mb"), 1.0),

    # --- the arena, which was the actual fix ------------------------------
    ("BENCHMARKS.md", "| arena **off** | 574 MB | 343 s | killed by the platform |",
     indexing("index_memory.json", "peak_rss_mb"), 1.0),
    ("BENCHMARKS.md", "| **arena ON** | **430 MB** | **301 s** | fits, 107 MB spare |",
     indexing("index_memory_arena_on.json", "peak_rss_mb"), 1.0),
    ("BENCHMARKS.md", "| `MALLOC_ARENA_MAX=2` | 565 MB | within noise |",
     indexing("index_memory_arena2.json", "peak_rss_mb"), 1.0),
    ("BENCHMARKS.md", "| `INDEX_BATCH_SIZE` 64 -> 8 | 596 MB | worse |",
     indexing("index_memory_b8.json", "peak_rss_mb"), 1.0),
    ("BENCHMARKS.md", "| `enable_mem_pattern=False` | 573 MB | no effect |",
     indexing("index_memory_nopattern.json", "peak_rss_mb"), 1.0),
    ("BENCHMARKS.md", "1,024 tokens peaks at 747 MB and still does not fit",
     indexing("index_memory_arena_on_t1024.json", "peak_rss_mb"), 1.0),

    # --- what the shipped embedding config costs --------------------------
    ("BENCHMARKS.md", "| semantic | ROUTED | mrr | 0.707 | 0.696 | -0.010 |",
     shipped("retrieval_v1_TEST_int8_t384.json", "ROUTED (shipping)", "mrr"), 0.0005),
    ("BENCHMARKS.md", "| identifier | ROUTED | recall@10 | 0.997 | 0.997 | +0.000 |",
     shipped("symbol_v1_TEST_int8_t384.json", "ROUTED (shipping)", "recall@10"), 0.0005),
    ("BENCHMARKS.md", "| identifier | dense only | mrr | 0.774 | 0.797 | +0.023 |",
     shipped("symbol_v1_TEST_int8_t384.json", "dense only", "mrr"), 0.0005),

    # --- eval reproducibility: the caveat on the last digit ---------------
    ("BENCHMARKS.md", "| dense only | recall@10 | 0.0067 |",
     repro("dense only", "recall@10"), 0.00005),
    ("BENCHMARKS.md", "| hybrid w=.5,.1,2 | recall@10 | 0.0033 |",
     repro("hybrid w=.5,.1,2", "recall@10"), 0.00005),
    ("BENCHMARKS.md", "| ROUTED | MRR | 0.0008 |",
     repro("ROUTED (shipping)", "mrr"), 0.00005),
    # The control: it must be exactly zero, or the isolation argument fails.
    ("BENCHMARKS.md", "| **symbol only** | *every metric* | **0.0000** |",
     repro("symbol only", "recall@10"), 0.0),
    ("BENCHMARKS.md", "| **lexical only** | *every metric* | **0.0000** |",
     repro("lexical only", "mrr"), 0.0),
]


def numbers_in(text: str) -> list[float]:
    r"""
    Every number the prose states, integers included.

    An earlier version matched only `\d+\.\d+`, so a claim written as
    "439 MB" was invisible to the checker and passed trivially. Thousands
    separators are stripped so "1,809" compares as 1809.
    """
    out = []
    for raw in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        with contextlib.suppress(ValueError):  # the regex cannot produce this
            out.append(float(raw.replace(",", "")))
    return out


def agrees(found: list[float], expected: float, tol: float) -> bool:
    """
    True if any number in the prose is the artifact value, allowing for the
    precision the prose chose to display it at.
    """
    for value in found:
        if abs(value - expected) <= tol:
            return True
        text = f"{value}"
        decimals = len(text.split(".")[1]) if "." in text else 0
        if abs(round(expected, decimals) - value) < 1e-9:
            return True
    return False


def main() -> int:
    failures: list[str] = []
    for rel, literal, expected, tol in CHECKS:
        path = ROOT / rel
        if not path.exists():
            failures.append(f"{rel}: file missing")
            continue
        body = path.read_text(encoding="utf-8")
        if literal not in body:
            failures.append(f"{rel}: text not found: {literal!r}")
            continue
        found = numbers_in(literal)
        if not agrees(found, expected, tol):
            failures.append(
                f"{rel}: {literal!r} does not contain {expected:.6f} "
                f"(numbers present: {found})"
            )

    print(f"checked {len(CHECKS)} documented numbers against "
          f"{len({c[0] for c in CHECKS})} files")
    if failures:
        print(f"\n{len(failures)} MISMATCH(ES):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all documented numbers match their artifacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
