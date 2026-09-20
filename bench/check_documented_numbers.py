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

    # --- embedder variants ---------------------------------------------
    ("backend/app/indexing/embedder.py", "fp16  1.22 chunks/s",
     load("embed_model.json")["variants"]["fp16"]["chunks_per_sec"], 0.005),
    ("backend/app/indexing/embedder.py", "int8  1.89 chunks/s",
     load("embed_model.json")["variants"]["int8"]["chunks_per_sec"], 0.005),
]


def numbers_in(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"\d+\.\d+", text)]


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
        # The literal must actually contain the artifact's value, rounded
        # the way it is written. Any of its numbers may be the match.
        if not any(abs(f - round(expected, len(str(f).split('.')[1]))) <= tol
                   for f in found):
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
