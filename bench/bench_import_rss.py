"""
Measure the import-time resident-set cost of the API server's dependency set.

Why this benchmark exists
-------------------------
The deployment target is a 512 MB container. Import-time RSS is spent before
the process serves a single request, so it is a hard subtraction from the
request budget -- not an average, not something that amortises.

The v1 stack imports chromadb and the langchain family inside the web process.
This measures what that costs and what removing it returns.

Two modes, because they answer different questions:

  cumulative  -- all modules imported into one interpreter, in order.
                 This is what the application actually does, so the total is
                 the number that matters operationally. Per-module deltas are
                 *not* standalone costs: shared transitive dependencies are
                 attributed to whichever module pulls them in first.

  isolated    -- one fresh interpreter per module.
                 Gives each module's true standalone cost. These sum to more
                 than the cumulative total, by exactly the shared deps.

Repetitions are run because RSS is noisy (allocator behaviour, page
granularity); we report the median.

Usage:
    python bench_import_rss.py --python <path-to-venv-python> --set current
    python bench_import_rss.py --compare      # runs both venvs, prints delta
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).parent
PROBE = BENCH_DIR / "rss_probe.py"

# The module sets under comparison.
#
# "current" mirrors what backend/app/main.py transitively imports today.
# "proposed" mirrors the pyproject.toml [project.dependencies] serving set.
MODULE_SETS: dict[str, list[str]] = {
    "current": [
        "fastapi",
        "uvicorn",
        "sqlalchemy.ext.asyncio",
        "asyncpg",
        "chromadb",
        "langchain",
        "langchain_community.llms",
        "langchain_chroma",
        "langgraph.graph",
        "redis.asyncio",
        "git",
        "slowapi",
        "jose",
        "passlib.context",
    ],
    "proposed": [
        "fastapi",
        "uvicorn",
        "sqlalchemy.ext.asyncio",
        "asyncpg",
        "pgvector.sqlalchemy",
        "httpx",
        "langgraph.graph",
        "jwt",
        "bcrypt",
        "tokenizers",
    ],
    # The indexer set runs out-of-process (Modal / separate worker). Measured
    # separately to show it is genuinely heavy and genuinely separable.
    "indexer": [
        "tree_sitter",
        "tree_sitter_language_pack",
        "onnxruntime",
        "numpy",
    ],
}


def run_probe(python: str, modules: list[str], isolated: bool) -> dict:
    """Run the probe once; return its parsed JSON."""
    if not isolated:
        out = subprocess.run(
            [python, str(PROBE), *modules],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if out.returncode != 0:
            raise RuntimeError(f"probe failed: {out.stderr[-2000:]}")
        return json.loads(out.stdout.strip().splitlines()[-1])

    # Isolated: a fresh interpreter per module.
    steps = []
    for mod in modules:
        out = subprocess.run(
            [python, str(PROBE), mod],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if out.returncode != 0:
            steps.append({"module": mod, "delta_mb": None, "error": out.stderr[-200:]})
            continue
        payload = json.loads(out.stdout.strip().splitlines()[-1])
        steps.append(payload["steps"][0])
    return {"steps": steps, "total_mb": None, "baseline_mb": None}


def measure(python: str, set_name: str, reps: int, isolated: bool) -> dict:
    """Run `reps` repetitions, return the median-total run plus spread."""
    modules = MODULE_SETS[set_name]
    runs = [run_probe(python, modules, isolated) for _ in range(reps)]

    if isolated:
        return {"set": set_name, "mode": "isolated", "runs": runs, "representative": runs[0]}

    totals = [r["total_mb"] for r in runs]
    median_total = statistics.median(totals)
    # Pick the run closest to the median as the representative breakdown.
    rep = min(runs, key=lambda r: abs(r["total_mb"] - median_total))
    return {
        "set": set_name,
        "mode": "cumulative",
        "reps": reps,
        "totals_mb": totals,
        "median_total_mb": round(median_total, 2),
        "min_total_mb": round(min(totals), 2),
        "max_total_mb": round(max(totals), 2),
        "baseline_mb": rep["baseline_mb"],
        "python": rep["python"],
        "representative": rep,
    }


def print_breakdown(result: dict) -> None:
    rep = result["representative"]
    print(f"\n  {'module':<34} {'delta MB':>10} {'cumul MB':>10} {'import s':>9}")
    print(f"  {'-' * 34} {'-' * 10} {'-' * 10} {'-' * 9}")
    for step in rep["steps"]:
        if step.get("error"):
            print(f"  {step['module']:<34} {'FAILED':>10}   {step['error'][:40]}")
            continue
        print(
            f"  {step['module']:<34} {step['delta_mb']:>10.1f} "
            f"{step.get('cumulative_mb', float('nan')):>10.1f} "
            f"{step.get('import_seconds', 0):>9.2f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", help="interpreter to probe with")
    ap.add_argument("--set", dest="set_name", choices=sorted(MODULE_SETS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--isolated", action="store_true")
    ap.add_argument("--json-out", type=Path)
    ap.add_argument(
        "--compare",
        action="store_true",
        help="measure current vs proposed vs indexer using bench/venvs/*",
    )
    args = ap.parse_args()

    results = {}

    if args.compare:
        cur_py = BENCH_DIR / "venvs" / "current" / "Scripts" / "python.exe"
        new_py = BENCH_DIR / "venvs" / "proposed" / "Scripts" / "python.exe"
        idx_py = BENCH_DIR / "venvs" / "indexer" / "Scripts" / "python.exe"
        if not cur_py.exists():
            cur_py = BENCH_DIR / "venvs" / "current" / "bin" / "python"
            new_py = BENCH_DIR / "venvs" / "proposed" / "bin" / "python"
            idx_py = BENCH_DIR / "venvs" / "indexer" / "bin" / "python"

        plan = [("current", cur_py), ("proposed", new_py), ("indexer", idx_py)]
        for name, py in plan:
            if not Path(py).exists():
                print(f"  ! skipping {name}: no interpreter at {py}")
                continue
            print(f"\n=== {name} (cumulative, {args.reps} reps) ===")
            res = measure(str(py), name, args.reps, isolated=False)
            results[name] = res
            print(
                f"  baseline {res['baseline_mb']:.1f} MB | "
                f"median total {res['median_total_mb']:.1f} MB "
                f"(min {res['min_total_mb']:.1f} / max {res['max_total_mb']:.1f})"
            )
            print_breakdown(res)

        if "current" in results and "proposed" in results:
            cur = results["current"]["median_total_mb"]
            new = results["proposed"]["median_total_mb"]
            print("\n" + "=" * 62)
            print(f"  current  serving-set import RSS : {cur:>8.1f} MB")
            print(f"  proposed serving-set import RSS : {new:>8.1f} MB")
            print(f"  reclaimed                       : {cur - new:>8.1f} MB "
                  f"({(cur - new) / cur * 100:.0f}%)")
            if "indexer" in results:
                print(f"  indexer set (out-of-process)    : "
                      f"{results['indexer']['median_total_mb']:>8.1f} MB")
            print("=" * 62)
    else:
        if not args.python or not args.set_name:
            ap.error("--python and --set are required unless --compare is used")
        res = measure(args.python, args.set_name, args.reps, args.isolated)
        results[args.set_name] = res
        if not args.isolated:
            print(
                f"baseline {res['baseline_mb']:.1f} MB | "
                f"median total {res['median_total_mb']:.1f} MB"
            )
        print_breakdown(res)

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
