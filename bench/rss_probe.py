"""
RSS probe -- child process for bench_import_rss.py.

Imports a list of modules, reporting resident-set size after each one.
Runs as its own process so the measurement starts from a clean interpreter.

psutil is imported *before* the baseline is taken, so its own footprint is
excluded from every reported delta.

Usage:
    python rss_probe.py <module> [<module> ...]
Emits a single JSON object on stdout.
"""

import importlib
import json
import sys
import time

import psutil

_PROC = psutil.Process()


def rss_mb() -> float:
    return _PROC.memory_info().rss / (1024 * 1024)


def main() -> int:
    modules = sys.argv[1:]

    # Baseline taken after psutil is loaded: everything below is attributable
    # to the modules under test, not to the measurement apparatus.
    baseline = rss_mb()

    steps = []
    for name in modules:
        before = rss_mb()
        t0 = time.perf_counter()
        error = None
        try:
            importlib.import_module(name)
        except BaseException as exc:  # noqa: BLE001 - a failed import is data
            error = f"{type(exc).__name__}: {exc}"[:300]
        elapsed = time.perf_counter() - t0
        after = rss_mb()

        steps.append(
            {
                "module": name,
                "delta_mb": round(after - before, 2),
                "cumulative_mb": round(after - baseline, 2),
                "import_seconds": round(elapsed, 4),
                "error": error,
            }
        )

    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "baseline_mb": round(baseline, 2),
                "total_mb": round(rss_mb() - baseline, 2),
                "peak_rss_mb": round(rss_mb(), 2),
                "steps": steps,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
