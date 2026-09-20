"""Build the identifier-lookup evaluation set from the benchmark corpus."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).parent
sys.path.insert(0, str(BENCH_DIR.parent / "backend"))
sys.path.insert(0, str(BENCH_DIR))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=BENCH_DIR.parent / "backend" / "eval_data" / "symbol_v1.jsonl")
    ap.add_argument("--sample", type=int, default=600,
                    help="cap the set; every unique symbol x 4 templates is a lot")
    args = ap.parse_args()

    from app.evaluation.dataset import build_symbol_lookup, split, write_jsonl
    from bench_quantization import collect_chunks

    chunks = collect_chunks(100_000)
    by_repo: dict[str, list] = {}
    for c in chunks:
        by_repo.setdefault(c.repo_id, []).append(c)

    examples = []
    for repo, rc in sorted(by_repo.items()):
        got = list(build_symbol_lookup(rc, repo))
        print(f"  {repo:<12} {len(rc):>5} chunks -> {len(got):>5} queries")
        examples.extend(got)

    print(f"\n{len(examples):,} total")
    if args.sample and len(examples) > args.sample:
        # Sample whole symbols, not individual queries, so all four phrasings
        # of a symbol stay together and per-template breakdowns stay balanced.
        rng = random.Random(20260920)
        by_symbol: dict[str, list] = {}
        for e in examples:
            by_symbol.setdefault(e.gold_chunk_ids[0], []).append(e)
        keys = sorted(by_symbol)
        rng.shuffle(keys)
        keep = keys[: max(1, args.sample // len(__import__("app.evaluation.dataset",
                fromlist=["x"]).SYMBOL_QUERY_TEMPLATES))]
        examples = [e for k in keep for e in by_symbol[k]]
        print(f"sampled down to {len(examples):,} ({len(keep)} symbols x 4 phrasings)")

    dev, test = split(examples)
    write_jsonl(examples, args.out)
    write_jsonl(dev, args.out.with_name(args.out.stem + "_dev.jsonl"))
    write_jsonl(test, args.out.with_name(args.out.stem + "_test.jsonl"))
    print(f"wrote {args.out}  dev={len(dev)} test={len(test)}")
    for e in examples[:5]:
        print(f'  "{e.query}"  ->  {e.gold_symbol_path}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
