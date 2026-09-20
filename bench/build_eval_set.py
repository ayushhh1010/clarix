"""
Build the retrieval evaluation set from the benchmark corpus.

Usage:
    python build_eval_set.py --out ../backend/eval_data/retrieval_v1.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).parent
sys.path.insert(0, str(BENCH_DIR.parent / "backend"))
sys.path.insert(0, str(BENCH_DIR))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=BENCH_DIR.parent / "backend" / "eval_data" / "retrieval_v1.jsonl")
    ap.add_argument("--limit", type=int, default=100_000)
    args = ap.parse_args()

    from bench_quantization import collect_chunks

    from app.evaluation.dataset import BuildStats, build_from_chunks, split, write_jsonl

    chunks = collect_chunks(args.limit)
    print(f"{len(chunks):,} chunks from the corpus")

    by_repo: dict[str, list] = {}
    for c in chunks:
        by_repo.setdefault(c.repo_id, []).append(c)

    stats = BuildStats()
    examples = []
    for repo, repo_chunks in sorted(by_repo.items()):
        got = list(build_from_chunks(repo_chunks, repo, stats))
        print(f"  {repo:<12} {len(repo_chunks):>5} chunks -> {len(got):>4} examples")
        examples.extend(got)

    print("\n" + stats.report())

    dev, test = split(examples)
    write_jsonl(examples, args.out)
    write_jsonl(dev, args.out.with_name(args.out.stem + "_dev.jsonl"))
    write_jsonl(test, args.out.with_name(args.out.stem + "_test.jsonl"))
    print(f"\nwrote {len(examples):,} examples -> {args.out}")
    print(f"  dev  {len(dev):,}   test {len(test):,}")

    langs: dict[str, int] = {}
    words: list[int] = []
    for e in examples:
        langs[e.language] = langs.get(e.language, 0) + 1
        words.append(e.meta["query_words"])
    print(f"  by language: {dict(sorted(langs.items(), key=lambda kv: -kv[1]))}")
    if words:
        words.sort()
        print(f"  query length: p50={words[len(words)//2]} "
              f"p95={words[int(len(words)*0.95)]} max={words[-1]} words")

    print("\nsample queries:")
    for e in examples[:6]:
        print(f'  [{e.language:<10}] "{e.query[:82]}"')
        print(f'               -> {e.gold_symbol_path[:82]}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
