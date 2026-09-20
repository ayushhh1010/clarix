"""
Run the retrieval evaluation: per-arm ablation with confidence intervals.

What this does
--------------
1. Chunks the corpus and strips docstrings from every chunk (leakage control
   -- see app/evaluation/dataset.py; without it the lexical arm matches the
   query verbatim and every configuration scores ~1.0).
2. Embeds the stripped chunks, cached on disk by content hash.
3. Loads them into an ephemeral PostgreSQL, migrated with the real Alembic
   migrations -- the same schema production uses, not a simplified stand-in.
4. Runs every configuration over the same queries and scores them.
5. Compares configurations with a paired bootstrap, corrected with Holm.

Every configuration sees identical queries and an identical index, so the
comparisons are paired and the only thing varying is the retrieval
configuration itself.

Usage:
    python run_retrieval_eval.py --split dev            # while tuning
    python run_retrieval_eval.py --split test           # report once
    python run_retrieval_eval.py --split dev --keep-docstrings   # leakage size
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import tempfile
import time

import numpy as np

BENCH_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(BENCH_DIR.parent / "backend"))
sys.path.insert(0, str(BENCH_DIR))

DIM = 768

# Arm combinations under test. "hybrid" is the shipping configuration; the
# single-arm rows are what justify (or fail to justify) each arm's cost.
ALL_ARMS = {"use_dense": True, "use_lexical": True, "use_symbol": True}


def w(dense: float, lexical: float, symbol: float) -> dict:
    return {"weights": {"dense": dense, "lexical": lexical, "symbol": symbol}}


# Single-arm rows establish each arm's standalone quality. The weighted rows
# are the response to the first run's finding: unweighted RRF scored WORSE
# than dense alone (recall@10 0.784 vs 0.908), because equal weights let a
# rank-1 hit from an arm with 0.339 recall displace a rank-1 hit from an arm
# with 0.908.
#
# Weights are swept on dev and reported once on test. Sweeping on the same
# split used to report would fit them to noise, which is what the dev/test
# split exists to prevent.
CONFIGS: dict[str, dict] = {
    "dense only": {"use_dense": True, "use_lexical": False, "use_symbol": False},
    "lexical only": {"use_dense": False, "use_lexical": True, "use_symbol": False},
    "symbol only": {"use_dense": False, "use_lexical": False, "use_symbol": True},
    "hybrid w=1,1,1": ALL_ARMS | w(1.0, 1.0, 1.0),
    "hybrid w=1,.5,.5": ALL_ARMS | w(1.0, 0.5, 0.5),
    "hybrid w=1,.25,.25": ALL_ARMS | w(1.0, 0.25, 0.25),
    "hybrid w=1,.1,.1": ALL_ARMS | w(1.0, 0.1, 0.1),
    "hybrid w=1,.25,.5": ALL_ARMS | w(1.0, 0.25, 0.5),
    # Symbol-heavy: on identifier queries the symbol arm alone beat dense by
    # +0.153 MRR, so the sweep above (which never weights it above 0.5) may
    # not have reached the optimum.
    "hybrid w=1,.25,1": ALL_ARMS | w(1.0, 0.25, 1.0),
    "hybrid w=1,.1,2": ALL_ARMS | w(1.0, 0.1, 2.0),
    "hybrid w=.5,.1,2": ALL_ARMS | w(0.5, 0.1, 2.0),
    # Passing no arm flags lets hybrid_search route per query. This row is
    # the actual shipping configuration; it should match dense-only on the
    # docstring set and the symbol-heavy hybrid on the identifier set. If it
    # does not match both, the router is mis-classifying.
    "ROUTED (shipping)": {},
}
BASELINE = "dense only"  # what v1 did


def embed_cached(texts: list[str], cache_path: pathlib.Path, batch: int) -> np.ndarray:
    """Embed, reusing any cached vectors keyed by content hash."""
    keys = [hashlib.sha256(t.encode("utf-8", "replace")).hexdigest() for t in texts]

    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        stored_keys = data["keys"]
        stored_vecs = data["vecs"]
        cache = {k: stored_vecs[i] for i, k in enumerate(stored_keys)}
        print(f"  cache: {len(cache):,} vectors on disk", flush=True)

    missing_idx = [i for i, k in enumerate(keys) if k not in cache]
    if missing_idx:
        from app.indexing.embedder import OnnxEmbedder

        print(f"  embedding {len(missing_idx):,} new texts ...", flush=True)
        emb = OnnxEmbedder()
        t0 = time.perf_counter()
        fresh = emb.embed([texts[i] for i in missing_idx], batch_size=batch)
        dt = time.perf_counter() - t0
        print(f"  {dt:.1f}s ({len(missing_idx) / dt:.2f}/s)", flush=True)
        for slot, i in enumerate(missing_idx):
            cache[keys[i]] = fresh[slot]

        all_keys = np.array(list(cache.keys()))
        all_vecs = np.stack([cache[k] for k in all_keys])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, keys=all_keys, vecs=all_vecs)

    return np.stack([cache[k] for k in keys])


def bits_literal(vec: np.ndarray) -> str:
    """
    Delegates to the shipped quantiser rather than reimplementing it.

    These two helpers previously had their own implementations. They were
    verified equivalent to production over 200 vectors including exact
    zeros -- but equivalence today is not a property, it is a coincidence
    that holds until someone changes the threshold or the packing. An
    evaluation harness that formats vectors differently from the serving
    path silently measures a system nobody ships, and every number in
    BENCHMARKS section 7 comes out of this file.
    """
    from app.indexing.embedder import EMBED_DIM, binary_quantize, bits_to_sql

    return bits_to_sql(binary_quantize(vec.reshape(1, -1))[0], EMBED_DIM)


def vec_literal(vec: np.ndarray) -> str:
    """The same halfvec literal the ingestion path writes."""
    from app.indexing.embedder import vector_to_sql

    return vector_to_sql(vec)


def start_postgres():
    from embedded_postgres import get_server

    datadir = pathlib.Path(tempfile.mkdtemp(prefix="clarix_eval_pg_"))
    return get_server(datadir, cleanup_mode=None)


def migrate(url: str) -> None:
    import os

    from alembic import command
    from alembic.config import Config

    backend = BENCH_DIR.parent / "backend"
    cfg = Config(str(backend / "alembic.ini"))
    cfg.set_main_option("script_location", str(backend / "alembic"))
    os.environ["ALEMBIC_DATABASE_URL"] = url.replace(
        "postgresql://", "postgresql+psycopg://", 1
    )
    command.upgrade(cfg, "head")


def load_index(raw_url: str, repo_id: str, chunks: list, vectors: np.ndarray,
               contents: list[str]) -> None:
    """Insert chunks with the (possibly stripped) content and its embedding."""
    import psycopg

    with psycopg.connect(raw_url, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO repositories (id, name, local_path, status) "
            "VALUES (%s, 'eval', '/tmp/eval', 'ready')",
            (repo_id,),
        )
        # Build the HNSW index after the load, not during: maintaining it
        # per-insert is materially slower.
        cur.execute("DROP INDEX IF EXISTS ix_chunks_bits_hnsw")

        for chunk, vec, content in zip(chunks, vectors, contents, strict=True):
            cur.execute(
                f"""
                INSERT INTO chunks (chunk_id, repo_id, content_sha, file_path,
                                    language, kind, node_type, symbol,
                                    parent_scope, symbol_path, start_line,
                                    end_line, token_count, content,
                                    embedding, embedding_bits)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        %s::halfvec({DIM}), %s::bit({DIM}))
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                (
                    chunk.chunk_id, repo_id, chunk.content_sha, chunk.file_path,
                    chunk.language, chunk.kind, chunk.node_type, chunk.symbol,
                    ".".join(chunk.parent_scope), chunk.symbol_path,
                    chunk.start_line, chunk.end_line, chunk.token_count,
                    content, vec_literal(vec), bits_literal(vec),
                ),
            )
        cur.execute(
            "CREATE INDEX ix_chunks_bits_hnsw ON chunks "
            "USING hnsw (embedding_bits bit_hamming_ops)"
        )
        cur.execute("ANALYZE chunks")
        cur.execute("SELECT count(*) FROM chunks")
        print(f"  indexed {cur.fetchone()[0]:,} chunks", flush=True)


async def run_config(session, repo_id: str, examples, query_vecs, name: str,
                     flags: dict, limit: int):
    from app.evaluation.metrics import RunResult, standard_metrics
    from app.retrieval.hybrid import hybrid_search

    result = RunResult(system=name)
    arm_yield: dict[str, int] = {}

    for ex, qvec in zip(examples, query_vecs, strict=True):
        t0 = time.perf_counter()
        hits, trace = await hybrid_search(
            session, repo_id, ex.query,
            query_bits=bits_literal(qvec), query_vector=vec_literal(qvec),
            limit=limit, **flags,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        retrieved = [h.chunk_id for h in hits]
        result.record(ex.query_id, standard_metrics(retrieved, set(ex.gold_chunk_ids)), elapsed)
        for arm, n in trace.arm_counts.items():
            arm_yield[arm] = arm_yield.get(arm, 0) + max(n, 0)
        # Mandatory: hybrid_search opens a SAVEPOINT per arm, and holding
        # hundreds of subtransactions open in one transaction drives
        # Postgres off the subxid cache. Without this the run degrades 3.5x
        # over 150 queries and the latency numbers become meaningless.
        await session.rollback()

    result.extra["arm_yield"] = arm_yield
    return result


async def main_async(args) -> int:
    from app.evaluation.dataset import read_jsonl, strip_docstring
    from app.evaluation.metrics import holm_bonferroni, paired_bootstrap
    from bench_quantization import collect_chunks
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    suffix = "" if args.split == "all" else f"_{args.split}"
    eval_path = (BENCH_DIR.parent / "backend" / "eval_data"
                 / f"{args.dataset}{suffix}.jsonl")
    examples = read_jsonl(eval_path)
    print(f"eval set: {len(examples):,} queries from {eval_path.name}")

    chunks = collect_chunks(args.max_chunks or 100_000)
    if args.max_chunks:
        # Subset mode, for validating the pipeline quickly. Queries whose
        # gold chunk fell outside the subset are dropped rather than scored
        # as misses -- counting an unreachable gold as a miss would make the
        # smoke run's numbers meaningless.
        available = {c.chunk_id for c in chunks}
        before = len(examples)
        examples = [e for e in examples
                    if any(g in available for g in e.gold_chunk_ids)]
        print(f"subset mode: {len(examples)}/{before} queries have reachable gold")
    if args.max_queries:
        examples = examples[: args.max_queries]
    print(f"corpus: {len(chunks):,} chunks | {len(examples):,} queries")
    if not examples:
        print("no queries with reachable gold; widen --max-chunks")
        return 1

    if args.keep_docstrings:
        contents = [c.content for c in chunks]
        print("  docstrings KEPT (leakage measurement run)")
    else:
        contents = [strip_docstring(c.content, c.language) for c in chunks]
        changed = sum(1 for c, s in zip(chunks, contents, strict=True) if c.content != s)
        print(f"  docstrings stripped from {changed:,} chunks")

    tag = "keep" if args.keep_docstrings else "strip"
    print("embedding chunks:")
    chunk_vecs = embed_cached(
        contents, BENCH_DIR / "results" / f"eval_chunks_{tag}.npz", args.batch
    )
    print("embedding queries:")
    query_vecs = embed_cached(
        [e.query for e in examples], BENCH_DIR / "results" / "eval_queries.npz", args.batch
    )

    gold = {g for e in examples for g in e.gold_chunk_ids}
    present = {c.chunk_id for c in chunks}
    missing = gold - present
    if missing:
        print(f"  WARNING: {len(missing)} gold chunks absent from the index")

    server = start_postgres()
    url = server.get_uri()
    migrate(url)
    repo_id = "e7a10000-0000-4000-8000-000000000001"
    print("loading index:")
    load_index(url, repo_id, chunks, chunk_vecs, contents)

    engine = create_async_engine(
        url.replace("postgresql://", "postgresql+asyncpg://", 1)
    )
    maker = async_sessionmaker(engine, expire_on_commit=False)

    results = {}
    async with maker() as session:
        for name, flags in CONFIGS.items():
            t0 = time.perf_counter()
            results[name] = await run_config(
                session, repo_id, examples, query_vecs, name, flags, args.limit
            )
            print(f"  {name:<18} {time.perf_counter() - t0:6.1f}s", flush=True)
    await engine.dispose()

    # --- report ---------------------------------------------------------
    headline = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]
    print(f"\n{'configuration':<18}" + "".join(f"{m:>22}" for m in headline)
          + f"{'p50 ms':>9}")
    print("-" * (18 + 22 * len(headline) + 9))
    for name, res in results.items():
        summary = res.summarise()
        row = f"{name:<20}"
        for m in headline:
            est = summary[m]
            row += f"{est.value:>10.3f} [{est.ci_low:.2f},{est.ci_high:.2f}]"
        row += f"{summary['latency_p50_ms'].value:>9.1f}"
        row += f"{summary['latency_p95_ms'].value:>9.1f}"
        print(row)

    print(f"\npaired comparisons vs '{BASELINE}' (Holm-corrected, alpha=0.05)")
    comparisons = []
    base = results[BASELINE]
    for name, res in results.items():
        if name == BASELINE:
            continue
        for metric in ("recall@10", "mrr"):
            comparisons.append(paired_bootstrap(
                base.per_query[metric], res.per_query[metric],
                metric=metric, baseline_name=BASELINE, candidate_name=name,
            ))
    holm_bonferroni(comparisons)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "dataset": args.dataset,
            "split": args.split,
            "queries": len(examples),
            "corpus_chunks": len(chunks),
            "docstrings_stripped": not args.keep_docstrings,
            "configs": {
                n: {k: v.to_dict() for k, v in r.summarise().items()}
                | {"arm_yield": r.extra.get("arm_yield", {})}
                for n, r in results.items()
            },
            "comparisons": [c.to_dict() for c in comparisons],
        }, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


def main() -> int:
    import asyncio

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--dataset", default="retrieval_v1",
                    help="retrieval_v1 (docstring) or symbol_v1 (identifier lookup)")
    ap.add_argument("--limit", type=int, default=20, help="results per query")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="subset the corpus (pipeline validation only)")
    ap.add_argument("--keep-docstrings", action="store_true")
    ap.add_argument("--json-out", type=pathlib.Path)
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
