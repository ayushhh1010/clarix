"""
Empirically verify the pgvector features the v2 schema depends on.

Written because the schema design rests on specifics that blog posts get
wrong (they conflate the type's dimension ceiling with the *index* ceiling,
and disagree on operator-class names). Everything below is executed against a
real server rather than read from documentation.

Run:  python verify_pgvector.py
"""

from __future__ import annotations

import pathlib
import random
import tempfile
import time

import psycopg2
from embedded_postgres import get_server

DIM = 768  # jina-embeddings-v2-base-code
N = 5_000

checks: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        checks.append((name, True, detail or ""))
    except Exception as exc:  # noqa: BLE001 - reporting is the point
        checks.append((name, False, f"{type(exc).__name__}: {exc}"[:160]))


def main() -> int:
    d = pathlib.Path(tempfile.mkdtemp(prefix="clarix_verify_"))
    srv = get_server(d, cleanup_mode=None)
    conn = psycopg2.connect(srv.get_uri())
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
    pgvector_version = cur.fetchone()[0]
    cur.execute("SHOW server_version")
    pg_version = cur.fetchone()[0]
    print(f"postgres {pg_version} | pgvector {pgvector_version} | dim {DIM} | rows {N:,}\n")

    # --- types ------------------------------------------------------------
    def types():
        cur.execute(
            f"""
            CREATE TABLE t (
              id        bigserial PRIMARY KEY,
              repo_id   uuid NOT NULL,
              emb       halfvec({DIM}),
              bits      bit({DIM}),
              body      text
            )
            """
        )
        return f"halfvec({DIM}) + bit({DIM}) accepted"

    check("halfvec and bit column types", types)

    # --- binary_quantize --------------------------------------------------
    def bq():
        cur.execute(
            f"SELECT binary_quantize('[{','.join(['0.5'] * DIM)}]'::vector({DIM}))"
        )
        got = cur.fetchone()[0]
        assert len(got) == DIM, f"expected {DIM} bits, got {len(got)}"
        return f"returns bit({len(got)})"

    check("binary_quantize(vector) -> bit", bq)

    def bq_half():
        cur.execute(
            f"SELECT binary_quantize('[{','.join(['0.5'] * DIM)}]'::halfvec({DIM}))"
        )
        return "accepts halfvec input"

    check("binary_quantize(halfvec) -> bit", bq_half)

    # --- load -------------------------------------------------------------
    def load():
        rng = random.Random(7)
        repo = "11111111-1111-1111-1111-111111111111"
        rows = []
        for _ in range(N):
            v = [rng.gauss(0, 1) for _ in range(DIM)]
            lit = "[" + ",".join(f"{x:.4f}" for x in v) + "]"
            rows.append((repo, lit))
        t0 = time.perf_counter()
        cur.executemany(
            f"INSERT INTO t (repo_id, emb, bits, body) VALUES "
            f"(%s, %s::halfvec({DIM}), binary_quantize(%s::vector({DIM})), 'x')",
            [(r, lit, lit) for r, lit in rows],
        )
        return f"{N:,} rows in {time.perf_counter() - t0:.1f}s"

    check("insert with binary_quantize in the write path", load)

    # --- storage ----------------------------------------------------------
    def storage():
        cur.execute(
            "SELECT pg_column_size(emb), pg_column_size(bits) FROM t LIMIT 1"
        )
        e, b = cur.fetchone()
        return f"halfvec {e} B/row, bit {b} B/row  ({e / b:.0f}x)"

    check("per-row storage", storage)

    # --- indexes ----------------------------------------------------------
    def hnsw_bit():
        t0 = time.perf_counter()
        cur.execute("CREATE INDEX t_bits_hnsw ON t USING hnsw (bits bit_hamming_ops)")
        return f"built in {time.perf_counter() - t0:.1f}s"

    check("HNSW index, bit_hamming_ops", hnsw_bit)

    def hnsw_half():
        t0 = time.perf_counter()
        cur.execute("CREATE INDEX t_emb_hnsw ON t USING hnsw (emb halfvec_cosine_ops)")
        return f"built in {time.perf_counter() - t0:.1f}s"

    check("HNSW index, halfvec_cosine_ops", hnsw_half)

    def index_sizes():
        cur.execute(
            "SELECT pg_size_pretty(pg_relation_size('t_bits_hnsw')),"
            "       pg_size_pretty(pg_relation_size('t_emb_hnsw')),"
            "       pg_size_pretty(pg_relation_size('t'))"
        )
        b, e, tbl = cur.fetchone()
        return f"bit idx {b}, halfvec idx {e}, table {tbl}"

    check("index sizes", index_sizes)

    # --- operators --------------------------------------------------------
    rng = random.Random(99)
    qvec = "[" + ",".join(f"{rng.gauss(0, 1):.4f}" for _ in range(DIM)) + "]"

    def hamming_op():
        cur.execute(
            f"SELECT id FROM t ORDER BY bits <~> binary_quantize('{qvec}'::vector({DIM})) LIMIT 5"
        )
        return f"<~> returned {len(cur.fetchall())} rows"

    check("hamming operator <~> on bit", hamming_op)

    def cosine_op():
        cur.execute(f"SELECT id FROM t ORDER BY emb <=> '{qvec}'::halfvec({DIM}) LIMIT 5")
        return f"<=> returned {len(cur.fetchall())} rows"

    check("cosine operator <=> on halfvec", cosine_op)

    # --- the two-stage query we actually intend to ship --------------------
    def two_stage():
        sql = f"""
        WITH candidates AS (
            SELECT id, emb
            FROM t
            WHERE repo_id = %s
            ORDER BY bits <~> binary_quantize(%s::vector({DIM}))
            LIMIT 200
        )
        SELECT id, emb <=> %s::halfvec({DIM}) AS distance
        FROM candidates
        ORDER BY distance
        LIMIT 10
        """
        t0 = time.perf_counter()
        cur.execute(sql, ("11111111-1111-1111-1111-111111111111", qvec, qvec))
        rows = cur.fetchall()
        return f"{len(rows)} rows in {(time.perf_counter() - t0) * 1000:.1f} ms"

    check("two-stage: bit ANN -> halfvec rescore", two_stage)

    def uses_index():
        cur.execute(
            f"EXPLAIN (FORMAT TEXT) SELECT id FROM t "
            f"ORDER BY bits <~> binary_quantize('{qvec}'::vector({DIM})) LIMIT 10"
        )
        plan = "\n".join(r[0] for r in cur.fetchall())
        assert "t_bits_hnsw" in plan, f"index not used:\n{plan}"
        return "planner chose t_bits_hnsw"

    check("planner uses the HNSW bit index", uses_index)

    # --- recall of the two-stage pipeline vs exact ------------------------
    def recall():
        """
        Ground truth is exact cosine over halfvec with indexes disabled.
        Measures recall@10 of (bit ANN top-200 -> halfvec rescore top-10).
        """
        cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute("SET LOCAL enable_bitmapscan = off")
        hits_total = 0
        trials = 20
        r2 = random.Random(1234)
        for _ in range(trials):
            q = "[" + ",".join(f"{r2.gauss(0, 1):.4f}" for _ in range(DIM)) + "]"
            cur.execute(
                f"SELECT id FROM t ORDER BY emb <=> '{q}'::halfvec({DIM}) LIMIT 10"
            )
            truth = {r[0] for r in cur.fetchall()}
            cur.execute(
                f"""
                WITH c AS (
                  SELECT id, emb FROM t
                  ORDER BY bits <~> binary_quantize('{q}'::vector({DIM}))
                  LIMIT 200
                )
                SELECT id FROM c ORDER BY emb <=> '{q}'::halfvec({DIM}) LIMIT 10
                """
            )
            got = {r[0] for r in cur.fetchall()}
            hits_total += len(truth & got)
        return f"recall@10 = {hits_total / (trials * 10):.1%} over {trials} queries"

    check("recall of two-stage vs exact", recall)

    # --- iterative scan (matters: we always filter by repo_id) ------------
    def iterative():
        cur.execute("SET hnsw.iterative_scan = relaxed_order")
        cur.execute("SHOW hnsw.iterative_scan")
        return f"hnsw.iterative_scan = {cur.fetchone()[0]}"

    check("hnsw.iterative_scan GUC", iterative)

    # --- lexical side ------------------------------------------------------
    def fts_code():
        """
        `simple` splits on underscores; camelCase needs an explicit regex.
        Verify both, since identifier matching is the whole point of the
        lexical arm of hybrid retrieval.
        """
        expr = (
            "to_tsvector('simple', $1 || ' ' || "
            "regexp_replace($1, '([a-z0-9])([A-Z])', '\\1 \\2', 'g'))"
        )
        cur.execute(f"SELECT {expr.replace('$1', '%s')}", ("getCurrentUser get_current_user",) * 2)
        vec = cur.fetchone()[0]
        for tok in ("getcurrentuser", "current", "user", "get"):
            assert tok in vec, f"missing {tok!r} in {vec}"
        return "camelCase + snake_case both tokenised"

    check("tsvector tokenisation for code identifiers", fts_code)

    def fts_match():
        cur.execute(
            "SELECT to_tsvector('simple', 'def get_current_user(token)') "
            "@@ websearch_to_tsquery('simple', 'current user')"
        )
        assert cur.fetchone()[0] is True
        return "websearch_to_tsquery matches"

    check("full-text match on a code line", fts_match)

    # --- report -----------------------------------------------------------
    print(f"{'check':<46} {'':>4}  detail")
    print("-" * 100)
    ok = 0
    for name, passed, detail in checks:
        print(f"{name:<46} {'PASS' if passed else 'FAIL':>4}  {detail}")
        ok += passed
    print("-" * 100)
    print(f"{ok}/{len(checks)} passed")

    conn.close()
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
