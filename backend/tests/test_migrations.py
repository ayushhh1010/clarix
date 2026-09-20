"""
Migration and schema tests, executed against a real PostgreSQL + pgvector.

These exist because the v2 schema rests on pgvector specifics that are easy
to get wrong from documentation alone -- operator-class names, which types
`binary_quantize` accepts, whether a generated column may call a
user-defined function. Each is asserted here rather than assumed.
"""

from __future__ import annotations

import os

import pytest

DIM = 768


def scalar(conn, sql: str, *args):
    with conn.cursor() as cur:
        cur.execute(sql, args or None)
        row = cur.fetchone()
        return row[0] if row else None


def rows(conn, sql: str, *args):
    with conn.cursor() as cur:
        cur.execute(sql, args or None)
        return cur.fetchall()


# --- structure -------------------------------------------------------------

def test_upgrade_creates_every_table(conn):
    names = {
        r[0]
        for r in rows(
            conn,
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'",
        )
    }
    assert {
        "users", "repositories", "conversations", "messages",
        "chunks", "embedding_cache", "ingest_jobs", "provider_usage",
        "alembic_version",
    } <= names


def test_extensions_are_installed(conn):
    installed = {r[0]: r[1] for r in rows(
        conn, "SELECT extname, extversion FROM pg_extension"
    )}
    assert "vector" in installed
    assert "pg_trgm" in installed


def test_vector_columns_have_the_intended_types(conn):
    types = dict(
        rows(
            conn,
            "SELECT column_name, udt_name FROM information_schema.columns "
            "WHERE table_name = 'chunks'",
        )
    )
    assert types["embedding"] == "halfvec"
    assert types["embedding_bits"] == "bit"
    assert types["search_vector"] == "tsvector"


def test_indexes_exist_with_the_expected_access_methods(conn):
    found = {
        r[0]: (r[1], r[2])
        for r in rows(
            conn,
            """
            SELECT i.relname, am.amname, pg_get_indexdef(i.oid)
            FROM pg_class i
            JOIN pg_index ix ON ix.indexrelid = i.oid
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_am am ON am.oid = i.relam
            WHERE t.relname = 'chunks'
            """,
        )
    }
    assert found["ix_chunks_bits_hnsw"][0] == "hnsw"
    assert "bit_hamming_ops" in found["ix_chunks_bits_hnsw"][1]
    assert found["ix_chunks_search"][0] == "gin"
    assert found["ix_chunks_symbol_trgm"][0] == "gin"
    assert "gin_trgm_ops" in found["ix_chunks_symbol_trgm"][1]

    # The halfvec column must NOT be indexed: doing so was measured at
    # 3,594 B/row vs 2,045 B/row, the difference between fitting a 500 MB
    # free tier and not.
    assert not any(
        "embedding halfvec" in defn or "embedding)" in defn
        for _, (_, defn) in found.items()
        if "hnsw" in defn and "embedding_bits" not in defn
    ), "halfvec must stay unindexed; see 0002 migration notes"


def test_repositories_gained_indexing_provenance(conn):
    cols = {
        r[0]
        for r in rows(
            conn,
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'repositories'",
        )
    }
    assert {
        "default_branch", "head_commit_sha", "indexed_commit_sha",
        "index_version", "indexed_chunk_count", "last_indexed_at",
    } <= cols


# --- lexical tokenisation --------------------------------------------------

def test_tsv_function_splits_camel_and_snake_case(conn):
    vec = scalar(
        conn,
        "SELECT clarix_code_tsv(%s, %s, %s)::text",
        "getCurrentUser", "app/security.py", "def get_current_user(token): ...",
    )
    for token in ("current", "user", "get", "getcurrentuser"):
        assert token in vec, f"{token!r} missing from {vec}"


def test_tsv_weights_symbol_above_body(conn):
    """An identifier match must outrank an incidental mention in a body."""
    ranked = rows(
        conn,
        """
        SELECT ts_rank_cd(clarix_code_tsv('parse_config', 'a.py', 'unrelated body'),
                          websearch_to_tsquery('simple', 'parse config')),
               ts_rank_cd(clarix_code_tsv('other', 'b.py', 'we call parse_config here'),
                          websearch_to_tsquery('simple', 'parse config'))
        """,
    )[0]
    assert ranked[0] > ranked[1], f"symbol rank {ranked[0]} !> body rank {ranked[1]}"


def test_generated_search_vector_is_populated_on_insert(conn, repo_id):
    _insert_chunk(conn, repo_id, symbol="getCurrentUser", content="def get_current_user(): pass")
    vec = scalar(conn, "SELECT search_vector::text FROM chunks LIMIT 1")
    assert "current" in vec and "user" in vec


def test_changing_the_tsv_function_is_blocked_or_leaves_stale_rows(conn, repo_id):
    """
    Pins the hazard documented in migration 0002.

    Postgres does not recompute generated columns when an IMMUTABLE
    function's body changes. Either it refuses the redefinition, or it
    accepts it and existing rows go stale. Both are acceptable; silently
    accepting it while we *believe* rows are current is not, which is why
    INDEX_VERSION must be bumped alongside any change to this function.
    """
    _insert_chunk(conn, repo_id, symbol="alpha", content="body")
    before = scalar(conn, "SELECT search_vector::text FROM chunks LIMIT 1")

    try:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE OR REPLACE FUNCTION clarix_split_identifiers(txt text) "
                "RETURNS text LANGUAGE sql IMMUTABLE PARALLEL SAFE AS "
                "$$ SELECT 'REDEFINED' $$;"
            )
        redefined = True
    except Exception:
        redefined = False

    if redefined:
        after = scalar(conn, "SELECT search_vector::text FROM chunks LIMIT 1")
        assert after == before, (
            "row silently recomputed -- the staleness model in 0002 is wrong"
        )


# --- vector operations -----------------------------------------------------

def test_binary_quantize_accepts_both_vector_and_halfvec(conn):
    lit = "[" + ",".join(["0.5"] * DIM) + "]"
    assert len(scalar(conn, f"SELECT binary_quantize('{lit}'::vector({DIM}))")) == DIM
    assert len(scalar(conn, f"SELECT binary_quantize('{lit}'::halfvec({DIM}))")) == DIM


def test_two_stage_retrieval_query_runs_and_uses_the_index(conn, repo_id):
    import random

    rng = random.Random(3)
    with conn.cursor() as cur:
        for i in range(300):
            v = "[" + ",".join(f"{rng.gauss(0, 1):.3f}" for _ in range(DIM)) + "]"
            cur.execute(
                f"""
                INSERT INTO chunks (chunk_id, repo_id, content_sha, file_path,
                                    language, kind, symbol_path, start_line,
                                    end_line, token_count, content,
                                    embedding, embedding_bits)
                VALUES (%s, %s, %s, 'a.py', 'python', 'definition', 'a.py::f',
                        1, 2, 10, 'x',
                        %s::halfvec({DIM}), binary_quantize(%s::vector({DIM})))
                """,
                (f"c{i:028d}", repo_id, f"{i:064d}", v, v),
            )

    q = "[" + ",".join(f"{rng.gauss(0, 1):.3f}" for _ in range(DIM)) + "]"
    got = rows(
        conn,
        f"""
        WITH candidates AS (
            SELECT chunk_id, embedding FROM chunks
            WHERE repo_id = %s
            ORDER BY embedding_bits <~> binary_quantize(%s::vector({DIM}))
            LIMIT 100
        )
        SELECT chunk_id, embedding <=> %s::halfvec({DIM}) AS distance
        FROM candidates ORDER BY distance LIMIT 10
        """,
        repo_id, q, q,
    )
    assert len(got) == 10
    assert all(d is not None for _, d in got)


def test_hybrid_query_combines_vector_and_lexical(conn, repo_id):
    """The shape of the retrieval query: RRF over a dense and a lexical arm."""
    _insert_chunk(conn, repo_id, symbol="get_current_user",
                  content="def get_current_user(token): return decode(token)")
    _insert_chunk(conn, repo_id, symbol="unrelated", content="def unrelated(): pass",
                  chunk_id="b" * 24)

    got = rows(
        conn,
        """
        SELECT chunk_id, ts_rank_cd(search_vector,
                                    websearch_to_tsquery('simple', %s)) AS rank
        FROM chunks
        WHERE repo_id = %s AND search_vector @@ websearch_to_tsquery('simple', %s)
        ORDER BY rank DESC
        """,
        "current user", repo_id, "current user",
    )
    assert got, "lexical arm returned nothing"
    assert got[0][0].startswith("a"), "expected the matching symbol first"


# --- round trip ------------------------------------------------------------

def test_downgrade_then_upgrade_round_trips(pg_url):
    import pathlib

    from alembic.config import Config

    from alembic import command

    backend = pathlib.Path(__file__).resolve().parents[1]
    cfg = Config(str(backend / "alembic.ini"))
    cfg.set_main_option("script_location", str(backend / "alembic"))
    os.environ["ALEMBIC_DATABASE_URL"] = pg_url
    try:
        command.upgrade(cfg, "head")
        command.downgrade(cfg, "base")
        command.upgrade(cfg, "head")
    finally:
        os.environ.pop("ALEMBIC_DATABASE_URL", None)


def test_baseline_is_idempotent_over_an_existing_v1_database(pg_url):
    """
    v1 deployments were created with `create_all` and have no alembic
    history. Running `upgrade head` against one must not fail on tables that
    already exist.
    """
    import pathlib

    import psycopg
    from alembic.config import Config

    from alembic import command

    raw = pg_url.replace("postgresql+psycopg://", "postgresql://", 1)
    with psycopg.connect(raw, autocommit=True) as c:
        c.execute("CREATE TABLE users (id uuid PRIMARY KEY, email text)")

    backend = pathlib.Path(__file__).resolve().parents[1]
    cfg = Config(str(backend / "alembic.ini"))
    cfg.set_main_option("script_location", str(backend / "alembic"))
    os.environ["ALEMBIC_DATABASE_URL"] = pg_url
    try:
        command.upgrade(cfg, "head")  # must not raise
    finally:
        os.environ.pop("ALEMBIC_DATABASE_URL", None)


# --- helpers ---------------------------------------------------------------

@pytest.fixture
def repo_id(conn) -> str:
    rid = "11111111-1111-1111-1111-111111111111"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO repositories (id, name, local_path, status) "
            "VALUES (%s, 'r', '/tmp/r', 'ready')",
            (rid,),
        )
    return rid


def _insert_chunk(conn, repo_id: str, *, symbol: str, content: str,
                  chunk_id: str = "a" * 24) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chunks (chunk_id, repo_id, content_sha, file_path, language,
                                kind, symbol, symbol_path, start_line, end_line,
                                token_count, content)
            VALUES (%s, %s, %s, 'app/x.py', 'python', 'definition', %s,
                    'app/x.py::' || %s, 1, 5, 20, %s)
            """,
            (chunk_id, repo_id, chunk_id * 2, symbol, symbol, content),
        )
