"""v2 retrieval schema: pgvector chunks, embedding cache, job queue, usage counters.

Verified against PostgreSQL 18.6 / pgvector 0.8.6 by bench/verify_pgvector.py
before this was written. Specifics that documentation and blog posts disagree
on -- operator-class names, whether `binary_quantize` accepts halfvec, the
distinction between the type's dimension ceiling and the *index* ceiling --
were each executed against a real server rather than assumed.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels = None
depends_on = None

EMBED_DIM = 768


# `halfvec` and `binary_quantize` were both added in pgvector 0.7.0
# (2024-04-29); `bit` indexing arrived in the same release. Everything this
# migration creates depends on them, and nothing here needs 0.8 -- iterative
# index scans are not used.
MIN_PGVECTOR = (0, 7, 0)


def _check_pgvector_version() -> None:
    """
    Fail early, and legibly, on a pgvector too old for this schema.

    Without this the migration gets as far as `CREATE TABLE ... halfvec(768)`
    and dies on `type "halfvec" does not exist`, which reads like a typo
    rather than a version problem. Managed providers pin their own pgvector
    build, so this is the failure a first deploy is most likely to hit.
    """
    installed = op.get_bind().exec_driver_sql(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    ).scalar()
    if installed is None:  # pragma: no cover - CREATE EXTENSION just ran
        raise RuntimeError("the `vector` extension is not installed")

    numeric = [int(p) for p in installed.split(".")[:3] if p.isdigit()]
    if tuple(numeric) < MIN_PGVECTOR:
        raise RuntimeError(
            f"pgvector {installed} is too old: this schema needs "
            f"{'.'.join(map(str, MIN_PGVECTOR))} or newer for halfvec and "
            f"binary_quantize. Upgrade the extension, or the server if it "
            f"does not offer a newer build."
        )


def upgrade() -> None:
    # --- extensions -------------------------------------------------------
    # Supabase, Neon and the embedded test server all ship these; CREATE
    # EXTENSION is idempotent and safe to re-run.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    _check_pgvector_version()

    # --- lexical tokenisation --------------------------------------------
    #
    # Postgres' `simple` configuration splits on underscores, so
    # `get_current_user` already yields get/current/user. It does NOT split
    # camelCase, so `getCurrentUser` stays a single token and a search for
    # "current user" misses it. The helper appends a split copy, keeping both
    # the intact identifier and its parts.
    #
    # OPERATIONAL HAZARD: `search_vector` is a generated column that calls
    # this function. Postgres does not recompute generated columns when an
    # IMMUTABLE function's body changes, so redefining it leaves stale
    # tsvectors behind. Any change here MUST bump models_v2.INDEX_VERSION,
    # which forces affected repositories to be reindexed.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION clarix_split_identifiers(txt text)
        RETURNS text
        LANGUAGE sql
        IMMUTABLE
        PARALLEL SAFE
        AS $$
            SELECT coalesce(txt, '') || ' ' ||
                   regexp_replace(coalesce(txt, ''), '([a-z0-9])([A-Z])', '\\1 \\2', 'g')
        $$;
        """
    )
    # Weighted so an identifier match outranks an incidental body mention:
    # A = symbol, B = file path, C = body.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION clarix_code_tsv(symbol text, file_path text, body text)
        RETURNS tsvector
        LANGUAGE sql
        IMMUTABLE
        PARALLEL SAFE
        AS $$
            SELECT setweight(to_tsvector('simple', clarix_split_identifiers(symbol)), 'A')
                || setweight(to_tsvector('simple', clarix_split_identifiers(file_path)), 'B')
                || setweight(to_tsvector('simple', clarix_split_identifiers(body)), 'C')
        $$;
        """
    )

    # --- repositories: indexing provenance --------------------------------
    # v1 tracked ingestion progress but not *what* was indexed, so there was
    # no way to tell a stale index from a current one, and no basis for
    # incremental reindexing.
    for column in (
        sa.Column("default_branch", sa.String(255), nullable=True),
        sa.Column("head_commit_sha", sa.String(40), nullable=True),
        sa.Column("indexed_commit_sha", sa.String(40), nullable=True),
        sa.Column("index_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("indexed_chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("indexed_token_count", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("last_indexed_at", sa.DateTime(timezone=True), nullable=True),
    ):
        op.add_column("repositories", column)

    # --- chunks -----------------------------------------------------------
    op.create_table(
        "chunks",
        sa.Column("chunk_id", sa.String(32), primary_key=True),
        sa.Column(
            "repo_id",
            UUID(as_uuid=False),
            sa.ForeignKey("repositories.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content_sha", sa.String(64), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("language", sa.String(32), nullable=False),
        sa.Column("kind", sa.String(24), nullable=False),
        sa.Column("node_type", sa.String(64), nullable=True),
        sa.Column("symbol", sa.Text(), nullable=True),
        sa.Column("parent_scope", sa.Text(), nullable=False, server_default=""),
        sa.Column("symbol_path", sa.Text(), nullable=False),
        sa.Column("start_line", sa.Integer(), nullable=False),
        sa.Column("end_line", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("part", sa.SmallInteger(), nullable=False, server_default="1"),
        sa.Column("part_of", sa.SmallInteger(), nullable=False, server_default="1"),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("index_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    # Vector and generated columns are added by raw DDL: SQLAlchemy has no
    # first-class halfvec/bit DDL emitter and the generated column needs an
    # exact expression.
    op.execute(f"ALTER TABLE chunks ADD COLUMN embedding halfvec({EMBED_DIM})")
    op.execute(f"ALTER TABLE chunks ADD COLUMN embedding_bits bit({EMBED_DIM})")
    op.execute(
        """
        ALTER TABLE chunks ADD COLUMN search_vector tsvector
        GENERATED ALWAYS AS (clarix_code_tsv(symbol, file_path, content)) STORED
        """
    )

    # Candidate generation runs on the bit column only. Indexing halfvec as
    # well was measured at 3,594 B/row (719 MB at 200k chunks) versus
    # 2,045 B/row (409 MB) for this layout -- the difference between fitting
    # a 500 MB free tier and not.
    #
    # NOTE for bulk ingest: building HNSW after a large load is materially
    # faster than maintaining it per-insert. The ingestion path should drop
    # and rebuild this index around a full reindex.
    op.execute(
        "CREATE INDEX ix_chunks_bits_hnsw ON chunks "
        "USING hnsw (embedding_bits bit_hamming_ops)"
    )
    op.execute("CREATE INDEX ix_chunks_search ON chunks USING gin (search_vector)")
    op.create_index("ix_chunks_repo_file", "chunks", ["repo_id", "file_path"])
    op.create_index("ix_chunks_content_sha", "chunks", ["content_sha"])
    op.execute(
        "CREATE INDEX ix_chunks_repo_symbol ON chunks (repo_id, symbol) "
        "WHERE symbol IS NOT NULL"
    )
    op.execute(
        "CREATE INDEX ix_chunks_symbol_trgm ON chunks USING gin (symbol gin_trgm_ops) "
        "WHERE symbol IS NOT NULL"
    )

    # --- embedding cache --------------------------------------------------
    op.create_table(
        "embedding_cache",
        sa.Column("model_id", sa.String(128), primary_key=True),
        sa.Column("content_sha", sa.String(64), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "last_used_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.execute(
        f"ALTER TABLE embedding_cache ADD COLUMN embedding halfvec({EMBED_DIM}) NOT NULL"
    )
    op.execute(
        f"ALTER TABLE embedding_cache ADD COLUMN embedding_bits bit({EMBED_DIM}) NOT NULL"
    )
    op.create_index("ix_embcache_last_used", "embedding_cache", ["last_used_at"])

    # --- job queue --------------------------------------------------------
    op.create_table(
        "ingest_jobs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "repo_id",
            UUID(as_uuid=False),
            sa.ForeignKey("repositories.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("payload", JSONB(), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column(
            "run_after",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("locked_by", sa.String(128), nullable=True),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "status IN ('queued','running','done','failed','dead')",
            name="ck_jobs_status",
        ),
    )
    # Partial indexes: these stay small as completed jobs accumulate.
    op.execute(
        "CREATE INDEX ix_jobs_claimable ON ingest_jobs (run_after) WHERE status = 'queued'"
    )
    op.execute(
        "CREATE INDEX ix_jobs_stale_locks ON ingest_jobs (heartbeat_at) "
        "WHERE status = 'running'"
    )
    op.create_index("ix_jobs_repo", "ingest_jobs", ["repo_id"])

    # --- provider usage ---------------------------------------------------
    op.create_table(
        "provider_usage",
        sa.Column("provider", sa.String(32), primary_key=True),
        sa.Column("model", sa.String(128), primary_key=True),
        # `window` is a reserved word in Postgres (WINDOW clause); naming it
        # `window_kind` avoids quoting it in every hand-written predicate.
        sa.Column("window_kind", sa.String(8), primary_key=True),
        sa.Column("window_start", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("requests", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("input_tokens", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("errors", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint("window_kind IN ('minute','day')", name="ck_usage_window"),
    )
    op.create_index("ix_usage_window", "provider_usage", ["window_kind", "window_start"])


def downgrade() -> None:
    op.drop_table("provider_usage")
    op.drop_table("ingest_jobs")
    op.drop_table("embedding_cache")
    op.drop_table("chunks")
    for name in (
        "last_indexed_at",
        "indexed_token_count",
        "indexed_chunk_count",
        "index_version",
        "indexed_commit_sha",
        "head_commit_sha",
        "default_branch",
    ):
        op.drop_column("repositories", name)
    op.execute("DROP FUNCTION IF EXISTS clarix_code_tsv(text, text, text)")
    op.execute("DROP FUNCTION IF EXISTS clarix_split_identifiers(text)")
