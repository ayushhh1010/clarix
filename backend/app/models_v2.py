"""
v2 schema: chunks, embedding cache, and the Postgres-backed job queue.

Kept separate from `models.py` during migration; the v1 tables (users,
repositories, conversations, messages) keep their definitions there and are
extended by migration rather than redefined.

Three decisions here are load-bearing and each is backed by a measurement:

1.  Vectors live in Postgres, not a separate store. The v1 split between
    ChromaDB (on an ephemeral disk) and Postgres could not be made
    transactional, so a restart left `status='ready'` pointing at an index
    that no longer existed -- and retrieval silently returned zero chunks
    rather than erroring. Here the vector write and the status flip commit
    together or not at all.

2.  `embedding` is `halfvec(768)`, `embedding_bits` is `bit(768)`, and only
    the bit column is indexed. Measured per-row cost (pgvector 0.8.6):
    halfvec value 1,544 B, its HNSW index ~2,050 B, bit value 101 B, bit
    HNSW index ~400 B. Indexing halfvec too would cost 3,594 B/row --
    719 MB at 200k chunks, over Supabase's 500 MB free tier. Indexing only
    the bits costs 2,045 B/row (409 MB), and the halfvec column is read
    exactly to rescore a small candidate set. See bench/bench_quantization.py
    for the recall this buys.

3.  The job queue is a table, not Redis. See the audit in the architecture
    notes: v1's Redis held only a facts set that nothing ever wrote to.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pgvector.sqlalchemy import BIT, HALFVEC
from sqlalchemy import (
    BigInteger,
    Column,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID

from app.database import Base

EMBED_DIM = 768
EMBED_MODEL_ID = "jinaai/jina-embeddings-v2-base-code"

# Bumped whenever chunking, the embedding model, or the vector layout
# changes. A repo whose `index_version` is behind is reindexed rather than
# queried, so a model swap can never silently mix vector spaces.
# 2: the embedding variant changed from fp16 to int8 at a 512-token cap
# so the indexer fits a 512 MiB instance. Different vectors, so every
# index built under version 1 is stale and gets rebuilt.
INDEX_VERSION = 2


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class Chunk(Base):
    """One indexed unit of a repository."""

    __tablename__ = "chunks"

    # Content-addressed and stable across line drift: sha256(repo, symbol
    # path). Incremental reindexing diffs on this, so a file shifting by 200
    # lines must not churn it.
    chunk_id = Column(String(32), primary_key=True)

    repo_id = Column(
        UUID(as_uuid=False),
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
    )

    # sha256(content). The embedding cache key -- unchanged text is never
    # re-embedded, which matters when the embedding budget is a free tier.
    content_sha = Column(String(64), nullable=False)

    file_path = Column(Text, nullable=False)
    language = Column(String(32), nullable=False)
    kind = Column(String(24), nullable=False)  # definition|container_header|window|prose
    node_type = Column(String(64), nullable=True)
    symbol = Column(Text, nullable=True)
    parent_scope = Column(Text, nullable=False, server_default="")
    symbol_path = Column(Text, nullable=False)

    start_line = Column(Integer, nullable=False)
    end_line = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=False)
    part = Column(SmallInteger, nullable=False, server_default="1")
    part_of = Column(SmallInteger, nullable=False, server_default="1")

    content = Column(Text, nullable=False)

    # Rescoring vector. Deliberately unindexed -- see module docstring.
    embedding = Column(HALFVEC(EMBED_DIM), nullable=True)
    # Candidate-generation vector. This is the one with an HNSW index.
    embedding_bits = Column(BIT(EMBED_DIM), nullable=True)

    # Lexical arm of hybrid retrieval. `clarix_code_tsv` is an IMMUTABLE
    # function defined in the migration: it splits camelCase (Postgres will
    # not), relies on the `simple` config to split snake_case, and weights
    # symbol above path above body.
    search_vector = Column(
        TSVECTOR,
        Computed(
            "clarix_code_tsv(symbol, file_path, content)",
            persisted=True,
        ),
        nullable=True,
    )

    index_version = Column(Integer, nullable=False, server_default=str(INDEX_VERSION))
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    __table_args__ = (
        # Candidate generation. `bit_hamming_ops` and the `<~>` operator were
        # verified against pgvector 0.8.6 in bench/verify_pgvector.py.
        Index(
            "ix_chunks_bits_hnsw",
            "embedding_bits",
            postgresql_using="hnsw",
            postgresql_ops={"embedding_bits": "bit_hamming_ops"},
        ),
        Index("ix_chunks_search", "search_vector", postgresql_using="gin"),
        Index("ix_chunks_repo_file", "repo_id", "file_path"),
        # Embedding-cache lookups and cross-repo dedup.
        Index("ix_chunks_content_sha", "content_sha"),
        # Exact symbol jump ("where is get_current_user defined").
        Index(
            "ix_chunks_repo_symbol",
            "repo_id",
            "symbol",
            postgresql_where=text("symbol IS NOT NULL"),
        ),
        # Fuzzy identifier match for typos and partial names.
        Index(
            "ix_chunks_symbol_trgm",
            "symbol",
            postgresql_using="gin",
            postgresql_ops={"symbol": "gin_trgm_ops"},
            postgresql_where=text("symbol IS NOT NULL"),
        ),
    )

    def __repr__(self) -> str:
        return f"<Chunk {self.symbol_path} L{self.start_line}-{self.end_line}>"


class EmbeddingCache(Base):
    """
    Content-addressed embedding cache, shared across repositories.

    Keyed by (model, content sha) rather than by chunk, so identical files in
    forks -- and unchanged files between commits -- are embedded once. The
    model id is part of the key because vectors from different models are not
    comparable, and mixing them is a silent failure rather than a loud one.
    """

    __tablename__ = "embedding_cache"

    model_id = Column(String(128), primary_key=True)
    content_sha = Column(String(64), primary_key=True)
    embedding = Column(HALFVEC(EMBED_DIM), nullable=False)
    embedding_bits = Column(BIT(EMBED_DIM), nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    last_used_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    __table_args__ = (Index("ix_embcache_last_used", "last_used_at"),)


class IngestJob(Base):
    """
    Durable work queue, claimed with `FOR UPDATE SKIP LOCKED`.

    v1 used FastAPI `BackgroundTasks`, which dies with the process: a
    spin-down mid-ingest left the repository permanently in `status
    ='ingesting'` with no retry and no way to notice. Rows here are claimed
    atomically, retried with backoff, and dead-lettered after `max_attempts`.

    Keeping this in Postgres means enqueueing a job and updating the
    repository row happen in one transaction.
    """

    __tablename__ = "ingest_jobs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    repo_id = Column(
        UUID(as_uuid=False),
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
    )
    kind = Column(String(32), nullable=False)  # full_index | incremental | delete
    payload = Column(JSONB, nullable=False, server_default="{}")

    status = Column(String(16), nullable=False, server_default="queued")
    attempts = Column(Integer, nullable=False, server_default="0")
    max_attempts = Column(Integer, nullable=False, server_default="3")

    run_after = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    locked_by = Column(String(128), nullable=True)
    locked_at = Column(DateTime(timezone=True), nullable=True)

    # Heartbeat. A worker that dies mid-job leaves a stale lock; the reaper
    # reclaims rows whose heartbeat has gone quiet rather than trusting the
    # worker to clean up after itself.
    heartbeat_at = Column(DateTime(timezone=True), nullable=True)

    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    __table_args__ = (
        # The polling index. Partial, so it stays small as completed jobs
        # accumulate.
        Index(
            "ix_jobs_claimable",
            "run_after",
            postgresql_where=text("status = 'queued'"),
        ),
        Index(
            "ix_jobs_stale_locks",
            "heartbeat_at",
            postgresql_where=text("status = 'running'"),
        ),
        Index("ix_jobs_repo", "repo_id"),
    )

    def __repr__(self) -> str:
        return f"<IngestJob {self.id} {self.kind} [{self.status}]>"


class ProviderUsage(Base):
    """
    Per-provider, per-window token and request counters for the LLM router.

    Free-tier limits bind on tokens, not requests: Groq's free
    `gpt-oss-120b` allows 8K tokens/minute and 200K/day, and Cerebras caps
    free-tier context at 8,192. The router needs to route *away* from a
    provider before it 429s, which means counting what has already been
    spent in the current window.

    A counter table rather than Redis: at the traffic this supports
    (~500-800 queries/day across all providers) an atomic
    `UPDATE ... RETURNING` is far faster than required, and it removes a
    service. Revisit above roughly 50 QPS.
    """

    __tablename__ = "provider_usage"

    provider = Column(String(32), primary_key=True)
    model = Column(String(128), primary_key=True)
    # `window` is reserved in Postgres; see migration 0002.
    window_kind = Column(String(8), primary_key=True)  # "minute" | "day"
    window_start = Column(DateTime(timezone=True), primary_key=True)

    requests = Column(Integer, nullable=False, server_default="0")
    input_tokens = Column(BigInteger, nullable=False, server_default="0")
    output_tokens = Column(BigInteger, nullable=False, server_default="0")
    errors = Column(Integer, nullable=False, server_default="0")
    updated_at = Column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    __table_args__ = (Index("ix_usage_window", "window_kind", "window_start"),)
