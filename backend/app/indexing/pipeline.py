"""
Streaming ingestion: walk -> chunk -> embed -> upsert, in bounded batches.

This is the fix for the measured OOM, and the shape is the whole point.

v1 held four large structures alive at once -- every file's text, every
chunk, every embedding, and then handed all of them to the vector store.
`bench/bench_ingest_memory.py` measured that shape at 164 MB for 10k chunks,
409 MB for 25k and 819 MB for 50k, because the embedder returned
`list[list[float]]` at 10.2x the cost of the same numbers as float32. With
~143 MB already resident after imports, a 512 MB container is exhausted at
roughly 22,000 chunks -- one medium repository.

Here nothing accumulates except a set of chunk ids (24 bytes each; ~7 MB at
100k chunks, and required for the mark-and-sweep delete). Files are chunked
and flushed in batches, embeddings are numpy float32, and each batch is
committed before the next is read, so peak memory is a function of
`batch_size` rather than of repository size.

Two further properties that v1 could not have:

  Transactional.   The final status flip and the chunk rows commit together.
                   v1 wrote vectors to ChromaDB on an ephemeral disk and the
                   status to Postgres; a restart left `status='ready'`
                   pointing at an index that no longer existed, and
                   retrieval silently returned zero chunks.

  Resumable-ish.   Work already done is cheap to redo: the embedding cache
                   is content-addressed, so a retried job re-embeds only
                   what actually changed.

NOTE on the HNSW index: it is NOT dropped around a bulk load. The index is
global to the `chunks` table, so dropping it to speed up one repository's
ingest would remove it for every other repository's queries at the same
time. (An earlier comment in migration 0002 suggested otherwise; it was
wrong for a multi-tenant table.)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.indexing.chunker import ASTChunker, Chunk
from app.indexing.embedder import (
    EMBED_DIM,
    binary_quantize,
    bits_to_sql,
    vector_to_sql,
)
from app.indexing.languages import is_prose, language_name_for_path

logger = logging.getLogger(__name__)

# Chunks per flush. Sets peak memory: 64 chunks of ~1.5 KB text plus a
# 64x768 float32 matrix is well under a megabyte, so the ceiling is the
# batch, not the repository.
DEFAULT_BATCH_SIZE = 64

# Files larger than this are skipped: generated bundles, lockfiles and
# vendored blobs cost embedding budget and answer no question anyone asks.
MAX_FILE_BYTES = 1_000_000

SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "venv", ".venv", "env",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", "out", ".next", "target", "vendor", "third_party",
    ".tox", ".idea", ".vscode", "site-packages", ".gradle",
})

SKIP_SUFFIXES = frozenset({
    ".min.js", ".min.css", ".map", ".lock", ".sum", ".pyc", ".so", ".dll",
    ".dylib", ".class", ".jar", ".zip", ".tar", ".gz", ".png", ".jpg",
    ".jpeg", ".gif", ".svg", ".ico", ".pdf", ".woff", ".woff2", ".ttf",
})


@dataclass
class IndexStats:
    files_seen: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    chunks_written: int = 0
    chunks_deleted: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    embed_seconds: float = 0.0
    total_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0

    def summary(self) -> str:
        return (
            f"{self.files_indexed}/{self.files_seen} files, "
            f"{self.chunks_written} chunks written, {self.chunks_deleted} removed, "
            f"cache {self.cache_hit_rate:.0%} ({self.cache_hits}/"
            f"{self.cache_hits + self.cache_misses}), "
            f"embed {self.embed_seconds:.1f}s of {self.total_seconds:.1f}s"
        )


def iter_source_files(root: Path) -> Iterator[Path]:
    """Yield indexable files, skipping vendored and generated trees."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        name = path.name.lower()
        if any(name.endswith(suffix) for suffix in SKIP_SUFFIXES):
            continue
        if language_name_for_path(str(path)) is None and not is_prose(str(path)):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def iter_chunks(
    root: Path, repo_id: str, chunker: ASTChunker, stats: IndexStats
) -> Iterator[Chunk]:
    """
    Chunk the tree lazily.

    A generator, not a list: `chunk_repository` in v1 returned every chunk
    for the whole repository at once, which is the first of the four
    structures that made ingestion O(repo) in memory.
    """
    for path in iter_source_files(root):
        stats.files_seen += 1
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            stats.files_skipped += 1
            stats.errors.append(f"{path}: {exc}")
            continue

        rel = path.relative_to(root).as_posix()
        chunks = chunker.chunk_file(repo_id, rel, source)
        if chunks:
            stats.files_indexed += 1
        else:
            stats.files_skipped += 1
        yield from chunks


def _batched(items: Iterator[Chunk], size: int) -> Iterator[list[Chunk]]:
    batch: list[Chunk] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


_INSERT_CHUNK_CACHED = text(
    """
    INSERT INTO chunks (
        chunk_id, repo_id, content_sha, file_path, language, kind, node_type,
        symbol, parent_scope, symbol_path, start_line, end_line, token_count,
        part, part_of, content, index_version, embedding, embedding_bits
    )
    -- `content_sha` appears twice with different inferred types (an
    -- inserted varchar and a WHERE comparand), which asyncpg rejects as
    -- "inconsistent types deduced for parameter". Two distinct bind names
    -- carrying the same value avoids the deduction entirely.
    SELECT :chunk_id, :repo_id, :content_sha, :file_path, :language, :kind,
           :node_type, :symbol, :parent_scope, :symbol_path, :start_line,
           :end_line, :token_count, :part, :part_of, :content, :index_version,
           ec.embedding, ec.embedding_bits
      FROM embedding_cache ec
     WHERE ec.model_id = :model_id AND ec.content_sha = :sha_lookup
    ON CONFLICT (chunk_id) DO UPDATE SET
        content_sha = EXCLUDED.content_sha,
        file_path = EXCLUDED.file_path,
        language = EXCLUDED.language,
        kind = EXCLUDED.kind,
        node_type = EXCLUDED.node_type,
        symbol = EXCLUDED.symbol,
        parent_scope = EXCLUDED.parent_scope,
        symbol_path = EXCLUDED.symbol_path,
        start_line = EXCLUDED.start_line,
        end_line = EXCLUDED.end_line,
        token_count = EXCLUDED.token_count,
        part = EXCLUDED.part,
        part_of = EXCLUDED.part_of,
        content = EXCLUDED.content,
        index_version = EXCLUDED.index_version,
        embedding = EXCLUDED.embedding,
        embedding_bits = EXCLUDED.embedding_bits
    """
)

_INSERT_CHUNK_FRESH = text(
    f"""
    INSERT INTO chunks (
        chunk_id, repo_id, content_sha, file_path, language, kind, node_type,
        symbol, parent_scope, symbol_path, start_line, end_line, token_count,
        part, part_of, content, index_version, embedding, embedding_bits
    ) VALUES (
        :chunk_id, :repo_id, :content_sha, :file_path, :language, :kind,
        :node_type, :symbol, :parent_scope, :symbol_path, :start_line,
        :end_line, :token_count, :part, :part_of, :content, :index_version,
        CAST(CAST(:embedding AS text) AS halfvec({EMBED_DIM})),
        CAST(CAST(:bits AS text) AS bit({EMBED_DIM}))
    )
    ON CONFLICT (chunk_id) DO UPDATE SET
        content_sha = EXCLUDED.content_sha,
        file_path = EXCLUDED.file_path,
        language = EXCLUDED.language,
        kind = EXCLUDED.kind,
        node_type = EXCLUDED.node_type,
        symbol = EXCLUDED.symbol,
        parent_scope = EXCLUDED.parent_scope,
        symbol_path = EXCLUDED.symbol_path,
        start_line = EXCLUDED.start_line,
        end_line = EXCLUDED.end_line,
        token_count = EXCLUDED.token_count,
        part = EXCLUDED.part,
        part_of = EXCLUDED.part_of,
        content = EXCLUDED.content,
        index_version = EXCLUDED.index_version,
        embedding = EXCLUDED.embedding,
        embedding_bits = EXCLUDED.embedding_bits
    """
)

_UPSERT_CACHE = text(
    f"""
    INSERT INTO embedding_cache (model_id, content_sha, embedding, embedding_bits)
    VALUES (
        :model_id, :content_sha,
        CAST(CAST(:embedding AS text) AS halfvec({EMBED_DIM})),
        CAST(CAST(:bits AS text) AS bit({EMBED_DIM}))
    )
    ON CONFLICT (model_id, content_sha) DO UPDATE SET last_used_at = now()
    """
)


def _chunk_params(chunk: Chunk, repo_id: str, index_version: int) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "repo_id": repo_id,
        "content_sha": chunk.content_sha,
        "file_path": chunk.file_path,
        "language": chunk.language,
        "kind": chunk.kind,
        "node_type": chunk.node_type,
        "symbol": chunk.symbol,
        "parent_scope": ".".join(chunk.parent_scope),
        "symbol_path": chunk.symbol_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "token_count": chunk.token_count,
        "part": chunk.part,
        "part_of": chunk.part_of,
        "content": chunk.content,
        "index_version": index_version,
    }


async def _flush_batch(
    db: AsyncSession,
    batch: list[Chunk],
    repo_id: str,
    embedder,
    model_id: str,
    index_version: int,
    stats: IndexStats,
) -> None:
    """Embed the batch's cache misses and upsert every chunk in it."""
    # One query tells us which of these bodies we have already embedded.
    # Identical content across forks, commits, or repositories is embedded
    # once -- which matters most when the embedding budget is a free tier.
    shas = list({c.content_sha for c in batch})
    cached_rows = await db.execute(
        text(
            "SELECT content_sha FROM embedding_cache "
            "WHERE model_id = :model_id AND content_sha = ANY(:shas)"
        ),
        {"model_id": model_id, "shas": shas},
    )
    cached: set[str] = {r.content_sha for r in cached_rows}

    # Deduplicate within the batch too: two identical bodies in one flush
    # must not be embedded twice.
    to_embed: dict[str, Chunk] = {
        c.content_sha: c for c in batch if c.content_sha not in cached
    }
    stats.cache_hits += len(batch) - sum(
        1 for c in batch if c.content_sha in to_embed
    )
    stats.cache_misses += len(to_embed)

    vectors: dict[str, np.ndarray] = {}
    if to_embed:
        shas_order = list(to_embed)
        t0 = time.perf_counter()
        matrix = embedder.embed([to_embed[s].content for s in shas_order])
        stats.embed_seconds += time.perf_counter() - t0
        packed = binary_quantize(matrix)
        for i, sha in enumerate(shas_order):
            vectors[sha] = matrix[i]
        # Write to the cache first: if the chunk insert fails and the job is
        # retried, the embedding work is not repeated.
        await db.execute(
            _UPSERT_CACHE,
            [
                {
                    "model_id": model_id,
                    "content_sha": sha,
                    "embedding": vector_to_sql(matrix[i]),
                    "bits": bits_to_sql(packed[i], EMBED_DIM),
                }
                for i, sha in enumerate(shas_order)
            ],
        )

    fresh_params, cached_params = [], []
    for chunk in batch:
        params = _chunk_params(chunk, repo_id, index_version)
        if chunk.content_sha in vectors:
            vec = vectors[chunk.content_sha]
            params["embedding"] = vector_to_sql(vec)
            params["bits"] = bits_to_sql(
                binary_quantize(vec.reshape(1, -1))[0], EMBED_DIM
            )
            fresh_params.append(params)
        else:
            params["model_id"] = model_id
            params["sha_lookup"] = chunk.content_sha
            cached_params.append(params)

    if fresh_params:
        await db.execute(_INSERT_CHUNK_FRESH, fresh_params)
    if cached_params:
        await db.execute(_INSERT_CHUNK_CACHED, cached_params)

    stats.chunks_written += len(batch)


async def index_repository(
    db: AsyncSession,
    repo_id: str,
    root: Path,
    embedder,
    chunker: ASTChunker,
    *,
    model_id: str,
    index_version: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    commit_id: str | None = None,
    heartbeat=None,
) -> IndexStats:
    """
    Index a checked-out repository into Postgres.

    Peak memory is a function of `batch_size`, not repository size. The
    only structure that grows with the repository is the set of chunk ids
    needed for the mark-and-sweep delete at the end.

    `heartbeat` is an optional awaitable called between batches so a long
    ingest keeps its queue lock alive.
    """
    stats = IndexStats()
    started = time.perf_counter()

    await db.execute(
        text(
            "UPDATE repositories SET status = 'ingesting', error_message = NULL, "
            "ingestion_phase = 'index', updated_at = now() WHERE id = :id"
        ),
        {"id": repo_id},
    )
    await db.commit()

    seen: set[str] = set()
    try:
        for batch in _batched(iter_chunks(root, repo_id, chunker, stats), batch_size):
            await _flush_batch(
                db, batch, repo_id, embedder, model_id, index_version, stats
            )
            seen.update(c.chunk_id for c in batch)

            # Publish live counts. The frontend renders a chunk counter and
            # a cache badge during indexing, and before this they were never
            # written -- the whole progress panel sat empty for the length
            # of the run, which on a 3,600-chunk repository is ~26 minutes.
            #
            # No percentage: `iter_chunks` is a generator, so the total is
            # genuinely unknown until the walk finishes. Reporting a
            # fraction of an unknown total would mean inventing one.
            await db.execute(
                text(
                    "UPDATE repositories SET "
                    "ingestion_total_chunks = :written, "
                    "ingestion_cached_chunks = :cached, "
                    "updated_at = now() WHERE id = :id"
                ),
                {
                    "id": repo_id,
                    "written": stats.chunks_written,
                    "cached": stats.cache_hits,
                },
            )
            # Commit per batch. Bounded transactions keep peak memory and
            # lock duration proportional to the batch, and avoid the
            # long-transaction hazards documented in app/retrieval/hybrid.py.
            await db.commit()
            if heartbeat is not None:
                await heartbeat()

        # Mark and sweep: anything not re-seen this run no longer exists in
        # the tree. Done last so a crash mid-run leaves the previous index
        # intact rather than half-deleted.
        deleted = await db.execute(
            text(
                "DELETE FROM chunks WHERE repo_id = :repo_id "
                "AND NOT (chunk_id = ANY(:seen)) RETURNING chunk_id"
            ),
            {"repo_id": repo_id, "seen": list(seen)},
        )
        stats.chunks_deleted = len(deleted.fetchall())

        stats.total_seconds = time.perf_counter() - started
        # The status flip and the chunk rows commit together: there is no
        # window in which a repository claims to be ready while its index
        # is absent.
        await db.execute(
            text(
                "UPDATE repositories SET status = 'ready', ingestion_phase = 'done', "
                "ingestion_progress = 100, chunk_count = :n, "
                "indexed_chunk_count = :n, indexed_token_count = :tokens, "
                "index_version = :version, indexed_commit_sha = :commit, "
                "last_indexed_at = now(), updated_at = now() WHERE id = :id"
            ),
            {
                "id": repo_id, "n": len(seen), "tokens": 0,
                "version": index_version, "commit": commit_id,
            },
        )
        await db.commit()

    except Exception as exc:
        await db.rollback()
        await db.execute(
            text(
                "UPDATE repositories SET status = 'failed', "
                "error_message = :err, updated_at = now() WHERE id = :id"
            ),
            {"id": repo_id, "err": str(exc)[:2000]},
        )
        await db.commit()
        raise

    logger.info("indexed repo %s: %s", repo_id[:8], stats.summary())
    return stats
