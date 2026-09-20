"""
Tests for streaming ingestion.

The properties asserted here are the ones whose absence produced the
measured v1 failures: unbounded memory (409 MB at 25k chunks), a
non-transactional status flip (`status='ready'` pointing at a vanished
index), and no incremental reuse (a one-line commit re-embedded everything).

A deterministic fake embedder is used so these run in seconds without
loading 320 MB of weights; the real embedder has its own correctness tests
in test_embedder.py.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import text

from app.indexing.chunker import ASTChunker
from app.indexing.pipeline import (
    IndexStats,
    index_repository,
    iter_chunks,
    iter_source_files,
)

REPO = "cccc0000-0000-4000-8000-000000000003"
DIM = 768


class FakeEmbedder:
    """Deterministic unit vectors derived from the text; no model needed."""

    def __init__(self):
        self.calls = 0
        self.texts_embedded = 0
        self.max_batch_seen = 0

    def embed(self, texts, batch_size=16, sort_by_length=True):
        self.calls += 1
        self.texts_embedded += len(texts)
        self.max_batch_seen = max(self.max_batch_seen, len(texts))
        out = np.empty((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            v = rng.normal(size=DIM).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out


@pytest.fixture
def chunker():
    # min_tokens=1: the fixture files are tiny, and the default merge policy
    # would coalesce their definitions into one chunk -- correct behaviour in
    # production, but it would make these tests assert the merger rather than
    # the pipeline.
    return ASTChunker(
        count_tokens=lambda s: max(1, len(s.split())), min_tokens=1
    )


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "app").mkdir(parents=True)
    (root / "node_modules" / "junk").mkdir(parents=True)
    (root / "app" / "auth.py").write_text(
        'def get_current_user(token):\n'
        '    """Resolve a user from a bearer token."""\n'
        '    return decode(token)\n\n'
        'def hash_password(pw):\n'
        '    return bcrypt.hashpw(pw)\n',
        encoding="utf-8",
    )
    (root / "app" / "util.py").write_text(
        "def slugify(value):\n    return value.lower().replace(' ', '-')\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# Demo\n\nThis repository exists for the ingestion tests.\n", encoding="utf-8"
    )
    # Must be skipped.
    (root / "node_modules" / "junk" / "bundle.js").write_text(
        "function x(){return 1}\n", encoding="utf-8"
    )
    (root / "app" / "vendor.min.js").write_text("var a=1;\n", encoding="utf-8")
    return root


@pytest.fixture
async def repo_row(async_session, conn):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO repositories (id, name, local_path, status) "
            "VALUES (%s, 'p', '/tmp/p', 'pending')",
            (REPO,),
        )
    return REPO


async def _index(session, root, chunker, embedder, **kw):
    return await index_repository(
        session, REPO, root, embedder, chunker,
        model_id="fake/test-model", batch_size=kw.pop("batch_size", 4), **kw,
    )


# --- file selection --------------------------------------------------------

def test_vendored_and_generated_files_are_skipped(tree):
    found = {p.name for p in iter_source_files(tree)}
    assert {"auth.py", "util.py", "README.md"} <= found
    assert "bundle.js" not in found, "node_modules must be skipped"
    assert "vendor.min.js" not in found, "minified bundles must be skipped"


def test_iter_chunks_is_lazy(tree, chunker):
    """
    A generator, not a list. v1's `chunk_repository` returned every chunk at
    once, which is the first structure that made ingestion O(repo) in memory.
    """
    import types

    gen = iter_chunks(tree, REPO, chunker, IndexStats())
    assert isinstance(gen, types.GeneratorType)
    first = next(gen)
    assert first.chunk_id


# --- indexing --------------------------------------------------------------

async def test_index_writes_chunks_and_marks_ready(async_session, repo_row, tree, chunker):
    emb = FakeEmbedder()
    stats = await _index(async_session, tree, chunker, emb)

    assert stats.chunks_written > 0
    assert stats.files_indexed >= 3

    row = (
        await async_session.execute(
            text("SELECT status, chunk_count, indexed_chunk_count, index_version, "
                 "last_indexed_at FROM repositories WHERE id = :i"),
            {"i": REPO},
        )
    ).one()
    assert row.status == "ready"
    assert row.chunk_count == stats.chunks_written
    assert row.last_indexed_at is not None

    n = (
        await async_session.execute(
            text("SELECT count(*) FROM chunks WHERE repo_id = :i"), {"i": REPO}
        )
    ).scalar_one()
    assert n == stats.chunks_written


# --- symlink containment ---------------------------------------------------
#
# A cloned repository is attacker-controlled content. These run wherever the
# host can create symlinks; on Windows that needs elevation, but the
# deployment target is Linux and CI is Linux, so they execute where the
# vulnerability is reachable.

NLC = chr(10)  # a newline, written without escapes


def _can_symlink(tmp_path) -> bool:
    try:
        (tmp_path / "_probe_target").write_text("x")
        (tmp_path / "_probe_link").symlink_to(tmp_path / "_probe_target")
    except (OSError, NotImplementedError):
        return False
    return True


@pytest.fixture
def symlinks(tmp_path):
    if not _can_symlink(tmp_path):
        pytest.skip("this host cannot create symlinks (needs elevation on Windows)")
    return tmp_path


def test_a_symlink_out_of_the_checkout_is_not_indexed(symlinks, tmp_path_factory):
    """
    The exfiltration path.

    Without this guard a repository containing `notes.txt -> /etc/passwd`
    -- or `-> /proc/self/environ`, which holds DATABASE_URL and
    EMBEDDING_API_KEY on the deployed indexer -- has that file read,
    chunked, embedded and stored as a chunk the submitter can retrieve by
    searching their own repository.
    """
    from app.indexing.pipeline import iter_source_files

    outside = tmp_path_factory.mktemp("outside")
    secret = outside / "secret.env"
    secret.write_text("DATABASE_URL=postgresql://u:REAL_PASSWORD@db/prod" + NLC)

    root = symlinks / "repo"
    root.mkdir()
    (root / "real.py").write_text("def ok():" + NLC + "    return 1" + NLC)
    # A name the extension filter happily admits.
    (root / "notes.txt").symlink_to(secret)
    (root / "config.yml").symlink_to(secret)

    found = {p.name for p in iter_source_files(root)}
    assert found == {"real.py"}, f"a symlink escaped the checkout: {found}"


def test_a_file_under_a_symlinked_directory_is_not_indexed(symlinks, tmp_path_factory):
    """
    The second door. The file itself is not a link, so an `is_symlink()`
    check on the file alone would pass it; only resolving against the root
    catches a symlinked parent.
    """
    from app.indexing.pipeline import iter_source_files

    outside = tmp_path_factory.mktemp("outside_dir")
    (outside / "leak.py").write_text("SECRET = 'exfiltrated'" + NLC)

    root = symlinks / "repo2"
    root.mkdir()
    (root / "keep.py").write_text("def ok():" + NLC + "    return 1" + NLC)
    try:
        (root / "vendored").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable on this host")

    found = {p.name for p in iter_source_files(root)}
    assert "leak.py" not in found, "a symlinked directory leaked its contents"
    assert found == {"keep.py"}


def test_ordinary_files_are_still_indexed(symlinks):
    """The guard must not throw the repository out with the symlinks."""
    from app.indexing.pipeline import iter_source_files

    root = symlinks / "repo3"
    (root / "pkg").mkdir(parents=True)
    (root / "a.py").write_text("def a():" + NLC + "    return 1" + NLC)
    (root / "pkg" / "b.py").write_text("def b():" + NLC + "    return 2" + NLC)
    (root / "README.md").write_text("# hello" + NLC)

    found = {p.name for p in iter_source_files(root)}
    assert found == {"a.py", "b.py", "README.md"}


async def test_live_progress_counters_are_published(
    async_session, repo_row, tree, chunker
):
    """
    The UI renders a chunk counter and a cache badge while indexing.

    Those columns were never written: the whole progress panel sat empty
    for the length of the run, which on a 3,600-chunk repository is about
    26 minutes of a dashboard that looks stuck. The counters are written
    per batch, so this asserts they survive to the end of a real index.
    """
    emb = FakeEmbedder()
    stats = await _index(async_session, tree, chunker, emb)

    row = (
        await async_session.execute(
            text(
                "SELECT ingestion_total_chunks, ingestion_cached_chunks, "
                "ingestion_phase FROM repositories WHERE id = :i"
            ),
            {"i": REPO},
        )
    ).one()
    assert row.ingestion_total_chunks == stats.chunks_written > 0, (
        "the chunk counter the dashboard reads was never written"
    )
    assert row.ingestion_phase == "done"


async def test_cached_chunk_counter_reflects_a_reindex(
    async_session, repo_row, tree, chunker
):
    """
    Second pass over unchanged content is served from the embedding cache,
    and the badge that advertises that must be driven by a real count.
    """
    emb = FakeEmbedder()
    await _index(async_session, tree, chunker, emb)
    stats = await _index(async_session, tree, chunker, emb)
    assert stats.cache_hits > 0, "the fixture should re-hit the cache"

    cached = (
        await async_session.execute(
            text("SELECT ingestion_cached_chunks FROM repositories WHERE id = :i"),
            {"i": REPO},
        )
    ).scalar_one()
    assert cached == stats.cache_hits


async def test_every_chunk_gets_both_vector_representations(
    async_session, repo_row, tree, chunker
):
    await _index(async_session, tree, chunker, FakeEmbedder())
    missing = (
        await async_session.execute(
            text("SELECT count(*) FROM chunks WHERE repo_id = :i "
                 "AND (embedding IS NULL OR embedding_bits IS NULL "
                 "     OR search_vector IS NULL)"),
            {"i": REPO},
        )
    ).scalar_one()
    assert missing == 0


async def test_symbols_and_scopes_are_persisted(async_session, repo_row, tree, chunker):
    await _index(async_session, tree, chunker, FakeEmbedder())
    symbols = {
        r.symbol
        for r in await async_session.execute(
            text("SELECT symbol FROM chunks WHERE repo_id = :i"), {"i": REPO}
        )
    }
    assert {"get_current_user", "hash_password", "slugify"} <= symbols


# --- incremental reuse -----------------------------------------------------

async def test_reindexing_unchanged_content_hits_the_cache(
    async_session, repo_row, tree, chunker
):
    """
    The property that makes a free embedding budget survive re-indexing:
    identical bodies are embedded once, ever.
    """
    first = await _index(async_session, tree, chunker, FakeEmbedder())
    assert first.cache_misses > 0
    assert first.cache_hit_rate == 0.0

    second_embedder = FakeEmbedder()
    second = await _index(async_session, tree, chunker, second_embedder)

    assert second.cache_misses == 0, "nothing changed; nothing should be re-embedded"
    assert second.cache_hit_rate == 1.0
    assert second_embedder.texts_embedded == 0
    assert second.chunks_written == first.chunks_written


async def test_only_changed_content_is_re_embedded(
    async_session, repo_row, tree, chunker
):
    await _index(async_session, tree, chunker, FakeEmbedder())

    (tree / "app" / "util.py").write_text(
        "def slugify(value):\n    return value.strip().lower().replace(' ', '_')\n",
        encoding="utf-8",
    )
    emb = FakeEmbedder()
    stats = await _index(async_session, tree, chunker, emb)

    assert stats.cache_misses == 1, f"expected 1 changed body, got {stats.cache_misses}"
    assert emb.texts_embedded == 1


async def test_deleted_files_have_their_chunks_removed(
    async_session, repo_row, tree, chunker
):
    """Mark and sweep: a file removed from the tree must leave the index."""
    await _index(async_session, tree, chunker, FakeEmbedder())
    before = (
        await async_session.execute(
            text("SELECT count(*) FROM chunks WHERE repo_id = :i AND file_path = 'app/util.py'"),
            {"i": REPO},
        )
    ).scalar_one()
    assert before > 0

    (tree / "app" / "util.py").unlink()
    stats = await _index(async_session, tree, chunker, FakeEmbedder())

    assert stats.chunks_deleted >= before
    after = (
        await async_session.execute(
            text("SELECT count(*) FROM chunks WHERE repo_id = :i AND file_path = 'app/util.py'"),
            {"i": REPO},
        )
    ).scalar_one()
    assert after == 0


async def test_chunk_ids_are_stable_across_reindex(
    async_session, repo_row, tree, chunker
):
    """Stable ids are what make the mark-and-sweep an update, not a churn."""
    await _index(async_session, tree, chunker, FakeEmbedder())
    first = {
        r.chunk_id
        for r in await async_session.execute(
            text("SELECT chunk_id FROM chunks WHERE repo_id = :i"), {"i": REPO}
        )
    }
    # Shift every line down without changing any definition.
    p = tree / "app" / "auth.py"
    p.write_text("import os\nimport sys\n\n\n" + p.read_text(encoding="utf-8"),
                 encoding="utf-8")

    stats = await _index(async_session, tree, chunker, FakeEmbedder())
    second = {
        r.chunk_id
        for r in await async_session.execute(
            text("SELECT chunk_id FROM chunks WHERE repo_id = :i"), {"i": REPO}
        )
    }
    assert first == second, "line drift must not churn chunk ids"
    assert stats.chunks_deleted == 0


# --- memory discipline -----------------------------------------------------

async def test_embedder_never_sees_more_than_one_batch(
    async_session, repo_row, tree, chunker
):
    """
    The whole point of the rewrite. v1 handed every embedding to the store
    at once, measured at 409 MB for 25k chunks; here the embedder is called
    per batch and never receives more than `batch_size` texts.
    """
    emb = FakeEmbedder()
    await _index(async_session, tree, chunker, emb, batch_size=2)
    assert emb.max_batch_seen <= 2, f"batch bound violated: {emb.max_batch_seen}"
    assert emb.calls >= 2, "expected multiple flushes"


async def test_duplicate_bodies_within_one_batch_are_embedded_once(
    async_session, repo_row, tmp_path, chunker
):
    root = tmp_path / "dup"
    root.mkdir()
    body = "def identical():\n    return 42\n"
    for i in range(4):
        (root / f"m{i}.py").write_text(body, encoding="utf-8")

    emb = FakeEmbedder()
    stats = await index_repository(
        async_session, REPO, root, emb, chunker,
        model_id="fake/test-model", batch_size=16,
    )
    assert stats.chunks_written == 4
    assert emb.texts_embedded == 1, "four identical bodies, one embedding"


# --- failure handling ------------------------------------------------------

async def test_a_failure_marks_the_repository_failed_and_reraises(
    async_session, repo_row, tree, chunker
):
    class Exploding(FakeEmbedder):
        def embed(self, texts, batch_size=16, sort_by_length=True):
            raise RuntimeError("provider unavailable")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await _index(async_session, tree, chunker, Exploding())

    row = (
        await async_session.execute(
            text("SELECT status, error_message FROM repositories WHERE id = :i"),
            {"i": REPO},
        )
    ).one()
    assert row.status == "failed"
    assert "provider unavailable" in row.error_message


async def test_a_mid_run_failure_leaves_the_previous_index_intact(
    async_session, repo_row, tree, chunker
):
    """
    Sweep-last matters: a crash must not leave a half-deleted index. The
    repository should still have its previous chunks, marked failed.
    """
    await _index(async_session, tree, chunker, FakeEmbedder())
    before = (
        await async_session.execute(
            text("SELECT count(*) FROM chunks WHERE repo_id = :i"), {"i": REPO}
        )
    ).scalar_one()

    (tree / "app" / "new.py").write_text("def brand_new():\n    return 1\n", encoding="utf-8")

    class FailsOnNewContent(FakeEmbedder):
        def embed(self, texts, batch_size=16, sort_by_length=True):
            raise RuntimeError("died mid-run")

    with pytest.raises(RuntimeError):
        await _index(async_session, tree, chunker, FailsOnNewContent())

    after = (
        await async_session.execute(
            text("SELECT count(*) FROM chunks WHERE repo_id = :i"), {"i": REPO}
        )
    ).scalar_one()
    assert after == before, "previous index must survive a failed reindex"


async def test_empty_repository_completes_without_error(
    async_session, repo_row, tmp_path, chunker
):
    empty = tmp_path / "empty"
    empty.mkdir()
    stats = await index_repository(
        async_session, REPO, empty, FakeEmbedder(), chunker,
        model_id="fake/test-model",
    )
    assert stats.chunks_written == 0
    status = (
        await async_session.execute(
            text("SELECT status FROM repositories WHERE id = :i"), {"i": REPO}
        )
    ).scalar_one()
    assert status == "ready"
