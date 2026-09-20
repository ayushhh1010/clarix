"""
Behavioural tests for the AST chunker.

Each test pins a property that the v1 chunker got wrong, so a regression here
re-introduces a defect we have measured. bench/bench_chunking.py quantifies
these across a real corpus; these tests make them fail fast and locally.
"""

from __future__ import annotations

import pytest

from app.indexing.chunker import ASTChunker, Chunk

pytest.importorskip("tree_sitter_language_pack")


def words(text: str) -> int:
    """Cheap deterministic token counter -- keeps tests off the network."""
    return max(1, len(text.split()))


@pytest.fixture
def chunker() -> ASTChunker:
    return ASTChunker(count_tokens=words, max_tokens=200, min_tokens=8)


def by_symbol(chunks: list[Chunk], name: str) -> Chunk:
    matches = [c for c in chunks if c.symbol == name]
    assert matches, f"no chunk for {name!r}; got {[c.symbol for c in chunks]}"
    return matches[0]


# --- decorators ------------------------------------------------------------

def test_python_decorators_are_inside_the_chunk(chunker: ASTChunker):
    """v1 started at the `def` line, dropping the route path from the index."""
    source = '''
@router.post("/api/chat")
@requires_auth
async def chat_handler(request: ChatRequest) -> ChatResponse:
    """Answer a question about a repository."""
    return await run(request)
'''
    chunk = by_symbol(chunker.chunk_file("r", "app/routes/chat.py", source), "chat_handler")
    assert '@router.post("/api/chat")' in chunk.content
    assert "@requires_auth" in chunk.content
    assert chunk.kind == "definition"


def test_leading_comment_is_absorbed_but_distant_header_is_not(chunker: ASTChunker):
    source = '''# Copyright 2026
# SPDX-License-Identifier: MIT


# Resolve the active user from the bearer token.
def get_current_user(token: str):
    return decode(token)
'''
    chunk = by_symbol(chunker.chunk_file("r", "app/security.py", source), "get_current_user")
    assert "Resolve the active user" in chunk.content
    assert "SPDX" not in chunk.content, "licence header is not this symbol's doc"


# --- containers ------------------------------------------------------------

def test_class_methods_are_individually_retrievable(chunker: ASTChunker):
    """v1 emitted one chunk per class and skipped the body entirely."""
    source = '''
class MemoryManager:
    """Unified memory interface."""

    version = 2

    def load_conversation(self, conversation_id: str, limit: int = 20):
        return self.db.fetch(conversation_id, limit)

    def save_user_message(self, conversation_id: str, content: str):
        return self.db.insert(conversation_id, "user", content)
'''
    chunks = chunker.chunk_file("r", "app/memory/manager.py", source)
    symbols = {c.symbol for c in chunks}
    assert {"MemoryManager", "load_conversation", "save_user_message"} <= symbols

    header = by_symbol(chunks, "MemoryManager")
    assert header.kind == "container_header"
    assert "version = 2" in header.content
    assert "def save_user_message" not in header.content

    method = by_symbol(chunks, "save_user_message")
    assert method.parent_scope == ("MemoryManager",)
    assert method.symbol_path == "app/memory/manager.py::MemoryManager.save_user_message"


def test_decorated_class_is_still_a_container(chunker: ASTChunker):
    source = '''
@dataclass
class Settings:
    """App settings."""

    debug: bool = False

    def as_dict(self):
        return {"debug": self.debug}
'''
    chunks = chunker.chunk_file("r", "app/config.py", source)
    header = by_symbol(chunks, "Settings")
    assert "@dataclass" in header.content
    assert header.kind == "container_header"
    assert by_symbol(chunks, "as_dict").parent_scope == ("Settings",)


def test_decorated_function_is_not_treated_as_a_container(chunker: ASTChunker):
    """Regression: `decorated_definition` once matched the container branch."""
    source = '''
@lru_cache()
def get_settings():
    def inner():
        return 1
    return inner
'''
    chunks = chunker.chunk_file("r", "app/config.py", source)
    outer = by_symbol(chunks, "get_settings")
    assert outer.kind == "definition"
    assert "@lru_cache()" in outer.content
    assert "def inner" in outer.content, "nested function belongs to its parent"


# --- javascript / typescript ----------------------------------------------

def test_exported_typescript_declarations_keep_the_export_keyword(chunker: ASTChunker):
    source = """
export async function fetchRepos(token: string): Promise<Repo[]> {
  const res = await fetch("/api/repo", { headers: auth(token) });
  return res.json();
}

export class ApiClient {
  constructor(private base: string) {}

  async get(path: string) {
    return fetch(this.base + path);
  }
}
"""
    chunks = chunker.chunk_file("r", "src/lib/api.ts", source)
    symbols = {c.symbol for c in chunks}
    assert "fetchRepos" in symbols
    assert "ApiClient" in symbols
    assert by_symbol(chunks, "fetchRepos").content.startswith("export async function")


def test_arrow_function_component_is_captured(chunker: ASTChunker):
    source = """
export const Dashboard = ({ repos }: Props) => {
  const [active, setActive] = useState<string | null>(null);
  return <div>{repos.length}</div>;
};
"""
    chunks = chunker.chunk_file("r", "src/app/dashboard/page.tsx", source)
    assert any(c.symbol == "Dashboard" for c in chunks), [c.symbol for c in chunks]


# --- sizing ----------------------------------------------------------------

def test_oversized_definition_is_split_and_every_part_carries_the_signature():
    small = ASTChunker(count_tokens=words, max_tokens=40, min_tokens=1)
    body = "\n".join(f"    value_{i} = compute(index_{i}, other_{i})" for i in range(120))
    source = f"def enormous(index_0, other_0):\n{body}\n"

    chunks = small.chunk_file("r", "app/big.py", source)
    assert len(chunks) > 1, "should have split"
    assert all(c.part_of == len(chunks) for c in chunks)
    for part in chunks[1:]:
        assert "def enormous" in part.content, "each part must identify its parent"
    assert {c.part for c in chunks} == set(range(1, len(chunks) + 1))


def test_tiny_adjacent_definitions_are_merged():
    merging = ASTChunker(count_tokens=words, max_tokens=500, min_tokens=40)
    source = "\n".join(f"def get_{i}():\n    return {i}\n" for i in range(12))
    chunks = merging.chunk_file("r", "app/accessors.py", source)
    assert len(chunks) < 12, f"expected merging, got {len(chunks)} chunks"
    assert merging.stats.tiny_merged > 0


def test_no_chunk_exceeds_the_configured_budget_on_real_source():
    small = ASTChunker(count_tokens=words, max_tokens=60, min_tokens=1)
    source = "\n".join(
        f"class C{i}:\n" + "\n".join(f"    def m{j}(self):\n        return {j}" for j in range(8))
        for i in range(4)
    )
    for chunk in small.chunk_file("r", "app/wide.py", source):
        # Split parts re-add a signature header, so allow modest slack; the
        # invariant that matters is that nothing approaches the encoder window.
        assert chunk.token_count <= 60 * 2


# --- identity --------------------------------------------------------------

def test_chunk_id_is_stable_when_the_file_shifts(chunker: ASTChunker):
    """Incremental reindexing diffs on chunk_id; line drift must not churn it."""
    body = 'def handler():\n    return 1\n'
    before = chunker.chunk_file("r", "app/x.py", body)
    after = chunker.chunk_file("r", "app/x.py", "import os\nimport sys\n\n\n" + body)

    assert by_symbol(before, "handler").chunk_id == by_symbol(after, "handler").chunk_id
    assert by_symbol(before, "handler").start_line != by_symbol(after, "handler").start_line


def test_content_sha_tracks_content_not_position(chunker: ASTChunker):
    """content_sha is the embedding cache key: equal bodies must not re-embed."""
    a = chunker.chunk_file("r", "app/x.py", "def handler():\n    return 1\n")
    b = chunker.chunk_file("r", "app/x.py", "\n\ndef handler():\n    return 1\n")
    c = chunker.chunk_file("r", "app/x.py", "def handler():\n    return 2\n")

    assert by_symbol(a, "handler").content_sha == by_symbol(b, "handler").content_sha
    assert by_symbol(a, "handler").content_sha != by_symbol(c, "handler").content_sha


# --- fallbacks -------------------------------------------------------------

def test_unparsed_file_falls_back_to_token_windows(chunker: ASTChunker):
    source = "\n".join(f"- bullet point number {i} with some prose" for i in range(400))
    chunks = chunker.chunk_file("r", "README.md", source)
    assert chunks
    assert all(c.kind == "prose" for c in chunks)
    assert all(c.token_count <= chunker.window_tokens * 1.5 for c in chunks)


def test_syntactically_broken_source_still_yields_chunks(chunker: ASTChunker):
    """Real repositories contain files that do not parse. Never drop them."""
    source = "def broken(:\n    this is not python at all ][\n"
    assert chunker.chunk_file("r", "app/broken.py", source)


def test_empty_and_whitespace_files_produce_nothing(chunker: ASTChunker):
    assert chunker.chunk_file("r", "app/empty.py", "") == []
    assert chunker.chunk_file("r", "app/blank.py", "\n\n   \n") == []


def test_line_numbers_point_at_the_real_source(chunker: ASTChunker):
    source = "import os\n\n\ndef target():\n    return os.getcwd()\n"
    chunk = by_symbol(chunker.chunk_file("r", "app/x.py", source), "target")
    lines = source.split("\n")
    assert lines[chunk.start_line - 1].startswith("def target")
    assert chunk.end_line >= chunk.start_line


# --- id uniqueness ---------------------------------------------------------

def test_chunk_ids_are_unique_within_a_file(chunker: ASTChunker):
    """
    Duplicate ids mean silent data loss: an upsert keyed on chunk_id keeps
    only the last writer. Symbol-less chunks all derive the same symbol_path
    (the bare file path), so this is the case that actually bites.
    """
    source = """
const a = 1;
const b = 2;
export default { a, b };

function named() { return a; }

(function () { return b; })();
(function () { return a + b; })();
"""
    chunks = chunker.chunk_file("r", "src/anon.js", source)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), f"duplicate chunk_id among {len(ids)} chunks"


def test_disambiguated_ids_are_reproducible(chunker: ASTChunker):
    source = "const a = 1;\nconst b = 2;\nconst c = 3;\n"
    first = [c.chunk_id for c in chunker.chunk_file("r", "src/x.js", source)]
    second = [c.chunk_id for c in chunker.chunk_file("r", "src/x.js", source)]
    assert first == second


def test_ids_unique_across_the_whole_corpus_sample(chunker: ASTChunker):
    """Same symbol name in different files must not collide."""
    a = chunker.chunk_file("r", "app/one.py", "def handler():\n    return 1\n")
    b = chunker.chunk_file("r", "app/two.py", "def handler():\n    return 1\n")
    assert {c.chunk_id for c in a}.isdisjoint({c.chunk_id for c in b})
    # ...but identical content must still share a content_sha, so the
    # embedding cache deduplicates it.
    assert a[0].content_sha == b[0].content_sha
