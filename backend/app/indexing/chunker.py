"""
AST-grounded code chunker.

Replaces the v1 indentation-heuristic chunker. Three defects motivated the
rewrite, all of them measurable (see bench/bench_chunking.py):

  1. Decorators were dropped. v1 started a chunk at the `def` line, so
     `@router.post("/api/chat")` above a handler never entered the index --
     removing the HTTP verb and route path from the searchable text.

  2. Classes were indexed as one chunk and their methods not at all. v1's
     `_chunk_by_structure` consumed a class up to the next dedent and then
     advanced past the whole body, so a 600-line class produced exactly one
     oversized chunk and zero method-level chunks.

  3. JS/TS block detection was a stated guess. The v1 source comments its own
     brace handling as a "simple heuristic"; arrow functions, object methods
     and class fields fell through to line-window chunking.

Design notes
------------
Chunk identity is split in two, because the two jobs differ:

  chunk_id     sha256(repo, path, symbol path, ordinal) -- stable while a
               symbol keeps its name, even if the file shifts by 200 lines.
               This is what incremental reindexing diffs against.

  content_sha  sha256(content) -- the embedding cache key. Unchanged content
               is never re-embedded, which matters when the embedding budget
               is a free tier.

Sizing is in tokens, never lines. A 50-line block ranges from ~120 tokens
(sparse Go) to ~900 (dense minified TS); a line budget cannot bound what the
encoder actually sees, and the encoder window is 8,192.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.indexing.languages import (
    LanguageSpec,
    is_prose,
    language_name_for_path,
    spec_for_path,
)

if TYPE_CHECKING:  # pragma: no cover
    from tree_sitter import Node

logger = logging.getLogger(__name__)


# --- Sizing policy ---------------------------------------------------------
#
# MAX is set well below the encoder's 8,192 limit: chunks near the ceiling
# crowd out every other result once 5-6 of them are packed into a 4,000-token
# generation budget. MIN prevents chunk explosion on files of one-line
# accessors, where thousands of 15-token chunks would dilute the index.
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MIN_TOKENS = 64
DEFAULT_WINDOW_TOKENS = 512
DEFAULT_WINDOW_OVERLAP_TOKENS = 64

# A definition whose *own* size exceeds this is split into parts, each
# prefixed with the signature so the fragment stays self-describing.
SPLIT_HEADER_MAX_LINES = 6


@dataclass(slots=True)
class Chunk:
    """One indexed unit of a repository."""

    chunk_id: str
    content_sha: str
    repo_id: str
    file_path: str
    language: str
    content: str
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    token_count: int
    kind: str  # definition | container_header | window | prose
    node_type: str | None = None
    symbol: str | None = None
    parent_scope: tuple[str, ...] = ()
    part: int = 1
    part_of: int = 1

    @property
    def symbol_path(self) -> str:
        """Fully-qualified symbol, e.g. `app/db.py::Settings.repos_path`."""
        parts = [*self.parent_scope]
        if self.symbol:
            parts.append(self.symbol)
        return f"{self.file_path}::{'.'.join(parts)}" if parts else self.file_path


@dataclass
class ChunkerStats:
    """Per-run counters. Surfaced in benchmarks and ingestion logs."""

    files_seen: int = 0
    files_parsed: int = 0
    files_windowed: int = 0
    files_failed: int = 0
    parse_errors: int = 0
    definitions_found: int = 0
    containers_found: int = 0
    oversized_split: int = 0
    tiny_merged: int = 0
    chunks_emitted: int = 0
    by_language: dict[str, int] = field(default_factory=dict)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


class ASTChunker:
    """
    Chunks source files along real syntax boundaries.

    Grammars are loaded lazily and cached per language, so a repository of one
    language never pays to load the other twelve.
    """

    def __init__(
        self,
        count_tokens: Callable[[str], int],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        min_tokens: int = DEFAULT_MIN_TOKENS,
        window_tokens: int = DEFAULT_WINDOW_TOKENS,
        window_overlap: int = DEFAULT_WINDOW_OVERLAP_TOKENS,
    ):
        self._count = count_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.window_tokens = window_tokens
        self.window_overlap = window_overlap
        self._parsers: dict[str, object] = {}
        self.stats = ChunkerStats()

    # -- grammar loading ---------------------------------------------------

    def _parser(self, grammar: str):
        cached = self._parsers.get(grammar)
        if cached is not None:
            return cached
        from tree_sitter_language_pack import get_parser

        parser = get_parser(grammar)
        self._parsers[grammar] = parser
        return parser

    # -- public API --------------------------------------------------------

    def chunk_file(self, repo_id: str, file_path: str, source: str) -> list[Chunk]:
        """Chunk one file. Never raises: a failure degrades to windowing."""
        self.stats.files_seen += 1
        if not source.strip():
            return []

        spec = spec_for_path(file_path)
        language = language_name_for_path(file_path) or ("prose" if is_prose(file_path) else "text")

        chunks: list[Chunk]
        if spec is None:
            self.stats.files_windowed += 1
            chunks = self._window(repo_id, file_path, source, language, kind="prose")
        else:
            try:
                chunks = self._chunk_with_ast(repo_id, file_path, source, spec, language)
                self.stats.files_parsed += 1
            except Exception as exc:  # noqa: BLE001 - degrade, never fail ingest
                logger.warning(
                    "AST chunking failed for %s (%s: %s); falling back to windows",
                    file_path,
                    type(exc).__name__,
                    exc,
                )
                self.stats.files_failed += 1
                chunks = self._window(repo_id, file_path, source, language, kind="window")

            # A parse that yields nothing useful (a file of only imports and
            # constants, say) still needs to be retrievable.
            if not chunks:
                self.stats.files_windowed += 1
                chunks = self._window(repo_id, file_path, source, language, kind="window")

        chunks = self._disambiguate_ids(chunks)
        self.stats.chunks_emitted += len(chunks)
        self.stats.by_language[language] = self.stats.by_language.get(language, 0) + len(chunks)
        return chunks

    @staticmethod
    def _disambiguate_ids(chunks: list[Chunk]) -> list[Chunk]:
        """
        Guarantee chunk_id uniqueness within a file.

        `chunk_id` is derived from the symbol path, which is stable across
        line drift -- the property incremental reindexing depends on. But a
        chunk with no resolvable symbol falls back to the bare file path, so
        every anonymous chunk in a file would otherwise share one id. With
        `ON CONFLICT DO UPDATE` on insert that is silent data loss: the last
        writer wins and the rest of the file vanishes from the index.

        Collisions are re-keyed with their ordinal among the colliding group.
        Order is deterministic (chunks are already line-sorted), so ids stay
        reproducible across runs, and a chunk that *does* have a symbol keeps
        its drift-stable id untouched.
        """
        seen: dict[str, int] = {}
        out: list[Chunk] = []
        for chunk in chunks:
            n = seen.get(chunk.chunk_id, 0)
            seen[chunk.chunk_id] = n + 1
            if n:
                chunk.chunk_id = _sha(f"{chunk.chunk_id}:dup{n}")[:24]
            out.append(chunk)
        return out

    # -- AST path ----------------------------------------------------------

    def _chunk_with_ast(
        self, repo_id: str, file_path: str, source: str, spec: LanguageSpec, language: str
    ) -> list[Chunk]:
        parser = self._parser(spec.grammar)
        data = source.encode("utf-8")
        tree = parser.parse(data)

        if tree.root_node.has_error:
            # A partial parse is still far better than line heuristics; we
            # count it so ingestion quality is observable rather than silent.
            self.stats.parse_errors += 1

        raw: list[Chunk] = []
        self._walk(tree.root_node, data, repo_id, file_path, spec, language, (), raw)
        return self._post_process(raw, data, repo_id, file_path, language)

    def _walk(
        self,
        node: Node,
        data: bytes,
        repo_id: str,
        file_path: str,
        spec: LanguageSpec,
        language: str,
        scope: tuple[str, ...],
        out: list[Chunk],
    ) -> None:
        for child in node.named_children:
            ntype = child.type

            # `export function foo()` / `export default class Bar {}`:
            # unwrap to the inner declaration but keep the wrapper's start
            # byte so the `export` keyword stays in the chunk.
            target, span_start = child, child.start_byte
            if ntype in spec.wrappers:
                inner = self._unwrap(child, spec)
                if inner is not None:
                    target, ntype = inner, inner.type

            if ntype in spec.definitions:
                # Decided on the unwrapped node: `@dec class Foo` is a
                # container, `@dec def foo` is not, even though both arrive
                # here as `decorated_definition`.
                if target.type in spec.containers:
                    self._emit_container(
                        child, target, span_start, data, repo_id, file_path, spec, language, scope, out
                    )
                else:
                    self.stats.definitions_found += 1
                    chunk = self._make_chunk(
                        data,
                        repo_id,
                        file_path,
                        language,
                        start_byte=self._absorb_comments(child, data, spec, span_start),
                        end_byte=child.end_byte,
                        kind="definition",
                        node_type=ntype,
                        symbol=self._name_of(target, data),
                        scope=scope,
                    )
                    if chunk:
                        out.append(chunk)
                # Do not descend into a non-container definition: its body is
                # already covered, and re-emitting inner closures duplicates
                # text across chunks.
                continue

            self._walk(child, data, repo_id, file_path, spec, language, scope, out)

    def _emit_container(
        self,
        node: Node,
        target: Node,
        span_start: int,
        data: bytes,
        repo_id: str,
        file_path: str,
        spec: LanguageSpec,
        language: str,
        scope: tuple[str, ...],
        out: list[Chunk],
    ) -> None:
        """
        Emit a header chunk for a class/impl/namespace, then index its members
        separately under an extended scope.

        The header spans from the container's first byte (decorators included)
        to the start of its first member -- capturing the signature, the
        docstring, and any field declarations, which is what "what is this
        class" queries actually need.
        """
        self.stats.containers_found += 1
        name = self._name_of(target, data)
        body = self._body_of(target)
        members = [
            c
            for c in (body.named_children if body is not None else [])
            if c.type in spec.definitions or c.type in spec.wrappers
        ]

        start = self._absorb_comments(node, data, spec, span_start)
        header_end = members[0].start_byte if members else node.end_byte

        header = self._make_chunk(
            data,
            repo_id,
            file_path,
            language,
            start_byte=start,
            end_byte=header_end,
            kind="container_header" if members else "definition",
            node_type=target.type,
            symbol=name,
            scope=scope,
        )
        if header:
            out.append(header)

        if body is not None and members:
            inner_scope = (*scope, name) if name else scope
            self._walk(body, data, repo_id, file_path, spec, language, inner_scope, out)

    # -- node helpers ------------------------------------------------------

    @staticmethod
    def _unwrap(node: Node, spec: LanguageSpec) -> Node | None:
        """Return the declaration inside a wrapper node, if there is one."""
        for child in node.named_children:
            if child.type in spec.definitions or child.type in spec.containers:
                return child
        return None

    @staticmethod
    def _name_of(node: Node, data: bytes) -> str | None:
        """Best-effort symbol name for a definition node."""
        named = node.child_by_field_name("name")
        if named is not None:
            return data[named.start_byte : named.end_byte].decode("utf-8", errors="replace")

        # Python: a decorated_definition carries its name on the inner node.
        for child in node.named_children:
            if child.type in {"function_definition", "class_definition"}:
                inner = child.child_by_field_name("name")
                if inner is not None:
                    return data[inner.start_byte : inner.end_byte].decode("utf-8", errors="replace")
            # JS/TS: `const handler = () => {}`
            if child.type == "variable_declarator":
                inner = child.child_by_field_name("name")
                if inner is not None:
                    return data[inner.start_byte : inner.end_byte].decode("utf-8", errors="replace")
        return None

    @staticmethod
    def _body_of(node: Node) -> Node | None:
        body = node.child_by_field_name("body")
        if body is not None:
            return body
        for child in node.named_children:
            if child.type in {"block", "class_body", "declaration_list", "field_declaration_list"}:
                return child
        return None

    @staticmethod
    def _absorb_comments(node: Node, data: bytes, spec: LanguageSpec, default: int) -> int:
        """
        Extend a chunk's start backwards over contiguous leading comments.

        A licence header 40 lines above is not documentation for this symbol,
        so only comments separated by at most one newline are absorbed.
        """
        start = min(default, node.start_byte)
        cursor = node.prev_named_sibling
        while cursor is not None and cursor.type in spec.comments:
            between = data[cursor.end_byte : start]
            if between.count(b"\n") > 1:
                break
            start = cursor.start_byte
            cursor = cursor.prev_named_sibling
        return start

    # -- chunk construction ------------------------------------------------

    def _make_chunk(
        self,
        data: bytes,
        repo_id: str,
        file_path: str,
        language: str,
        *,
        start_byte: int,
        end_byte: int,
        kind: str,
        node_type: str | None,
        symbol: str | None,
        scope: tuple[str, ...],
        part: int = 1,
        part_of: int = 1,
    ) -> Chunk | None:
        raw = data[start_byte:end_byte].decode("utf-8", errors="replace")
        content = raw.strip("\n")
        if not content.strip():
            return None

        leading_blank = len(raw) - len(raw.lstrip("\n"))
        start_line = data[:start_byte].count(b"\n") + 1 + leading_blank
        end_line = start_line + content.count("\n")

        symbol_path = f"{file_path}::{'.'.join([*scope, symbol])}" if symbol else file_path
        ident = f"{repo_id}:{symbol_path}:{part}" if part_of > 1 else f"{repo_id}:{symbol_path}"

        return Chunk(
            chunk_id=_sha(ident)[:24],
            content_sha=_sha(content),
            repo_id=repo_id,
            file_path=file_path,
            language=language,
            content=content,
            start_line=start_line,
            end_line=end_line,
            token_count=self._count(content),
            kind=kind,
            node_type=node_type,
            symbol=symbol,
            parent_scope=scope,
            part=part,
            part_of=part_of,
        )

    # -- post-processing ---------------------------------------------------

    def _post_process(
        self, chunks: list[Chunk], data: bytes, repo_id: str, file_path: str, language: str
    ) -> list[Chunk]:
        """Split oversized chunks, merge undersized neighbours, sort by line."""
        chunks.sort(key=lambda c: (c.start_line, c.end_line))

        sized: list[Chunk] = []
        for chunk in chunks:
            if chunk.token_count > self.max_tokens:
                parts = list(self._split(chunk))
                self.stats.oversized_split += 1
                sized.extend(parts)
            else:
                sized.append(chunk)

        return self._merge_tiny(sized)

    def _split(self, chunk: Chunk) -> Iterator[Chunk]:
        """
        Split an oversized definition into line groups under the budget.

        Each part is prefixed with the definition's signature lines so a
        fragment retrieved on its own still identifies what it belongs to.
        A body fragment with no signature is nearly useless at rerank time.
        """
        lines = chunk.content.split("\n")
        header_lines = lines[: min(SPLIT_HEADER_MAX_LINES, max(1, len(lines) // 8))]
        header = "\n".join(header_lines)
        header_tokens = self._count(header)
        budget = max(self.max_tokens - header_tokens, self.max_tokens // 2)

        groups: list[tuple[int, list[str]]] = []
        current: list[str] = []
        current_tokens = 0
        offset = 0

        for i, line in enumerate(lines):
            line_tokens = self._count(line) or 1
            if current and current_tokens + line_tokens > budget:
                groups.append((offset, current))
                current, current_tokens, offset = [], 0, i
            current.append(line)
            current_tokens += line_tokens
        if current:
            groups.append((offset, current))

        total = len(groups)
        for idx, (line_offset, group) in enumerate(groups, start=1):
            body = "\n".join(group)
            content = body if idx == 1 else f"{header}\n    ...\n{body}"
            start_line = chunk.start_line + line_offset
            yield Chunk(
                chunk_id=_sha(f"{chunk.repo_id}:{chunk.symbol_path}:{idx}")[:24],
                content_sha=_sha(content),
                repo_id=chunk.repo_id,
                file_path=chunk.file_path,
                language=chunk.language,
                content=content,
                start_line=start_line,
                end_line=start_line + len(group) - 1,
                token_count=self._count(content),
                kind=chunk.kind,
                node_type=chunk.node_type,
                symbol=chunk.symbol,
                parent_scope=chunk.parent_scope,
                part=idx,
                part_of=total,
            )

    def _merge_tiny(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Coalesce runs of undersized adjacent chunks sharing a parent scope.

        Files of one-line accessors would otherwise produce thousands of
        15-token chunks, which dilutes the index and wastes embedding budget.
        Merging is only done across chunks that are already neighbours in the
        file and in the same scope, so the result stays syntactically coherent.
        """
        if not chunks:
            return []

        merged: list[Chunk] = []
        buffer: list[Chunk] = []

        def flush() -> None:
            if not buffer:
                return
            if len(buffer) == 1:
                merged.append(buffer[0])
                buffer.clear()
                return
            first, last = buffer[0], buffer[-1]
            content = "\n\n".join(c.content for c in buffer)
            names = [c.symbol for c in buffer if c.symbol]
            self.stats.tiny_merged += len(buffer)
            merged.append(
                Chunk(
                    chunk_id=_sha(f"{first.repo_id}:{first.symbol_path}:merged:{len(buffer)}")[:24],
                    content_sha=_sha(content),
                    repo_id=first.repo_id,
                    file_path=first.file_path,
                    language=first.language,
                    content=content,
                    start_line=first.start_line,
                    end_line=last.end_line,
                    token_count=self._count(content),
                    kind=first.kind,
                    node_type=first.node_type,
                    symbol=names[0] if names else None,
                    parent_scope=first.parent_scope,
                )
            )
            buffer.clear()

        for chunk in chunks:
            if chunk.token_count >= self.min_tokens:
                flush()
                merged.append(chunk)
                continue
            if buffer and (
                buffer[-1].parent_scope != chunk.parent_scope
                or sum(c.token_count for c in buffer) + chunk.token_count > self.max_tokens
            ):
                flush()
            buffer.append(chunk)
        flush()
        return merged

    # -- window fallback ---------------------------------------------------

    def _window(
        self, repo_id: str, file_path: str, source: str, language: str, kind: str
    ) -> list[Chunk]:
        """
        Token-budgeted sliding window for files we do not parse.

        Windows are sized in tokens rather than lines so that a markdown table
        and a minified bundle both land inside the encoder's window.
        """
        lines = source.split("\n")
        chunks: list[Chunk] = []
        start = 0
        ordinal = 0

        while start < len(lines):
            tokens = 0
            end = start
            while end < len(lines) and tokens < self.window_tokens:
                tokens += self._count(lines[end]) or 1
                end += 1

            content = "\n".join(lines[start:end]).strip("\n")
            if content.strip():
                ordinal += 1
                ident = f"{repo_id}:{file_path}:w{ordinal}"
                chunks.append(
                    Chunk(
                        chunk_id=_sha(ident)[:24],
                        content_sha=_sha(content),
                        repo_id=repo_id,
                        file_path=file_path,
                        language=language,
                        content=content,
                        start_line=start + 1,
                        end_line=end,
                        token_count=self._count(content),
                        kind=kind,
                        node_type=None,
                        symbol=None,
                        parent_scope=(),
                    )
                )

            if end >= len(lines):
                break

            # Step back by roughly `window_overlap` tokens worth of lines so
            # a definition straddling a boundary appears whole in one window.
            back, budget = 0, self.window_overlap
            while back < (end - start - 1) and budget > 0:
                budget -= self._count(lines[end - 1 - back]) or 1
                back += 1
            start = max(end - back, start + 1)

        return chunks
