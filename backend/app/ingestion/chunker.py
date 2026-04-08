"""
Code chunker — splits source files into semantically meaningful chunks.

Strategy:
1. For Python / JS / TS  → attempt function / class boundary detection via indentation
2. Fallback              → sliding window of N lines with M line overlap
3. Each chunk carries rich metadata for downstream retrieval.

Note: We use line-by-line parsing instead of regex to avoid catastrophic backtracking.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

from app.config import get_settings
from app.ingestion.parser import ParsedFile

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class CodeChunk:
    """A single chunk of code with metadata."""

    chunk_id: str
    repo_id: str
    file_path: str
    language: str
    content: str
    start_line: int
    end_line: int
    chunk_type: str  # "function", "class", "block"
    name: Optional[str] = None  # function/class name if applicable

    @property
    def metadata(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "repo_id": self.repo_id,
            "file_path": self.file_path,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type,
            "name": self.name or "",
        }


def _make_chunk_id(repo_id: str, file_path: str, start_line: int) -> str:
    raw = f"{repo_id}:{file_path}:{start_line}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _get_indent_level(line: str) -> int:
    """Return the number of leading spaces (tabs count as 4 spaces)."""
    count = 0
    for ch in line:
        if ch == " ":
            count += 1
        elif ch == "\t":
            count += 4
        else:
            break
    return count


def _is_definition_start(line: str, language: str) -> tuple[bool, str, Optional[str]]:
    """
    Check if line starts a function or class definition.
    Returns (is_definition, type, name).
    """
    stripped = line.lstrip()

    if language == "python":
        if stripped.startswith("class "):
            # Extract class name: "class Foo:" or "class Foo(Bar):"
            rest = stripped[6:].split("(")[0].split(":")[0].strip()
            return True, "class", rest if rest else None
        if stripped.startswith("def ") or stripped.startswith("async def "):
            # Extract function name
            if stripped.startswith("async def "):
                rest = stripped[10:]
            else:
                rest = stripped[4:]
            name = rest.split("(")[0].strip()
            return True, "function", name if name else None

    elif language in ("javascript", "typescript"):
        # function foo() or async function foo()
        if stripped.startswith("function ") or stripped.startswith("async function "):
            if stripped.startswith("async function "):
                rest = stripped[15:]
            else:
                rest = stripped[9:]
            name = rest.split("(")[0].strip()
            return True, "function", name if name else None
        # export function foo() or export async function foo()
        if stripped.startswith("export "):
            rest = stripped[7:].lstrip()
            if rest.startswith("function ") or rest.startswith("async function "):
                if rest.startswith("async function "):
                    rest = rest[15:]
                else:
                    rest = rest[9:]
                name = rest.split("(")[0].strip()
                return True, "function", name if name else None
            if rest.startswith("class "):
                name = rest[6:].split("{")[0].split("(")[0].split(" ")[0].strip()
                return True, "class", name if name else None
        # class Foo { or class Foo extends Bar {
        if stripped.startswith("class "):
            name = stripped[6:].split("{")[0].split("(")[0].split(" ")[0].strip()
            return True, "class", name if name else None

    return False, "", None


def _chunk_by_structure(parsed: ParsedFile, repo_id: str) -> list[CodeChunk]:
    """
    Split by function/class boundaries using line-by-line parsing.
    Much faster than regex for large files.
    """
    if parsed.language not in ("python", "javascript", "typescript"):
        return []

    lines = parsed.content.split("\n")
    chunks: list[CodeChunk] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        is_def, def_type, name = _is_definition_start(line, parsed.language)

        if is_def:
            start_line = i + 1  # 1-indexed
            base_indent = _get_indent_level(line)

            # Find the end of this definition
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                # Skip empty lines and comments
                stripped = next_line.strip()
                if (
                    not stripped
                    or stripped.startswith("#")
                    or stripped.startswith("//")
                ):
                    j += 1
                    continue

                next_indent = _get_indent_level(next_line)

                # For Python: end when we hit same or lower indent level with content
                if parsed.language == "python":
                    if next_indent <= base_indent:
                        break
                # For JS/TS: we need to track braces or use heuristics
                else:
                    # Simple heuristic: same indent with a new definition
                    if next_indent <= base_indent:
                        is_new_def, _, _ = _is_definition_start(
                            next_line, parsed.language
                        )
                        if is_new_def:
                            break
                j += 1

            end_line = j  # exclusive, but we want inclusive
            content = "\n".join(lines[i:j]).strip()

            if content:
                chunks.append(
                    CodeChunk(
                        chunk_id=_make_chunk_id(
                            repo_id, parsed.relative_path, start_line
                        ),
                        repo_id=repo_id,
                        file_path=parsed.relative_path,
                        language=parsed.language,
                        content=content,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type=def_type,
                        name=name,
                    )
                )

            i = j
        else:
            i += 1

    return chunks


def _chunk_by_sliding_window(parsed: ParsedFile, repo_id: str) -> list[CodeChunk]:
    """Sliding-window line-based chunking with overlap."""
    lines = parsed.content.split("\n")
    total = len(lines)
    size = settings.chunk_size_lines
    overlap = settings.chunk_overlap_lines
    step = max(size - overlap, 1)

    chunks: list[CodeChunk] = []
    for start in range(0, total, step):
        end = min(start + size, total)
        chunk_lines = lines[start:end]
        content = "\n".join(chunk_lines).strip()
        if not content:
            continue
        chunks.append(
            CodeChunk(
                chunk_id=_make_chunk_id(repo_id, parsed.relative_path, start + 1),
                repo_id=repo_id,
                file_path=parsed.relative_path,
                language=parsed.language,
                content=content,
                start_line=start + 1,
                end_line=end,
                chunk_type="block",
                name=None,
            )
        )
        if end >= total:
            break

    return chunks


def chunk_file(parsed: ParsedFile, repo_id: str) -> list[CodeChunk]:
    """
    Chunk a single parsed file.
    Uses structural chunking for supported languages, otherwise sliding window.
    """
    # Try structural chunking first
    structural = _chunk_by_structure(parsed, repo_id)
    if structural:
        logger.debug(
            "Structural chunking: %s → %d chunks", parsed.relative_path, len(structural)
        )
        return structural

    # Fallback to sliding window
    windowed = _chunk_by_sliding_window(parsed, repo_id)
    logger.debug("Window chunking: %s → %d chunks", parsed.relative_path, len(windowed))
    return windowed


def chunk_repository(parsed_files, repo_id: str) -> list[CodeChunk]:
    """Chunk all parsed files from a repository."""
    all_chunks: list[CodeChunk] = []
    for pf in parsed_files:
        all_chunks.extend(chunk_file(pf, repo_id))
    logger.info("Total chunks for repo %s: %d", repo_id, len(all_chunks))
    return all_chunks
