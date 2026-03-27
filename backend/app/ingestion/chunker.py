"""
Code chunker — splits source files into semantically meaningful chunks.

Strategy:
1. For Python / JS / TS  → attempt function / class boundary detection via regex
2. Fallback              → sliding window of N lines with M line overlap
3. Each chunk carries rich metadata for downstream retrieval.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
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


# ── Regex patterns for function / class detection ──────────────────

_PATTERNS: dict[str, list[re.Pattern]] = {
    "python": [
        re.compile(r"^(class\s+(\w+)[\s\S]*?)(?=\nclass\s|\ndef\s(?!\s)|\Z)", re.MULTILINE),
        re.compile(r"^((?:async\s+)?def\s+(\w+)[\s\S]*?)(?=\n(?:async\s+)?def\s|\nclass\s|\Z)", re.MULTILINE),
    ],
    "javascript": [
        re.compile(r"^((?:export\s+)?(?:async\s+)?function\s+(\w+)[\s\S]*?)(?=\n(?:export\s+)?(?:async\s+)?function\s|\Z)", re.MULTILINE),
        re.compile(r"^((?:export\s+)?class\s+(\w+)[\s\S]*?)(?=\nclass\s|\nfunction\s|\Z)", re.MULTILINE),
    ],
    "typescript": [
        re.compile(r"^((?:export\s+)?(?:async\s+)?function\s+(\w+)[\s\S]*?)(?=\n(?:export\s+)?(?:async\s+)?function\s|\Z)", re.MULTILINE),
        re.compile(r"^((?:export\s+)?class\s+(\w+)[\s\S]*?)(?=\nclass\s|\nfunction\s|\Z)", re.MULTILINE),
    ],
}


def _make_chunk_id(repo_id: str, file_path: str, start_line: int) -> str:
    raw = f"{repo_id}:{file_path}:{start_line}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _chunk_by_structure(parsed: ParsedFile, repo_id: str) -> list[CodeChunk]:
    """Attempt to split by function / class boundaries."""
    patterns = _PATTERNS.get(parsed.language, [])
    if not patterns:
        return []

    chunks: list[CodeChunk] = []
    found_spans: list[tuple[int, int]] = []

    for pattern in patterns:
        for match in pattern.finditer(parsed.content):
            body = match.group(1)
            name = match.group(2) if match.lastindex and match.lastindex >= 2 else None
            start_offset = match.start()
            start_line = parsed.content[:start_offset].count("\n") + 1
            end_line = start_line + body.count("\n")

            # Avoid overlapping matches
            overlap = False
            for s, e in found_spans:
                if not (end_line < s or start_line > e):
                    overlap = True
                    break
            if overlap:
                continue

            found_spans.append((start_line, end_line))
            chunk_type = "class" if (name and body.lstrip().startswith("class")) else "function"
            chunks.append(CodeChunk(
                chunk_id=_make_chunk_id(repo_id, parsed.relative_path, start_line),
                repo_id=repo_id,
                file_path=parsed.relative_path,
                language=parsed.language,
                content=body.strip(),
                start_line=start_line,
                end_line=end_line,
                chunk_type=chunk_type,
                name=name,
            ))

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
        chunks.append(CodeChunk(
            chunk_id=_make_chunk_id(repo_id, parsed.relative_path, start + 1),
            repo_id=repo_id,
            file_path=parsed.relative_path,
            language=parsed.language,
            content=content,
            start_line=start + 1,
            end_line=end,
            chunk_type="block",
            name=None,
        ))
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
        logger.debug("Structural chunking: %s → %d chunks", parsed.relative_path, len(structural))
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
