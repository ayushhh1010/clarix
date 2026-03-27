"""
File parser — walks a repository directory and reads supported source files.
"""

import logging
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".cpp", ".c", ".h", ".hpp",
    ".java",
    ".go",
    ".rs",
    ".json", ".yaml", ".yml",
    ".md", ".txt",
    ".html", ".css", ".scss",
    ".sh", ".bash",
    ".toml", ".cfg", ".ini",
    ".sql",
    ".proto",
    ".dockerfile",
    ".env",
}

SKIP_DIRS: set[str] = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".next", ".nuxt", "coverage", ".idea", ".vscode",
    "vendor", "target", "bin", "obj",
}

MAX_FILE_SIZE_BYTES = 512 * 1024  # 512 KB — skip very large files


class ParsedFile:
    """Represents a parsed source file."""

    __slots__ = ("path", "relative_path", "content", "language", "line_count")

    def __init__(self, path: Path, relative_path: str, content: str, language: str, line_count: int):
        self.path = path
        self.relative_path = relative_path
        self.content = content
        self.language = language
        self.line_count = line_count


_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".cpp": "cpp", ".c": "c", ".h": "c", ".hpp": "cpp",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".json": "json",
    ".yaml": "yaml", ".yml": "yaml",
    ".md": "markdown",
    ".html": "html",
    ".css": "css", ".scss": "scss",
    ".sql": "sql",
    ".sh": "shell", ".bash": "shell",
    ".toml": "toml",
    ".proto": "protobuf",
}


def _detect_language(ext: str) -> str:
    return _EXT_TO_LANGUAGE.get(ext, "text")


def parse_repository(repo_path: str) -> Generator[ParsedFile, None, None]:
    """
    Walk a repository directory and yield ParsedFile objects for every
    supported source file.
    """
    root = Path(repo_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    file_count = 0
    for file_path in root.rglob("*"):
        # Skip directories in the blocklist
        if any(skip in file_path.parts for skip in SKIP_DIRS):
            continue

        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
            logger.debug("Skipping large file: %s", file_path)
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Could not read %s: %s", file_path, exc)
            continue

        relative = str(file_path.relative_to(root)).replace("\\", "/")
        line_count = content.count("\n") + 1
        language = _detect_language(ext)

        file_count += 1
        yield ParsedFile(
            path=file_path,
            relative_path=relative,
            content=content,
            language=language,
            line_count=line_count,
        )

    logger.info("Parsed %d files from %s", file_count, repo_path)
