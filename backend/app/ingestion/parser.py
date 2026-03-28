"""
File parser — walks a repository directory and reads supported source files.
Optimized for free-tier deployment — skips low-value files to reduce
embedding time and memory usage.
"""

import logging
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

# ── Only include files valuable for code understanding ────────
SUPPORTED_EXTENSIONS: set[str] = {
    # Core source code — highest value
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".cpp", ".c", ".h", ".hpp",
    ".java", ".go", ".rs",
    # Web
    ".html", ".css", ".scss",
    # Data / config that affects behavior
    ".sql", ".proto",
    # Shell scripts
    ".sh", ".bash",
}

# ── Skip these file names entirely ───────────────────────────
SKIP_FILENAMES: set[str] = {
    "package-lock.json", "yarn.lock", "poetry.lock",
    "Pipfile.lock", "pnpm-lock.yaml", "composer.lock",
    "Gemfile.lock", ".env", ".env.example", ".env.local",
    ".gitignore", ".gitattributes", ".prettierrc",
    ".eslintrc", ".editorconfig", "LICENSE",
    "CHANGELOG.md", "CONTRIBUTING.md",
}

# ── Skip these file name patterns ────────────────────────────
SKIP_FILENAME_PATTERNS: set[str] = {
    ".min.js", ".min.css", ".bundle.js",
    ".chunk.js", ".map", ".d.ts",
}

SKIP_DIRS: set[str] = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".next", ".nuxt", "coverage", ".idea", ".vscode",
    "vendor", "target", "bin", "obj", "out",
    "public", "static", "assets", "images", "fonts",
    "migrations", ".github",
}

MAX_FILE_SIZE_BYTES = 100 * 1024  # 100 KB
MIN_LINE_COUNT = 5


class ParsedFile:
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
    ".html": "html",
    ".css": "css", ".scss": "scss",
    ".sql": "sql",
    ".sh": "shell", ".bash": "shell",
    ".proto": "protobuf",
}


def _detect_language(ext: str) -> str:
    return _EXT_TO_LANGUAGE.get(ext, "text")


def _should_skip_file(file_path: Path) -> bool:
    name = file_path.name
    if name in SKIP_FILENAMES:
        return True
    if any(name.endswith(p) for p in SKIP_FILENAME_PATTERNS):
        return True
    return False


def parse_repository(repo_path: str) -> Generator[ParsedFile, None, None]:
    root = Path(repo_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    file_count = 0
    skipped_count = 0

    for file_path in root.rglob("*"):
        if any(skip in file_path.parts for skip in SKIP_DIRS):
            continue
        if not file_path.is_file():
            continue
        if _should_skip_file(file_path):
            skipped_count += 1
            continue

        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            skipped_count += 1
            continue
        if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
            skipped_count += 1
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Could not read %s: %s", file_path, exc)
            continue

        line_count = content.count("\n") + 1
        if line_count < MIN_LINE_COUNT:
            skipped_count += 1
            continue

        relative = str(file_path.relative_to(root)).replace("\\", "/")
        language = _detect_language(ext)
        file_count += 1

        yield ParsedFile(
            path=file_path,
            relative_path=relative,
            content=content,
            language=language,
            line_count=line_count,
        )

    logger.info("Parsed %d files from %s (%d skipped)", file_count, repo_path, skipped_count)