"""
File Tree Tool — lists the directory structure of the repository.
"""

import os
from pathlib import Path

from langchain_core.tools import tool, BaseTool

# Directories to skip when listing
_SKIP = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".next", ".nuxt", "coverage", ".idea", ".vscode",
    "vendor", "target", "bin", "obj",
}


def create_file_tree_tool(repo_path: str) -> BaseTool:
    """Create a file tree listing tool bound to a specific repo path."""

    @tool
    def list_files(directory: str = "", max_depth: int = 3) -> str:
        """List files and directories in the repository in a tree format.

        Args:
            directory: Relative path to list (empty string = repo root).
            max_depth: Maximum depth to traverse (default 3).

        Returns:
            Formatted directory tree string.
        """
        base = Path(repo_path) / directory
        try:
            resolved = base.resolve()
            repo_resolved = Path(repo_path).resolve()
            if not str(resolved).startswith(str(repo_resolved)):
                return "Error: Path traversal detected — access denied."
        except Exception:
            return f"Error: Invalid path '{directory}'."

        if not resolved.is_dir():
            return f"Error: Directory not found — '{directory}'."

        lines: list[str] = [f"📁 {directory or '.'}"]
        _walk(resolved, "", 0, max_depth, lines)

        if len(lines) > 200:
            lines = lines[:200]
            lines.append(f"\n... (truncated, showing 200 of more entries)")

        return "\n".join(lines)

    return list_files


def _walk(path: Path, prefix: str, depth: int, max_depth: int, out: list[str]) -> None:
    """Recursively build tree lines."""
    if depth >= max_depth:
        return

    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return

    dirs = [e for e in entries if e.is_dir() and e.name not in _SKIP]
    files = [e for e in entries if e.is_file()]

    all_items = dirs + files
    for i, entry in enumerate(all_items):
        is_last = (i == len(all_items) - 1)
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "

        if entry.is_dir():
            out.append(f"{prefix}{connector}📁 {entry.name}/")
            _walk(entry, prefix + extension, depth + 1, max_depth, out)
        else:
            size_kb = entry.stat().st_size / 1024
            out.append(f"{prefix}{connector}📄 {entry.name} ({size_kb:.1f} KB)")
