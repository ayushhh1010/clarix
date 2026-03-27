"""
File Reader Tool — reads the content of a specific file from the repository.
"""

import os
from pathlib import Path

from langchain_core.tools import tool, BaseTool


def create_file_reader_tool(repo_path: str) -> BaseTool:
    """Create a file reader tool bound to a specific repo path."""

    @tool
    def get_file(file_path: str, start_line: int = 0, end_line: int = 0) -> str:
        """Read the contents of a file from the repository.
        Can optionally read a specific range of lines.

        Args:
            file_path: Relative path to the file within the repository (e.g. "src/auth/login.py").
            start_line: Optional start line (1-indexed). 0 means read from beginning.
            end_line: Optional end line (1-indexed, inclusive). 0 means read to end.

        Returns:
            The file contents with line numbers, or an error message.
        """
        # Resolve and validate path (prevent directory traversal)
        full_path = Path(repo_path) / file_path
        try:
            resolved = full_path.resolve()
            repo_resolved = Path(repo_path).resolve()
            if not str(resolved).startswith(str(repo_resolved)):
                return "Error: Path traversal detected — access denied."
        except Exception:
            return f"Error: Invalid path '{file_path}'."

        if not resolved.is_file():
            return f"Error: File not found — '{file_path}'."

        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"Error reading file: {exc}"

        lines = content.split("\n")

        # Apply line range
        if start_line > 0 or end_line > 0:
            s = max(start_line - 1, 0)
            e = end_line if end_line > 0 else len(lines)
            lines = lines[s:e]
            offset = s
        else:
            offset = 0

        # Format with line numbers
        numbered = []
        for i, line in enumerate(lines, start=offset + 1):
            numbered.append(f"{i:>5} | {line}")

        file_ext = resolved.suffix.lstrip(".")
        header = f"**File: {file_path}** ({len(numbered)} lines shown)\n"
        return header + f"```{file_ext}\n" + "\n".join(numbered) + "\n```"

    return get_file
