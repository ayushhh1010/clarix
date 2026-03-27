"""
Code Modifier Tool — applies targeted text modifications to files in the repository.
"""

import os
from pathlib import Path

from langchain_core.tools import tool, BaseTool


def create_code_modifier_tool(repo_path: str) -> BaseTool:
    """Create a code modifier tool bound to a specific repo path."""

    @tool
    def modify_file(file_path: str, old_content: str, new_content: str) -> str:
        """Modify a file by replacing a specific block of text with new content.
        This is a surgical edit — only the specified old_content is replaced.

        Args:
            file_path: Relative path to the file within the repository.
            old_content: The exact text to find and replace (must match exactly).
            new_content: The replacement text.

        Returns:
            Success or error message with the modification details.
        """
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
            content = resolved.read_text(encoding="utf-8")
        except Exception as exc:
            return f"Error reading file: {exc}"

        # Verify old_content exists
        occurrences = content.count(old_content)
        if occurrences == 0:
            return (
                f"Error: The specified old_content was not found in '{file_path}'.\n"
                f"Make sure you are providing the exact text including whitespace."
            )
        if occurrences > 1:
            return (
                f"Warning: Found {occurrences} occurrences of old_content in '{file_path}'. "
                f"Please provide a more specific old_content string to avoid ambiguity. "
                f"Only the first occurrence will be replaced."
            )

        # Apply the modification
        new_file_content = content.replace(old_content, new_content, 1)

        try:
            resolved.write_text(new_file_content, encoding="utf-8")
        except Exception as exc:
            return f"Error writing file: {exc}"

        # Calculate diff summary
        old_lines = old_content.count("\n") + 1
        new_lines = new_content.count("\n") + 1

        return (
            f"✅ Successfully modified `{file_path}`.\n"
            f"- Replaced {old_lines} line(s) with {new_lines} line(s).\n"
            f"- Changes saved to disk."
        )

    return modify_file
