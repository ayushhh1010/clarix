"""
Code Analysis Tool — find function/class definitions by name in the repository.
"""

import re
from pathlib import Path

from langchain_core.tools import tool, BaseTool

# Patterns for common languages
_DEFINITION_PATTERNS: dict[str, list[re.Pattern]] = {
    ".py": [
        re.compile(r"^(\s*)((?:async\s+)?def\s+{name}\s*\(.*?\))", re.MULTILINE),
        re.compile(r"^(\s*)(class\s+{name}\s*[\(:])", re.MULTILINE),
    ],
    ".js": [
        re.compile(r"^(\s*)((?:export\s+)?(?:async\s+)?function\s+{name}\s*\()", re.MULTILINE),
        re.compile(r"^(\s*)((?:export\s+)?class\s+{name}\s*[\{{])", re.MULTILINE),
        re.compile(r"^(\s*)(const\s+{name}\s*=\s*(?:async\s+)?\()", re.MULTILINE),
    ],
    ".ts": [
        re.compile(r"^(\s*)((?:export\s+)?(?:async\s+)?function\s+{name}\s*[\(<])", re.MULTILINE),
        re.compile(r"^(\s*)((?:export\s+)?class\s+{name}\s*[\{<])", re.MULTILINE),
        re.compile(r"^(\s*)((?:export\s+)?(?:const|let)\s+{name}\s*[\=:])", re.MULTILINE),
    ],
    ".java": [
        re.compile(r"^(\s*)((?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)+{name}\s*\()", re.MULTILINE),
        re.compile(r"^(\s*)((?:public|private|protected)?\s*class\s+{name}\s*)", re.MULTILINE),
    ],
    ".go": [
        re.compile(r"^()(func\s+(?:\(\w+\s+\*?\w+\)\s+)?{name}\s*\()", re.MULTILINE),
        re.compile(r"^()(type\s+{name}\s+struct)", re.MULTILINE),
    ],
    ".rs": [
        re.compile(r"^(\s*)((?:pub\s+)?fn\s+{name}\s*[\(<])", re.MULTILINE),
        re.compile(r"^(\s*)((?:pub\s+)?struct\s+{name}\s*)", re.MULTILINE),
    ],
    ".cpp": [
        re.compile(r"^(\s*)((?:\w+\s+)+{name}\s*\()", re.MULTILINE),
        re.compile(r"^(\s*)(class\s+{name}\s*)", re.MULTILINE),
    ],
}

_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", ".next", "vendor", "target",
}


def create_code_analysis_tool(repo_path: str) -> BaseTool:
    """Create a code analysis tool bound to a specific repo path."""

    @tool
    def find_function_definition(name: str) -> str:
        """Find where a function, class, or symbol is defined in the codebase.

        Args:
            name: The exact name of the function, class, or variable to find.

        Returns:
            Locations and surrounding code where the symbol is defined.
        """
        if not name or not name.strip():
            return "Error: Please provide a symbol name to search for."

        name = name.strip()
        results: list[str] = []
        root = Path(repo_path)

        for file_path in root.rglob("*"):
            if any(skip in file_path.parts for skip in _SKIP_DIRS):
                continue
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()
            patterns_templates = _DEFINITION_PATTERNS.get(ext)
            if not patterns_templates:
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            for pattern_template in patterns_templates:
                # Build pattern with the actual name
                pattern = re.compile(
                    pattern_template.pattern.replace("{name}", re.escape(name)),
                    pattern_template.flags,
                )
                for match in pattern.finditer(content):
                    line_num = content[:match.start()].count("\n") + 1
                    lines = content.split("\n")
                    # Show context: 2 lines before, the match, 10 lines after
                    start = max(0, line_num - 3)
                    end = min(len(lines), line_num + 12)
                    snippet = "\n".join(
                        f"{i + 1:>5} | {lines[i]}" for i in range(start, end)
                    )
                    rel_path = str(file_path.relative_to(root)).replace("\\", "/")
                    results.append(
                        f"**{rel_path}** (line {line_num}):\n"
                        f"```{ext.lstrip('.')}\n{snippet}\n```\n"
                    )

        if not results:
            return f"No definition found for '{name}' in the codebase."

        header = f"Found {len(results)} definition(s) for `{name}`:\n\n"
        return header + "\n---\n".join(results[:10])  # cap at 10 results

    return find_function_definition
