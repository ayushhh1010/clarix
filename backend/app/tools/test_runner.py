"""
Test Runner Tool — runs test suites in the cloned repository.
"""

import subprocess
import os
from pathlib import Path

from langchain_core.tools import tool, BaseTool


def create_test_runner_tool(repo_path: str) -> BaseTool:
    """Create a test runner tool bound to a specific repo path."""

    @tool
    def run_tests(test_command: str = "") -> str:
        """Run the test suite for the repository.
        If no command is specified, auto-detects the test framework.

        Args:
            test_command: Optional specific test command to run (e.g. "pytest tests/test_auth.py -v").
                          If empty, auto-detects based on project files.

        Returns:
            Test output (stdout + stderr), truncated if too long.
        """
        root = Path(repo_path)

        if test_command:
            cmd = test_command
        else:
            # Auto-detect test framework
            if (root / "pytest.ini").exists() or (root / "pyproject.toml").exists() or (root / "setup.py").exists():
                cmd = "python -m pytest --tb=short -q"
            elif (root / "package.json").exists():
                cmd = "npm test"
            elif (root / "Cargo.toml").exists():
                cmd = "cargo test"
            elif (root / "go.mod").exists():
                cmd = "go test ./..."
            elif (root / "pom.xml").exists():
                cmd = "mvn test"
            else:
                return "Could not auto-detect test framework. Please specify a test_command."

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(root),
                timeout=120,  # 2 minute timeout
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            output = ""
            if result.stdout:
                output += f"**STDOUT:**\n```\n{result.stdout}\n```\n"
            if result.stderr:
                output += f"**STDERR:**\n```\n{result.stderr}\n```\n"

            status = "✅ PASSED" if result.returncode == 0 else f"❌ FAILED (exit code {result.returncode})"
            output = f"**Status:** {status}\n**Command:** `{cmd}`\n\n{output}"

            # Truncate if too long
            if len(output) > 8000:
                output = output[:8000] + "\n\n... (output truncated)"

            return output

        except subprocess.TimeoutExpired:
            return f"⏰ Test execution timed out after 120 seconds.\nCommand: `{cmd}`"
        except Exception as exc:
            return f"Error running tests: {exc}"

    return run_tests
