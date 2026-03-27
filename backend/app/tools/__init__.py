"""
Tools package — LangChain tools that agents can invoke to inspect and manipulate code.
"""

from typing import Sequence
from langchain_core.tools import BaseTool

from app.tools.code_search import create_code_search_tool
from app.tools.file_reader import create_file_reader_tool
from app.tools.file_tree import create_file_tree_tool
from app.tools.code_analysis import create_code_analysis_tool
from app.tools.test_runner import create_test_runner_tool
from app.tools.code_modifier import create_code_modifier_tool


def get_all_tools(repo_id: str, repo_path: str) -> list[BaseTool]:
    """Return all available tools bound to a specific repository."""
    return [
        create_code_search_tool(repo_id),
        create_file_reader_tool(repo_path),
        create_file_tree_tool(repo_path),
        create_code_analysis_tool(repo_path),
        create_test_runner_tool(repo_path),
        create_code_modifier_tool(repo_path),
    ]
