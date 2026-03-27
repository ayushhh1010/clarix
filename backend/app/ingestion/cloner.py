"""
Repository cloner — clones Git repos to local storage.
"""

import logging
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

from git import Repo, GitCommandError

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _extract_repo_name(url: str) -> str:
    """Extract a human-readable repo name from a Git URL."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    name = path.split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name or "unknown_repo"


def clone_repo(url: str, repo_id: str) -> tuple[str, str]:
    """
    Clone a Git repository to the local repos directory.

    Returns:
        (repo_name, local_path)
    """
    repo_name = _extract_repo_name(url)
    dest = settings.repos_path / repo_id
    dest_str = str(dest)

    if dest.exists():
        logger.warning("Destination %s already exists — removing", dest_str)
        shutil.rmtree(dest_str, ignore_errors=True)

    logger.info("Cloning %s → %s", url, dest_str)
    try:
        Repo.clone_from(
            url,
            dest_str,
            depth=1,  # shallow clone for speed
            single_branch=True,
        )
    except GitCommandError as exc:
        logger.error("Git clone failed: %s", exc)
        raise RuntimeError(f"Failed to clone repository: {exc}") from exc

    logger.info("Successfully cloned %s (%s)", repo_name, dest_str)
    return repo_name, dest_str


def remove_repo(repo_id: str) -> None:
    """Delete a cloned repository from disk."""
    dest = settings.repos_path / repo_id
    if dest.exists():
        shutil.rmtree(str(dest), ignore_errors=True)
        logger.info("Removed repo directory %s", dest)
