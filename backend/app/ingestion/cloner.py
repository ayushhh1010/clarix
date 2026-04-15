"""
Repository cloner — clones Git repos to local storage.
"""

import logging
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse


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


import re

# Patterns that are verbose git progress noise, not real errors
_NOISE_PATTERNS = re.compile(
    r"(Updating files:|POST git-upload-pack|Cloning into|remote:|Receiving objects:|"
    r"Resolving deltas:|Unpacking objects:|cmdline:)",
    re.IGNORECASE,
)


def _sanitize_clone_error(raw: str) -> str:
    """
    Strip verbose git progress output from a GitCommandError message,
    keeping only the meaningful error lines.
    """
    lines = raw.splitlines()
    useful: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip progress noise
        if _NOISE_PATTERNS.search(stripped):
            continue
        # Skip lines that are purely percentage progress like "78% (5034/6453)"
        if re.fullmatch(r"[\d]+%\s*\(\d+/\d+\)", stripped):
            continue
        useful.append(stripped)

    if not useful:
        return "Failed to clone repository. The repository may contain files with paths that are too long for this system."

    # Deduplicate while preserving order, cap length
    seen: set[str] = set()
    deduped: list[str] = []
    for u in useful:
        if u not in seen:
            seen.add(u)
            deduped.append(u)

    summary = "; ".join(deduped[:5])  # at most 5 unique error lines
    return f"Clone failed: {summary}"


def _inject_token(url: str) -> str:
    """Inject GitHub PAT into URL for authenticated cloning."""
    token = os.getenv("GITHUB_TOKEN", "")
    if token and "github.com" in url:
        url = url.replace(
            "https://github.com",
            f"https://{token}@github.com"
        )
    return url


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

    # Inject token for authenticated cloning in production
    auth_url = _inject_token(url)

    import git as _git  # deferred: git must be available when cloning
    from git import Repo, GitCommandError

    # Explicitly locate the git binary — Render and some containers have git
    # installed but not always on the PATH that Python's subprocess sees.
    git_bin = shutil.which("git") or shutil.which("git.exe")
    if git_bin:
        _git.refresh(git_bin)
    else:
        raise RuntimeError(
            "git executable not found. Install git or set GIT_PYTHON_GIT_EXECUTABLE."
        )

    logger.info("Cloning %s → %s", url, dest_str)  # log original URL, not token URL
    try:
        Repo.clone_from(
            auth_url,  # ← use authenticated URL
            dest_str,
            depth=1,
            single_branch=True,
            # Enable long paths on Windows to handle repos with deeply nested
            # file structures (e.g. Oppia) that exceed the 260-char limit.
            multi_options=["--config", "core.longpaths=true"],
            allow_unsafe_options=True,
        )
    except GitCommandError as exc:
        logger.error("Git clone failed: %s", exc)
        # Extract a clean, user-friendly error message instead of dumping
        # the full verbose git output (progress bars, file counts, etc.)
        clean_msg = _sanitize_clone_error(str(exc))
        raise RuntimeError(clean_msg) from exc

    logger.info("Successfully cloned %s (%s)", repo_name, dest_str)
    return repo_name, dest_str


def remove_repo(repo_id: str) -> None:
    """Delete a cloned repository from disk."""
    dest = settings.repos_path / repo_id
    if dest.exists():
        shutil.rmtree(str(dest), ignore_errors=True)
        logger.info("Removed repo directory %s", dest)
