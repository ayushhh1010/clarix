"""
Serve repository file content without storing repositories.

The v2 indexer clones to a temp directory and deletes it, so there is no
`local_path` to read from -- which is deliberate. v1 read files off the
worker's disk, and on an ephemeral filesystem that path was dangling after
every restart, so the file viewer returned "not found" for repositories the
API simultaneously reported as `ready`.

Content is fetched from the forge's raw endpoint, pinned to the exact commit
that was indexed. That has three properties worth having: no storage cost,
no drift between what was indexed and what is displayed, and no possibility
of serving a stale checkout.

SSRF: the URL is derived from a user-supplied repository URL, so the host is
validated against the same allowlist the cloner uses before any request is
made. `app/indexing/source.py` documents why that is an allowlist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import quote, urlparse

import httpx

logger = logging.getLogger(__name__)

FETCH_TIMEOUT_SECONDS = 10.0
MAX_CONTENT_BYTES = 2 * 1024 * 1024

# Raw-content endpoints per forge. Kept explicit rather than guessed: a
# wrong template silently 404s for a whole provider.
_RAW_TEMPLATES = {
    "github.com": "https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}",
    "www.github.com": "https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}",
    "gitlab.com": "https://gitlab.com/{owner}/{repo}/-/raw/{ref}/{path}",
    "www.gitlab.com": "https://gitlab.com/{owner}/{repo}/-/raw/{ref}/{path}",
    "bitbucket.org": "https://bitbucket.org/{owner}/{repo}/raw/{ref}/{path}",
    "codeberg.org": "https://codeberg.org/{owner}/{repo}/raw/commit/{ref}/{path}",
}


class ContentError(Exception):
    """Content could not be fetched, with a reason worth showing."""


@dataclass
class FileContent:
    path: str
    text: str
    truncated: bool
    ref: str
    source_url: str


def _split_repo_url(repo_url: str) -> tuple[str, str, str]:
    """Return (host, owner, repo) or raise."""
    from app.indexing.source import UnsafeSourceError, validate_url

    try:
        safe = validate_url(repo_url)
    except UnsafeSourceError as exc:
        raise ContentError(str(exc)) from exc

    parsed = urlparse(safe)
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2:
        raise ContentError(f"cannot derive owner/repo from {repo_url!r}")
    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    return parsed.hostname.lower(), owner, repo


def raw_url(repo_url: str, ref: str, file_path: str) -> str:
    """Build the forge raw-content URL for one file at one commit."""
    host, owner, repo = _split_repo_url(repo_url)
    template = _RAW_TEMPLATES.get(host)
    if template is None:
        raise ContentError(f"no raw-content endpoint known for {host!r}")
    if not ref:
        raise ContentError("no indexed commit to pin content to")

    # Path segments are quoted individually so slashes survive but any
    # other character that would change the URL's meaning does not.
    safe_path = "/".join(quote(seg, safe="") for seg in file_path.split("/") if seg)
    if not safe_path or ".." in file_path:
        raise ContentError(f"invalid file path {file_path!r}")
    return template.format(owner=quote(owner, safe=""), repo=quote(repo, safe=""),
                           ref=quote(ref, safe=""), path=safe_path)


async def fetch_file(
    repo_url: str, ref: str, file_path: str, *, client: httpx.AsyncClient | None = None
) -> FileContent:
    """
    Fetch one file at the indexed commit.

    Redirects are not followed: a forge that redirects a raw request is
    either wrong or being used to reach somewhere the allowlist would not
    have permitted.
    """
    url = raw_url(repo_url, ref, file_path)
    owns_client = client is None
    client = client or httpx.AsyncClient(
        timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=False
    )
    try:
        response = await client.get(url)
    except httpx.HTTPError as exc:
        raise ContentError(f"could not reach the repository host: {exc}") from exc
    finally:
        if owns_client:
            await client.aclose()

    if response.status_code == 404:
        raise ContentError(f"{file_path} is not present at commit {ref[:8]}")
    # Anything that is not a 2xx is an error, including 3xx. Redirects are
    # deliberately not followed -- a raw endpoint that redirects is either
    # wrong or is being used to reach a host the allowlist would refuse --
    # and treating one as success served an empty file as if it were real.
    if not 200 <= response.status_code < 300:
        raise ContentError(
            f"repository host returned {response.status_code} for {file_path}"
        )

    raw = response.content
    truncated = len(raw) > MAX_CONTENT_BYTES
    if truncated:
        raw = raw[:MAX_CONTENT_BYTES]
    return FileContent(
        path=file_path,
        text=raw.decode("utf-8", errors="replace"),
        truncated=truncated,
        ref=ref,
        source_url=url,
    )


def build_tree(paths: list[str]) -> list[dict]:
    """
    Turn a flat list of indexed file paths into a nested tree.

    Built from what is actually in the index rather than from a directory
    listing, so the viewer can only show files that are genuinely
    searchable -- no more clicking a file the index has never seen.
    """
    root: dict = {}
    for path in sorted(set(paths)):
        segments = [s for s in path.split("/") if s]
        if not segments:
            continue
        node = root
        for segment in segments[:-1]:
            entry = node.setdefault(segment, {"is_file": False, "children": {}})
            entry["is_file"] = False
            node = entry["children"]
        node.setdefault(segments[-1], {"is_file": True, "children": {}})

    def to_list(mapping: dict, prefix: str = "") -> list[dict]:
        # Directories first, then files, each alphabetically -- the ordering
        # every file tree uses, so the viewer needs no sorting of its own.
        ordered = sorted(mapping.items(), key=lambda kv: (kv[1]["is_file"], kv[0]))
        out = []
        for name, meta in ordered:
            full = f"{prefix}{name}"
            if meta["is_file"]:
                out.append({"name": name, "path": full, "type": "file", "children": None})
            else:
                out.append({
                    "name": name,
                    "path": full,
                    "type": "directory",
                    "children": to_list(meta["children"], f"{full}/"),
                })
        return out

    return to_list(root)
