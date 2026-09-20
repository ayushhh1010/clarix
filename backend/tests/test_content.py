"""
Tests for serving file content without storing repositories.

The raw URL is derived from a user-supplied repository URL, so the host
allowlist applies here exactly as it does in the cloner: this is an
outbound request built from attacker-influenced input.
"""

from __future__ import annotations

import httpx
import pytest

from app.content import ContentError, build_tree, fetch_file, raw_url

GH = "https://github.com/pallets/flask.git"
SHA = "d73fa1c0f0e4c4e1d0e9b0a1b2c3d4e5f6070809"


# --- URL construction ------------------------------------------------------

def test_github_raw_url():
    assert raw_url(GH, SHA, "src/flask/app.py") == (
        f"https://raw.githubusercontent.com/pallets/flask/{SHA}/src/flask/app.py"
    )


def test_gitlab_and_bitbucket_use_their_own_shapes():
    assert "/-/raw/" in raw_url("https://gitlab.com/g/p.git", SHA, "a.py")
    assert "/raw/" in raw_url("https://bitbucket.org/t/r.git", SHA, "a.py")


def test_unknown_host_is_rejected_by_the_allowlist():
    with pytest.raises(ContentError):
        raw_url("https://evil.example.com/a/b.git", SHA, "x.py")


def test_unsafe_repo_url_is_rejected():
    for bad in ("ext::sh -c evil", "file:///etc/passwd", "https://localhost/x.git"):
        with pytest.raises(ContentError):
            raw_url(bad, SHA, "x.py")


def test_path_traversal_is_rejected():
    with pytest.raises(ContentError, match="invalid file path"):
        raw_url(GH, SHA, "../../../etc/passwd")


def test_path_segments_are_url_quoted():
    url = raw_url(GH, SHA, "src/a b/c#d.py")
    assert "%20" in url and "%23" in url
    assert url.count("/src/") == 1


def test_missing_ref_is_rejected():
    """Without a pinned commit we would serve whatever HEAD happens to be."""
    with pytest.raises(ContentError, match="no indexed commit"):
        raw_url(GH, "", "a.py")


def test_url_without_owner_and_repo_is_rejected():
    with pytest.raises(ContentError, match="owner/repo"):
        raw_url("https://github.com/onlyowner", SHA, "a.py")


# --- fetching --------------------------------------------------------------

def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler),
                             follow_redirects=False)


async def test_fetch_returns_decoded_text():
    async def handler(request):
        assert str(request.url).endswith("/src/flask/app.py")
        return httpx.Response(200, content=b"class Flask:\n    pass\n")

    got = await fetch_file(GH, SHA, "src/flask/app.py", client=_client(handler))
    assert got.text.startswith("class Flask")
    assert got.ref == SHA
    assert not got.truncated


async def test_fetch_truncates_very_large_files():
    from app.content import MAX_CONTENT_BYTES

    async def handler(request):
        return httpx.Response(200, content=b"x" * (MAX_CONTENT_BYTES + 5000))

    got = await fetch_file(GH, SHA, "big.py", client=_client(handler))
    assert got.truncated
    assert len(got.text) == MAX_CONTENT_BYTES


async def test_missing_file_names_the_commit():
    async def handler(request):
        return httpx.Response(404)

    with pytest.raises(ContentError, match="not present at commit"):
        await fetch_file(GH, SHA, "gone.py", client=_client(handler))


async def test_upstream_error_is_surfaced_not_swallowed():
    async def handler(request):
        return httpx.Response(503)

    with pytest.raises(ContentError, match="503"):
        await fetch_file(GH, SHA, "a.py", client=_client(handler))


async def test_network_failure_is_reported_clearly():
    async def handler(request):
        raise httpx.ConnectError("dns failure")

    with pytest.raises(ContentError, match="could not reach"):
        await fetch_file(GH, SHA, "a.py", client=_client(handler))


async def test_redirects_are_not_followed():
    """A redirect on a raw request could reach past the host allowlist."""
    async def handler(request):
        return httpx.Response(302, headers={"location": "https://evil.example.com/x"})

    with pytest.raises(ContentError, match="302"):
        await fetch_file(GH, SHA, "a.py", client=_client(handler))


async def test_invalid_utf8_is_replaced_not_fatal():
    async def handler(request):
        return httpx.Response(200, content=b"ok \xff\xfe bytes")

    got = await fetch_file(GH, SHA, "a.py", client=_client(handler))
    assert "ok" in got.text


# --- tree construction -----------------------------------------------------

def test_build_tree_nests_directories():
    tree = build_tree(["app/auth.py", "app/util/io.py", "README.md"])
    names = {n["name"]: n for n in tree}
    assert names["app"]["type"] == "directory"
    assert names["README.md"]["type"] == "file"

    app = names["app"]["children"]
    assert {c["name"] for c in app} == {"util", "auth.py"}
    util = next(c for c in app if c["name"] == "util")
    assert util["type"] == "directory"
    assert util["children"][0]["path"] == "app/util/io.py"


def test_build_tree_orders_directories_before_files():
    tree = build_tree(["z.py", "a/b.py"])
    assert [n["type"] for n in tree] == ["directory", "file"]


def test_build_tree_deduplicates_and_handles_edges():
    assert build_tree([]) == []
    assert build_tree(["", "/"]) == []
    tree = build_tree(["a/b.py", "a/b.py"])
    assert len(tree) == 1 and len(tree[0]["children"]) == 1


def test_build_tree_paths_are_full_paths():
    tree = build_tree(["src/deep/nested/file.py"])
    node = tree[0]
    while node["children"]:
        node = node["children"][0]
    assert node["path"] == "src/deep/nested/file.py"
