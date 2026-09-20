"""
Tests for the evaluation-set builder.

`strip_docstring` is the leakage control for the whole retrieval benchmark.
If it silently fails to strip, the query text stays in the indexed chunk, the
lexical arm matches it verbatim, every configuration scores near 1.0, and the
benchmark can no longer tell any two systems apart -- while still producing
confident-looking numbers. That failure is invisible without these tests.
"""

from __future__ import annotations

import pytest

from app.evaluation.dataset import (
    EvalExample,
    _first_sentence,
    build_from_chunks,
    extract_docstring,
    is_usable_query,
    split,
    strip_docstring,
)


class FakeChunk:
    """Minimal stand-in for app.indexing.chunker.Chunk."""

    def __init__(self, content, language="python", symbol="f", kind="definition",
                 chunk_id="c1", file_path="a.py", token_count=40):
        self.content = content
        self.language = language
        self.symbol = symbol
        self.kind = kind
        self.chunk_id = chunk_id
        self.file_path = file_path
        self.token_count = token_count
        self.symbol_path = f"{file_path}::{symbol}"


PY_DOC = '''def get_current_user(token: str):
    """Resolve the authenticated user from a bearer token."""
    return decode(token)
'''

PY_DECORATED = '''@lru_cache()
def get_settings():
    """Load application settings from the environment."""
    return Settings()
'''

GO_DOC = """// ServeHTTP conforms to the http.Handler interface.
func (engine *Engine) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	engine.handleHTTPRequest(c)
}
"""

JS_DOC = """/**
 * Fetch every repository for the signed-in user.
 * @param token auth token
 */
export async function fetchRepos(token) {
  return fetch("/api/repo");
}
"""


# --- extraction ------------------------------------------------------------

def test_extracts_python_docstring():
    assert extract_docstring(PY_DOC, "python") == (
        "Resolve the authenticated user from a bearer token."
    )


def test_extracts_go_line_comment_doc():
    """Regression: Go contributed 0 of 1,185 chunks before this was handled."""
    assert extract_docstring(GO_DOC, "go") == (
        "ServeHTTP conforms to the http.Handler interface."
    )


def test_extracts_jsdoc_and_drops_tag_lines():
    got = extract_docstring(JS_DOC, "javascript")
    assert "Fetch every repository" in got
    assert "@param" not in got


def test_returns_empty_when_there_is_no_doc():
    assert extract_docstring("def f():\n    return 1\n", "python") == ""
    assert extract_docstring("func f() {}\n", "go") == ""


# --- stripping (the leakage control) --------------------------------------

def test_strips_python_docstring_but_keeps_the_signature():
    out = strip_docstring(PY_DOC, "python")
    assert "Resolve the authenticated user" not in out
    assert "def get_current_user" in out
    assert "return decode(token)" in out


def test_strips_docstring_from_a_decorated_function():
    out = strip_docstring(PY_DECORATED, "python")
    assert "Load application settings" not in out
    assert "@lru_cache()" in out
    assert "return Settings()" in out


def test_strips_go_doc_comment_but_keeps_the_body():
    out = strip_docstring(GO_DOC, "go")
    assert "conforms to the http.Handler" not in out
    assert "func (engine *Engine) ServeHTTP" in out


def test_strips_jsdoc_block():
    out = strip_docstring(JS_DOC, "javascript")
    assert "Fetch every repository" not in out
    assert "export async function fetchRepos" in out


def test_stripping_leaves_inline_body_comments_alone():
    """
    Only the leading doc block is leakage. Comments inside the body are
    signal that production retrieval also sees, so removing them would make
    the eval index differ from production more than necessary.
    """
    src = '''def f():
    """Doc."""
    # this explains the tricky bit
    return 1
'''
    out = strip_docstring(src, "python")
    assert "Doc." not in out
    assert "this explains the tricky bit" in out


def test_stripping_is_idempotent():
    once = strip_docstring(PY_DOC, "python")
    assert strip_docstring(once, "python") == once


def test_stripping_a_chunk_without_a_doc_changes_nothing():
    src = "def f():\n    return 1\n"
    assert strip_docstring(src, "python") == src


def test_query_text_is_absent_from_the_stripped_chunk():
    """The property the whole benchmark rests on, stated directly."""
    for src, lang in ((PY_DOC, "python"), (GO_DOC, "go"), (JS_DOC, "javascript")):
        query = _first_sentence(extract_docstring(src, lang))
        assert query, f"no query extracted for {lang}"
        stripped = strip_docstring(src, lang)
        assert query not in stripped, f"{lang}: query leaked into the indexed text"


# --- query quality filters -------------------------------------------------

@pytest.mark.parametrize("text,reason", [
    ("", "empty"),
    ("TODO: fix this later properly someday", "boilerplate"),
    ("Gets user.", "too_short"),
    (" ".join(["word"] * 60), "too_long"),
])
def test_rejects_unusable_queries(text, reason):
    usable, got = is_usable_query(text, "some_symbol")
    assert not usable
    assert got == reason


def test_rejects_a_docstring_that_only_restates_its_identifier():
    # Long enough to clear the length filters (they run first, being
    # cheaper), so this genuinely exercises the identifier check.
    usable, reason = is_usable_query("Gets and returns the current user.", "get_current_user")
    assert not usable
    assert reason == "restates_identifier"


def test_length_filters_run_before_the_identifier_check():
    """Filters are ordered cheapest-first; this pins that ordering."""
    usable, reason = is_usable_query("Get the user.", "get_user")
    assert not usable
    assert reason == "too_short"


def test_accepts_a_genuine_description():
    usable, reason = is_usable_query(
        "Resolve the authenticated principal from a bearer token.", "get_current_user"
    )
    assert usable, reason


def test_first_sentence_stops_before_parameter_blocks():
    doc = "Do the thing properly and well.\n\nArgs:\n    x: the input\n"
    assert _first_sentence(doc) == "Do the thing properly and well."


# --- building --------------------------------------------------------------

def test_build_emits_one_example_per_usable_chunk():
    chunks = [
        FakeChunk(PY_DOC, chunk_id="a", symbol="get_current_user"),
        FakeChunk(PY_DECORATED, chunk_id="b", symbol="get_settings"),
        FakeChunk("def undocumented():\n    return 1\n", chunk_id="c", symbol="undocumented"),
    ]
    got = list(build_from_chunks(chunks, "repo"))
    assert len(got) == 2
    assert {e.gold_chunk_ids[0] for e in got} == {"a", "b"}


def test_build_deduplicates_identical_docstrings():
    chunks = [
        FakeChunk(PY_DOC, chunk_id="a", symbol="one"),
        FakeChunk(PY_DOC, chunk_id="b", symbol="two"),
    ]
    assert len(list(build_from_chunks(chunks, "repo"))) == 1


def test_build_flags_queries_that_name_their_symbol():
    """Go doc convention leads with the identifier; those are easy wins."""
    go = FakeChunk(GO_DOC, language="go", symbol="ServeHTTP", chunk_id="g")
    py = FakeChunk(PY_DOC, symbol="get_current_user", chunk_id="p")
    by_id = {e.gold_chunk_ids[0]: e for e in build_from_chunks([go, py], "repo")}
    assert by_id["g"].meta["names_symbol"] is True
    assert by_id["p"].meta["names_symbol"] is False


def test_build_skips_non_definition_chunks():
    chunks = [FakeChunk(PY_DOC, kind="prose", chunk_id="a")]
    assert list(build_from_chunks(chunks, "repo")) == []


# --- splitting -------------------------------------------------------------

def _examples(n: int) -> list[EvalExample]:
    return [
        EvalExample(query_id=f"q{i}", query=f"query number {i}",
                    gold_chunk_ids=[f"c{i}"], gold_symbol_path=f"a.py::f{i}",
                    repo="r", language="python")
        for i in range(n)
    ]


def test_split_is_deterministic_and_disjoint():
    dev1, test1 = split(_examples(100))
    dev2, test2 = split(_examples(100))
    assert [e.query_id for e in dev1] == [e.query_id for e in dev2]
    assert [e.query_id for e in test1] == [e.query_id for e in test2]
    assert not ({e.query_id for e in dev1} & {e.query_id for e in test1})
    assert len(dev1) + len(test1) == 100


def test_split_respects_the_ratio():
    dev, test = split(_examples(100), train=0.7)
    assert len(dev) == 70
    assert len(test) == 30


def test_split_does_not_depend_on_input_order():
    """Otherwise a reordered corpus silently reshuffles dev/test."""
    forward = _examples(50)
    backward = list(reversed(_examples(50)))
    assert [e.query_id for e in split(forward)[1]] == [
        e.query_id for e in split(backward)[1]
    ]
