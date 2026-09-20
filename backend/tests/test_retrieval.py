"""
Hybrid retrieval tests.

The pure-function tests (RRF, identifier extraction) run anywhere. The arm
tests run the real SQL against real PostgreSQL + pgvector, because the whole
point of the schema work was that these queries use the indexes and return
what we think they return.
"""

from __future__ import annotations

import pytest

from app.retrieval.hybrid import (
    RetrievedChunk,
    extract_identifier_terms,
    hybrid_search,
    reciprocal_rank_fusion,
)

DIM = 768


def make(cid: str, arm: str, rank: int, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, file_path="a.py", language="python", symbol=cid,
        symbol_path=f"a.py::{cid}", kind="definition", start_line=1,
        end_line=2, token_count=10, content="x", arms={arm: (rank, score)},
    )


# --- fusion ----------------------------------------------------------------

def test_rrf_ranks_chunks_found_by_multiple_arms_higher():
    dense = [make("a", "dense", 1), make("b", "dense", 2)]
    lexical = [make("b", "lexical", 1), make("c", "lexical", 2)]

    fused = reciprocal_rank_fusion([dense, lexical])
    ids = [c.chunk_id for c in fused]

    # `b` is rank 2 in one arm and rank 1 in the other. `a` is rank 1 in one
    # arm and absent from the other. Agreement across arms should win.
    assert ids[0] == "b", ids
    assert set(ids) == {"a", "b", "c"}


def test_rrf_records_provenance_from_every_arm():
    fused = reciprocal_rank_fusion(
        [[make("a", "dense", 3)], [make("a", "lexical", 1)], [make("a", "symbol", 2)]]
    )
    assert len(fused) == 1
    assert set(fused[0].arms) == {"dense", "lexical", "symbol"}
    assert fused[0].arms["lexical"][0] == 1


def test_rrf_score_matches_the_formula():
    fused = reciprocal_rank_fusion(
        [[make("a", "dense", 1)], [make("a", "lexical", 4)]], k=60,
        weights={"dense": 1.0, "lexical": 1.0},
    )
    assert fused[0].rrf_score == pytest.approx(1 / 61 + 1 / 64)


def test_rrf_handles_empty_arms():
    assert reciprocal_rank_fusion([[], [], []]) == []
    assert len(reciprocal_rank_fusion([[make("a", "dense", 1)], []])) == 1


def test_rrf_respects_limit():
    arm = [make(str(i), "dense", i + 1) for i in range(50)]
    assert len(reciprocal_rank_fusion([arm], limit=7)) == 7


# --- identifier extraction -------------------------------------------------

@pytest.mark.parametrize(
    "query,expected",
    [
        ("where is get_current_user defined", ["get_current_user"]),
        ("what does getCurrentUser do?", ["getCurrentUser"]),
        ("explain app.security.decode_access_token", ["app.security.decode_access_token"]),
        ("how does chunk_file() work", ["chunk_file"]),
        ("how does authentication work in this app", []),
        ("what is the purpose of this repo", []),
    ],
)
def test_identifier_extraction(query, expected):
    assert extract_identifier_terms(query) == expected


def test_identifier_extraction_is_conservative():
    """
    Plain English must not reach the symbol arm. If it fires on every query
    it contributes a near-random ranking to every fusion.
    """
    for q in ["how do I run the tests", "describe the ingestion pipeline"]:
        assert extract_identifier_terms(q) == [], q


# --- arms against real Postgres -------------------------------------------

def _query_vectors(seed: int = 7) -> tuple[str, str]:
    """
    A deterministic non-zero query vector.

    Must not be all zeros: pgvector's `<=>` on a zero vector yields NaN
    (cosine is undefined without a magnitude), and NaN silently fails every
    ordering and comparison downstream. Real embeddings are L2-normalised so
    this cannot occur in production, but a test fixture can produce it.
    """
    import math

    vals = [math.sin(seed * (i + 1) * 0.01) for i in range(DIM)]
    norm = math.sqrt(sum(v * v for v in vals))
    vals = [v / norm for v in vals]
    bits = "".join("1" if v > 0 else "0" for v in vals)
    return bits, "[" + ",".join(f"{v:.6f}" for v in vals) + "]"


@pytest.mark.asyncio
async def test_all_three_arms_run_and_report_provenance(async_session, seeded_repo):
    bits, vec = _query_vectors()
    results, trace = await hybrid_search(
        async_session, seeded_repo,
        "where is get_current_user defined",
        query_bits=bits, query_vector=vec,
    )

    assert results, "no results from any arm"
    assert trace.arm_counts["dense"] > 0
    assert trace.arm_counts["lexical"] > 0
    assert trace.arm_counts["symbol"] > 0
    assert all(count >= 0 for count in trace.arm_counts.values()), "an arm errored"
    assert trace.total_ms > 0


@pytest.mark.asyncio
async def test_symbol_arm_puts_the_exact_match_first(async_session, seeded_repo):
    bits, vec = _query_vectors()
    results, _ = await hybrid_search(
        async_session, seeded_repo, "get_current_user",
        query_bits=bits, query_vector=vec, use_dense=False, use_lexical=False,
    )
    assert results[0].symbol == "get_current_user"
    assert results[0].arms["symbol"][1] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_lexical_arm_matches_camelcase_via_split(async_session, seeded_repo):
    """`getCurrentUserCamel` must be findable by searching 'current user'."""
    bits, vec = _query_vectors()
    results, _ = await hybrid_search(
        async_session, seeded_repo, "current user",
        query_bits=bits, query_vector=vec, use_dense=False, use_symbol=False,
    )
    symbols = [r.symbol for r in results]
    assert "getCurrentUserCamel" in symbols, symbols


@pytest.mark.asyncio
async def test_lexical_arm_finds_prose_chunks(async_session, seeded_repo):
    bits, vec = _query_vectors()
    results, _ = await hybrid_search(
        async_session, seeded_repo, "indexes a git repository",
        query_bits=bits, query_vector=vec, use_dense=False, use_symbol=False,
    )
    assert any(r.file_path == "README.md" for r in results)


@pytest.mark.asyncio
async def test_dense_arm_returns_results_with_a_distance(async_session, seeded_repo):
    bits, vec = _query_vectors()
    results, trace = await hybrid_search(
        async_session, seeded_repo, "authentication",
        query_bits=bits, query_vector=vec, use_lexical=False, use_symbol=False,
    )
    assert results
    assert all("dense" in r.arms for r in results)
    assert all(r.arms["dense"][1] >= 0 for r in results)


@pytest.mark.asyncio
async def test_results_are_scoped_to_the_repository(async_session, seeded_repo, conn):
    """Cross-repo leakage would be a data-isolation bug, not a ranking one."""
    other = "33333333-3333-3333-3333-333333333333"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO repositories (id, name, local_path, status) "
            "VALUES (%s, 'other', '/tmp/o', 'ready')", (other,),
        )
        cur.execute(
            """
            INSERT INTO chunks (chunk_id, repo_id, content_sha, file_path, language,
                                kind, symbol, symbol_path, start_line, end_line,
                                token_count, content)
            VALUES ('otherchunk000000000000', %s, 'deadbeef', 'x.py', 'python',
                    'definition', 'get_current_user', 'x.py::get_current_user',
                    1, 2, 5, 'def get_current_user(): pass')
            """,
            (other,),
        )

    bits, vec = _query_vectors()
    results, _ = await hybrid_search(
        async_session, seeded_repo, "get_current_user",
        query_bits=bits, query_vector=vec,
    )
    assert all(r.chunk_id != "otherchunk000000000000" for r in results)


@pytest.mark.asyncio
async def test_a_failing_arm_degrades_instead_of_raising(async_session, seeded_repo):
    """
    A malformed vector makes the dense arm fail. The query must still return
    lexical and symbol results rather than 500 -- a worse answer beats none.
    """
    results, trace = await hybrid_search(
        async_session, seeded_repo, "get_current_user",
        query_bits="not-a-bitstring", query_vector="not-a-vector",
    )
    assert trace.arm_counts["dense"] == -1, "expected the dense arm to fail"
    assert results, "query should still return results from the surviving arms"
    assert all("dense" not in r.arms for r in results)


@pytest.mark.asyncio
async def test_empty_repository_returns_nothing_without_erroring(async_session, conn):
    empty = "44444444-4444-4444-4444-444444444444"
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO repositories (id, name, local_path, status) "
            "VALUES (%s, 'empty', '/tmp/e', 'ready')", (empty,),
        )
    bits, vec = _query_vectors()
    results, trace = await hybrid_search(
        async_session, empty, "anything", query_bits=bits, query_vector=vec
    )
    assert results == []
    assert trace.fused_count == 0


@pytest.mark.asyncio
async def test_zero_query_vector_yields_nan_distances(async_session, seeded_repo):
    """
    Pins a hazard found while writing these tests: an all-zero query vector
    makes cosine distance undefined, and pgvector returns NaN rather than
    erroring. NaN compares false against everything, so results silently
    mis-order instead of failing loudly.

    Production embeddings are L2-normalised and cannot be zero, but anything
    constructing a query vector by hand must not produce one.
    """
    import math

    results, _ = await hybrid_search(
        async_session, seeded_repo, "anything",
        query_bits="0" * DIM, query_vector="[" + ",".join(["0.0"] * DIM) + "]",
        use_lexical=False, use_symbol=False,
    )
    assert results, "the ANN arm still returns rows"
    assert any(math.isnan(r.arms["dense"][1]) for r in results), (
        "expected NaN distances from a zero query vector"
    )


# --- routing ---------------------------------------------------------------

from app.retrieval.hybrid import (  # noqa: E402
    ARM_WEIGHTS_IDENTIFIER,
    ARM_WEIGHTS_SEMANTIC,
    route_query,
)


@pytest.mark.parametrize("query", [
    "where is get_current_user defined",
    "getCurrentUser",
    "chunk_file()",
    "app.security.decode_access_token",
])
def test_identifier_queries_route_to_the_symbol_heavy_config(query):
    d = route_query(query)
    assert d.route == "identifier"
    assert d.use_dense and d.use_lexical and d.use_symbol
    assert d.weights == ARM_WEIGHTS_IDENTIFIER


@pytest.mark.parametrize("query", [
    "how does authentication work",
    "explain the ingestion pipeline",
    "what is the purpose of this repository",
    "how do I run the tests",
])
def test_semantic_queries_route_to_dense_only(query):
    d = route_query(query)
    assert d.route == "semantic"
    assert d.use_dense
    assert not d.use_lexical and not d.use_symbol
    assert d.weights == ARM_WEIGHTS_SEMANTIC


def test_routing_weights_match_the_measured_optimum():
    """
    Pins the tuned operating points. Changing either of these invalidates
    every number in BENCHMARKS.md section 7 and requires re-running
    bench/run_retrieval_eval.py on both datasets.
    """
    assert ARM_WEIGHTS_SEMANTIC == {"dense": 1.0, "lexical": 0.0, "symbol": 0.0}
    assert ARM_WEIGHTS_IDENTIFIER == {"dense": 0.5, "lexical": 0.1, "symbol": 2.0}


@pytest.mark.asyncio
async def test_hybrid_search_routes_when_no_flags_are_given(async_session, seeded_repo):
    bits, vec = _query_vectors()
    _, semantic = await hybrid_search(
        async_session, seeded_repo, "how does authentication work",
        query_bits=bits, query_vector=vec,
    )
    assert semantic.route == "semantic"
    assert "lexical" not in semantic.arm_counts and "symbol" not in semantic.arm_counts

    _, ident = await hybrid_search(
        async_session, seeded_repo, "where is get_current_user defined",
        query_bits=bits, query_vector=vec,
    )
    assert ident.route == "identifier"
    assert {"dense", "lexical", "symbol"} <= set(ident.arm_counts)


@pytest.mark.asyncio
async def test_explicit_flags_override_the_router(async_session, seeded_repo):
    """Ablation runs must be able to pin a configuration."""
    bits, vec = _query_vectors()
    _, trace = await hybrid_search(
        async_session, seeded_repo, "where is get_current_user defined",
        query_bits=bits, query_vector=vec,
        use_dense=True, use_lexical=False, use_symbol=False,
    )
    assert trace.route == "explicit"
    assert "lexical" not in trace.arm_counts


def test_weighted_rrf_lets_a_strong_arm_outrank_a_weak_one():
    """
    The mechanism behind the measured failure of equal-weight fusion: a
    rank-1 hit from a weak arm tied with a rank-1 hit from a strong one.
    """
    strong = [make("good", "dense", 1)]
    weak = [make("bad", "symbol", 1)]

    equal = reciprocal_rank_fusion([strong, weak],
                                   weights={"dense": 1.0, "symbol": 1.0})
    assert equal[0].rrf_score == equal[1].rrf_score, "equal weights should tie"

    weighted = reciprocal_rank_fusion([strong, weak],
                                      weights={"dense": 1.0, "symbol": 0.1})
    assert weighted[0].chunk_id == "good"
    assert weighted[0].rrf_score > weighted[1].rrf_score


def test_zero_weight_arm_contributes_nothing():
    fused = reciprocal_rank_fusion(
        [[make("a", "dense", 1)], [make("b", "lexical", 1)]],
        weights={"dense": 1.0, "lexical": 0.0},
    )
    by_id = {c.chunk_id: c for c in fused}
    assert by_id["b"].rrf_score == 0.0
    assert by_id["a"].rrf_score > 0.0


# --- identifier detection: regressions found by the router eval ------------

from app.retrieval.hybrid import looks_like_identifier  # noqa: E402


@pytest.mark.parametrize("token", ["URL", "HTTP", "JSON", "HTML", "XML", "CLI", "WSGI"])
def test_acronyms_are_not_identifiers(token):
    """
    These are prose words in documentation. Treating an uppercase letter as
    evidence of an identifier misrouted 59.2% of the docstring dev split;
    URL/HTTP/JSON alone accounted for 72 of 307 misroutes.
    """
    assert not looks_like_identifier(token)


@pytest.mark.parametrize("token", ["Content-Type", "non-UTF-8", "well-known"])
def test_hyphenated_words_are_not_identifiers(token):
    assert not looks_like_identifier(token)


@pytest.mark.parametrize("token", [
    "get_current_user",
    "getCurrentUser",
    "chunk_file()",
    "app.security.decode_access_token",
    "__get__",                          # markup stripping once ate the dunders
    "test_digest_auth_with_200",        # "200".isidentifier() is False
    "test_HTTP_200_OK_GET_WITH_PARAMS",
])
def test_real_identifiers_are_detected(token):
    assert looks_like_identifier(token), token


@pytest.mark.parametrize("token", [":class:`Response`", "``remove_cookie_by_name``"])
def test_markup_is_stripped_before_testing(token):
    assert looks_like_identifier(token), token


def test_prose_mentioning_an_identifier_stays_semantic():
    """
    The failure that dragged routed recall@10 to 0.788: a long docstring
    sentence that happens to name a symbol is still a semantic query.
    """
    q = ("Deletes a cookie given a name. Wraps ``http.cookiejar.CookieJar``'s "
         "``remove_cookie_by_name``")
    assert route_query(q).route == "semantic"


@pytest.mark.parametrize("query", [
    "where is matches defined",
    "show me the File implementation",
    "find the definition of runner",
])
def test_intent_phrasing_routes_plain_word_symbols(query):
    """
    `matches`, `File` and `runner` are real symbols indistinguishable from
    prose by shape. The intent phrase is the evidence.
    """
    assert route_query(query).route == "identifier"


def test_liberal_extraction_recovers_a_plain_word_symbol():
    assert extract_identifier_terms("where is matches defined") == []
    assert extract_identifier_terms("where is matches defined", liberal=True) == ["matches"]


def test_liberal_extraction_prefers_shaped_identifiers_when_present():
    got = extract_identifier_terms("where is get_current_user defined", liberal=True)
    assert got == ["get_current_user"], got


# --- router generalisation (held-out phrasings) ----------------------------
#
# The symbol_v1 evaluation set is built from four templates, and the intent
# regex contains phrases from those same templates -- so its 99.0% routing
# rate on that set is CIRCULAR and must not be quoted as router accuracy.
#
# None of the phrasings below were used to write the regex. They are caught
# (or not) by the short-query + identifier-shape path, which is the part
# that has to generalise. n=15 is small; this is a smoke test for
# generalisation, not a measurement of it. Real user queries would replace it.

HELD_OUT_LOOKUPS = [
    "which file has get_current_user",
    "get_current_user function",
    "source of decode_access_token",
    "def of chunk_file",
    "the ChunkerStats class",
    "open hash_password",
    "file containing build_from_chunks",
    "what calls reciprocal_rank_fusion",
]

HELD_OUT_PROSE = [
    "how is authentication handled across the request lifecycle",
    "explain the way responses get serialised before they reach the client",
    "what happens when an upload exceeds the configured size limit",
    "describe the caching strategy used for repeated queries",
    "why would a request be retried more than once",
]


@pytest.mark.parametrize("query", HELD_OUT_LOOKUPS)
def test_router_generalises_to_unseen_lookup_phrasings(query):
    assert route_query(query).route == "identifier", query


@pytest.mark.parametrize("query", HELD_OUT_PROSE)
def test_router_does_not_fire_on_unseen_prose(query):
    assert route_query(query).route == "semantic", query
