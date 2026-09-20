"""
Hybrid retrieval: dense + lexical + symbol, fused with Reciprocal Rank Fusion.

Why three arms
--------------
Dense-only retrieval is what v1 did, and it is structurally weak at the most
common code question. "Where is `get_current_user` defined" is an exact
identifier lookup; embeddings are good at paraphrase and bad at precise
token identity, so the right chunk often ranks below semantically-similar
neighbours. Conversely, lexical search alone cannot answer "how does auth
work". The arms fail in different directions, which is the only reason
fusing them helps.

  dense    two-stage: bit Hamming ANN for candidates, halfvec rescore.
           Answers conceptual questions.
  lexical  weighted tsvector (symbol > path > body) with camelCase and
           snake_case expansion. Answers "mentions these words".
  symbol   exact match, then trigram similarity. Answers "where is X".

Fusion is weighted RRF rather than score normalisation: the arms produce
cosine distance, ts_rank_cd and trigram similarity, which are not on
comparable scales. RRF consumes only ranks, so it needs no calibration
between them.

The weights and the arm selection are NOT fixed -- they are chosen per query
by `route_query`, because measurement showed no single configuration wins.
Fusing all three helps on identifier lookups and actively hurts on semantic
questions; see the numbers in the Routing section below. The intuition in
the paragraph above is correct about *why* fusion can help, but it was wrong
to conclude that fusing always helps, and only the two evaluation sets
surfaced that.

Every result carries provenance: which arms found it and at what rank. That
is what makes the eval harness able to attribute a win or a regression to an
arm rather than to "retrieval".
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# RRF's smoothing constant. 60 is the value from the original Cormack et al.
# paper and is the conventional default; it damps the influence of the very
# top ranks so one arm cannot dominate the fusion on its own.
RRF_K = 60

# Candidates pulled from the binary index before halfvec rescoring. Over-
# fetching trades latency for recall; the operating point is chosen from
# bench/bench_quantization.py rather than guessed.
DEFAULT_ANN_CANDIDATES = 400

# Per-arm depth entering fusion.
DEFAULT_ARM_DEPTH = 50

# --- Routing -------------------------------------------------------------
#
# There is no single fusion that wins. Measured on the dev splits of two
# evaluation sets (bench/run_retrieval_eval.py, paired bootstrap,
# Holm-corrected):
#
#   docstring/semantic queries (n=519)
#     dense only        recall@10 0.908   MRR 0.737   <- best
#     symbol only       recall@10 0.285   MRR 0.234   (MRR -0.503, p<.0001)
#     every hybrid      significantly WORSE on MRR, even weighted 1/.1/.1
#
#   identifier-lookup queries (n=300)
#     dense only        recall@10 0.953   MRR 0.794
#     symbol only       recall@10 0.953   MRR 0.948
#     hybrid .5/.1/2    recall@10 0.990   MRR 0.960   <- best, dominates
#                                                        symbol-only on every
#                                                        metric
#
# The arms genuinely fail in opposite directions, so the configuration is
# chosen per query rather than fixed. Equal-weight RRF -- the obvious
# default -- is the worst option on one set and mid-table on the other, and
# would have shipped unnoticed without these two benchmarks.
ARM_WEIGHTS_SEMANTIC: dict[str, float] = {"dense": 1.0, "lexical": 0.0, "symbol": 0.0}
ARM_WEIGHTS_IDENTIFIER: dict[str, float] = {"dense": 0.5, "lexical": 0.1, "symbol": 2.0}

# Used when a caller pins arms explicitly and supplies no weights (ablation
# runs). Equal weights here means "do not silently apply a tuned operating
# point behind the caller's back"; the tuned points are the two ARM_WEIGHTS_*
# constants and are applied by the router.
DEFAULT_ARM_WEIGHTS: dict[str, float] = {"dense": 1.0, "lexical": 1.0, "symbol": 1.0}


@dataclass(frozen=True)
class RoutingDecision:
    """Which arms to run, how to weight them, and why."""

    use_dense: bool
    use_lexical: bool
    use_symbol: bool
    weights: dict[str, float]
    route: str

    def flags(self) -> dict[str, bool]:
        return {
            "use_dense": self.use_dense,
            "use_lexical": self.use_lexical,
            "use_symbol": self.use_symbol,
        }


_SEMANTIC_ROUTE = RoutingDecision(
    use_dense=True, use_lexical=False, use_symbol=False,
    weights=ARM_WEIGHTS_SEMANTIC, route="semantic",
)
_IDENTIFIER_ROUTE = RoutingDecision(
    use_dense=True, use_lexical=True, use_symbol=True,
    weights=ARM_WEIGHTS_IDENTIFIER, route="identifier",
)


def route_query(query: str) -> RoutingDecision:
    """
    Pick a retrieval configuration from the shape of the query.

    The signal is whether the query contains identifier-like tokens --
    underscores, dots, internal capitals, trailing `()`. That is the same
    test the symbol arm uses to build its term list, so a query that routes
    to the identifier configuration is exactly a query the symbol arm can
    act on.

    Deliberately a rule, not a classifier: it is inspectable, costs
    microseconds, has no training data to drift from, and the measured gap
    between the two routes is large enough that a marginal accuracy
    improvement would not pay for the opacity.
    """
    # An explicit intent phrase is sufficient on its own. Many real symbols
    # are ordinary words (`matches`, `runner`, `File`) that no shape test can
    # distinguish from prose; the phrasing is the evidence.
    if _LOOKUP_INTENT.search(query):
        return _IDENTIFIER_ROUTE

    # Otherwise an identifier-shaped token is required. An identifier alone
    # is NOT enough in a long query: documentation prose mentions identifiers
    # constantly ("Wraps http.cookiejar.CookieJar's remove_cookie_by_name"),
    # and routing those to the symbol-heavy configuration measured worse than
    # dense -- 59.2% of the docstring dev split misrouted that way, dragging
    # routed recall@10 to 0.788 against dense-only's 0.908.
    if extract_identifier_terms(query) and len(query.split()) <= BARE_LOOKUP_MAX_WORDS:
        return _IDENTIFIER_ROUTE
    return _SEMANTIC_ROUTE

# Vector width, kept in one place so the SQL casts cannot drift from the
# schema. Changing it requires a migration and an index rebuild.
EMBED_DIM = 768

# Bind parameters are cast text -> target type rather than bound directly.
# asyncpg infers the parameter type from the cast and, for `bit`, demands
# an asyncpg.BitString rather than a str -- binding a string raises
# DataError. Going via text keeps this layer free of driver-specific
# types and works identically under psycopg.
_BITS = f"CAST(CAST(:qbits AS text) AS bit({EMBED_DIM}))"
_VEC = f"CAST(CAST(:qvec AS text) AS halfvec({EMBED_DIM}))"


@dataclass
class RetrievedChunk:
    chunk_id: str
    file_path: str
    language: str
    symbol: str | None
    symbol_path: str
    kind: str
    start_line: int
    end_line: int
    token_count: int
    content: str

    # Provenance: arm -> (rank, raw score). Present only for arms that
    # returned this chunk.
    arms: dict[str, tuple[int, float]] = field(default_factory=dict)
    rrf_score: float = 0.0

    @property
    def citation(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "language": self.language,
            "symbol": self.symbol,
            "symbol_path": self.symbol_path,
            "kind": self.kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "token_count": self.token_count,
            "citation": self.citation,
            "rrf_score": round(self.rrf_score, 6),
            "arms": {k: {"rank": r, "score": round(s, 6)} for k, (r, s) in self.arms.items()},
        }


@dataclass
class RetrievalTrace:
    """Per-query timing and arm yield. Emitted to traces and to the eval set."""

    query: str
    repo_id: str
    route: str = "explicit"
    arm_counts: dict[str, int] = field(default_factory=dict)
    arm_ms: dict[str, float] = field(default_factory=dict)
    fused_count: int = 0
    total_ms: float = 0.0

    def log(self) -> None:
        logger.info(
            "retrieval repo=%s route=%s arms=%s ms=%s fused=%d total=%.1fms",
            self.repo_id[:8],
            self.route,
            self.arm_counts,
            {k: round(v, 1) for k, v in self.arm_ms.items()},
            self.fused_count,
            self.total_ms,
        )


_SELECT_COLS = """
    c.chunk_id, c.file_path, c.language, c.symbol, c.symbol_path, c.kind,
    c.start_line, c.end_line, c.token_count, c.content
"""

# Two-stage dense search. The inner query rides the HNSW index on the packed
# bit column; the outer one rescores that candidate set against halfvec.
# Rescoring is exact over a few hundred rows, which is why the halfvec
# column needs no index of its own -- the saving that keeps the table inside
# a 500 MB free tier.
# NOTE on S608 (suppressed for this module in pyproject.toml): the f-strings
# below interpolate only module-level constants -- _SELECT_COLS, _BITS, _VEC.
# Every value originating from a user (query text, repo id, depths) is passed
# as a bind parameter and never formatted into the SQL. A per-line ruff
# suppression comment is not usable inline here: the reported line is the
# opening `f"""`, so the comment would land inside the string literal and
# corrupt the query.
_DENSE_SQL = f"""
WITH candidates AS (
    SELECT c.chunk_id
    FROM chunks c
    WHERE c.repo_id = :repo_id
      AND c.embedding_bits IS NOT NULL
    ORDER BY c.embedding_bits <~> {_BITS}
    LIMIT :ann_candidates
)
SELECT {_SELECT_COLS}, (c.embedding <=> {_VEC}) AS score
FROM chunks c
JOIN candidates ON candidates.chunk_id = c.chunk_id
ORDER BY score ASC
LIMIT :depth
"""

# `to_tsquery` with explicit OR, not `websearch_to_tsquery`.
# websearch_to_tsquery ANDs every term, so "where is get_current_user
# defined" requires the words "where" and "defined" to appear in the source
# -- the arm then returns nothing for almost every natural-language query.
# OR-ing the terms and letting ts_rank_cd order them is what makes this a
# ranking arm rather than a filter.
_LEXICAL_SQL = f"""
SELECT {_SELECT_COLS},
       ts_rank_cd(c.search_vector, q.query) AS score
FROM chunks c, to_tsquery('simple', :tsquery) AS q(query)
WHERE c.repo_id = :repo_id
  AND c.search_vector @@ q.query
ORDER BY score DESC
LIMIT :depth
"""

# Function words carry no retrieval signal and the `simple` configuration
# does no stopword removal of its own. Left in, they match nearly every
# chunk and flatten ts_rank_cd.
_STOPWORDS = frozenset(["a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "doing", "have", "has", "had", "having", "i", "you", "he", "she", "it", "we", "they", "this", "that", "these", "those", "what", "which", "who", "whom", "whose", "where", "when", "why", "how", "in", "on", "at", "to", "from", "of", "for", "with", "about", "against", "between", "into", "during", "before", "after", "above", "below", "up", "down", "out", "off", "over", "under", "again", "then", "once", "here", "there", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "should", "now", "and", "or", "but", "if", "please", "show", "me", "tell", "explain", "does", "using", "use", "used"])


def build_lexical_tsquery(query: str) -> str:
    """
    Build an OR-ed tsquery from a natural-language query.

    Identifiers are expanded the same way the index expands them, so a query
    for `getCurrentUser` also matches rows indexed under get/current/user.
    Returns "" when nothing survives filtering, in which case the caller
    skips the arm rather than issuing a query that cannot match.
    """
    import re

    terms: set[str] = set()
    for raw in re.split(r"[^A-Za-z0-9_.]+", query):
        if not raw:
            continue
        token = raw.strip("._")
        if len(token) < 2 or token.lower() in _STOPWORDS:
            continue
        terms.add(token.lower())
        # Mirror the index-side camelCase and snake_case expansion.
        for part in re.split(r"[_.]+|(?<=[a-z0-9])(?=[A-Z])", token):
            if len(part) >= 2 and part.lower() not in _STOPWORDS:
                terms.add(part.lower())

    return " | ".join(sorted(terms))

# Exact identifier match first, then trigram similarity for typos and
# partial names. UNION rather than OR so the planner can use both the
# btree and the trigram index instead of falling back to a sequential scan.
_SYMBOL_SQL = f"""
WITH exact AS (
    SELECT {_SELECT_COLS}, 1.0::float8 AS score
    FROM chunks c
    WHERE c.repo_id = :repo_id AND c.symbol = ANY(:terms)
    LIMIT :depth
),
fuzzy AS (
    SELECT {_SELECT_COLS}, similarity(c.symbol, :q)::float8 AS score
    FROM chunks c
    WHERE c.repo_id = :repo_id
      AND c.symbol IS NOT NULL
      AND c.symbol % :q
      AND c.chunk_id NOT IN (SELECT chunk_id FROM exact)
    ORDER BY score DESC
    LIMIT :depth
)
SELECT * FROM exact
UNION ALL
SELECT * FROM fuzzy
ORDER BY score DESC
LIMIT :depth
"""


def _row_to_chunk(row) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row.chunk_id,
        file_path=row.file_path,
        language=row.language,
        symbol=row.symbol,
        symbol_path=row.symbol_path,
        kind=row.kind,
        start_line=row.start_line,
        end_line=row.end_line,
        token_count=row.token_count,
        content=row.content,
    )


# Markup that wraps identifiers in documentation prose: Sphinx roles
# (:func:`x`, :class:`~pkg.Y`), backticks, and reST/markdown emphasis.
# NB: does not strip underscores. An earlier version did, to handle reST
# emphasis, and turned `__get__` into `get` -- which then failed every
# identifier test. Dunder names are identifiers and must survive.
_MARKUP = re.compile(r"^:?\w*:?`+~?|`+$|^\*+|\*+$")


def looks_like_identifier(token: str) -> bool:
    """
    Whether a single token is plausibly a code identifier.

    Measured false positives this rules out (from 307 misrouted queries on
    the docstring dev split):

      acronyms      URL(25) HTTP(24) JSON(23) HTML(13) XML CLI WSGI -- these
                    are prose words in documentation, not identifiers, so an
                    uppercase letter alone is not evidence. Mixed case is:
                    a camelCase identifier has a lowercase letter *before* an
                    uppercase one.
      hyphenated    Content-Type(22) -- a hyphen is not legal in an
                    identifier in any language we index.
      markup        :class:`Response` -- stripped before testing.
    """
    # Markup is explicit evidence: documentation that writes :class:`Response`
    # or ``remove_cookie_by_name`` is naming code, whatever the token's shape.
    # Safe to accept because the router additionally requires an intent
    # phrase or a short query, so a long docstring full of :class: references
    # still routes semantic.
    marked = bool(_MARKUP.search(token))
    token = _MARKUP.sub("", token).strip("`'\"?.,:;!()")
    if len(token) < 3 or "-" in token:
        return False
    if marked:
        return True
    if token.endswith("()"):
        return True
    if "_" in token or "." in token:
        # Reject sentence-internal punctuation such as "e.g". Digit-only
        # segments are legal inside an identifier even though
        # `"200".isidentifier()` is False -- without this,
        # `test_digest_auth_with_200` was rejected.
        parts = [p for p in re.split(r"[_.]", token) if p]
        return bool(parts) and all(p.isidentifier() or p.isdigit() for p in parts)
    # camelCase: a lowercase letter followed somewhere later by an uppercase.
    lower = next((i for i, c in enumerate(token) if c.islower()), None)
    return lower is not None and any(c.isupper() for c in token[lower + 1:])


# Phrasings that state an explicit lookup intent.
_LOOKUP_INTENT = re.compile(
    r"\b(?:where\s+is|where\s+are|find|locate|show\s+me|"
    r"definition\s+of|defined|declaration|implementation|jump\s+to|go\s+to)\b",
    re.IGNORECASE,
)

# A query at or below this many words that contains an identifier is treated
# as a lookup even without an intent phrase -- someone typing a bare symbol
# name. Above it, prose dominates and an incidental identifier mention (a
# docstring saying "serializes as JSON") must not flip the route.
BARE_LOOKUP_MAX_WORDS = 6

# Words that carry lookup intent or are plain function words. Stripped when
# recovering a symbol name from an intent-bearing query.
_LOOKUP_STOPWORDS = frozenset(["where", "when", "what", "which", "who", "how", "why", "is", "are", "was", "the", "a", "an", "of", "for", "to", "in", "on", "at", "find", "locate", "show", "me", "tell", "please", "definition", "define", "defined", "declaration", "implementation", "implement", "implemented", "function", "method", "class", "code", "does", "do", "this", "that", "it", "its", "and", "or", "but", "from", "with", "by", "see", "used", "use", "using"])


def extract_identifier_terms(query: str, liberal: bool = False) -> list[str]:
    """
    Pull plausible identifier tokens out of a query.

    "where is get_current_user defined" -> ["get_current_user"]

    Conservative by default: feeding every English word to the symbol arm
    would make it fire on every query and contribute a near-random ranking
    to every fusion.

    `liberal` additionally accepts plain words when nothing matched by shape.
    Many real symbols are ordinary lowercase words -- `matches`, `runner`,
    `File`, `Limits` -- and are indistinguishable from prose in isolation.
    Only used once an explicit intent phrase ("where is X defined") has
    already established that the query IS a lookup, so the surrounding
    phrasing carries the evidence the token shape cannot.
    """
    tokens = query.replace("(", " (").replace(")", ") ").replace(",", " ").split()

    def clean(raw: str) -> str:
        return _MARKUP.sub("", raw).strip("`'\"?.,:;!").removesuffix("()")

    terms = [clean(raw) for raw in tokens if looks_like_identifier(raw)]
    if terms or not liberal:
        return terms

    return [
        c for c in (clean(raw) for raw in tokens)
        if len(c) >= 3 and c.lower() not in _LOOKUP_STOPWORDS
    ]


async def _run_arm(db: AsyncSession, sql: str, params: dict, name: str,
                   trace: RetrievalTrace) -> list[RetrievedChunk]:
    t0 = time.perf_counter()
    try:
        # Each arm runs inside its own SAVEPOINT. Without one, a failing
        # statement aborts the enclosing transaction and every *subsequent*
        # arm fails too with "current transaction is aborted" -- so the
        # degradation this except-block is meant to provide would not
        # actually happen. Found by test_a_failing_arm_degrades_instead_of_raising.
        #
        # SESSION LIFETIME CONSTRAINT. Savepoints are subtransactions, and
        # Postgres caches only PGPROC_MAX_CACHED_SUBXIDS (64 by default)
        # subxids per top-level transaction. Past that the cache is marked
        # overflowed and visibility checks fall back to pg_subtrans SLRU
        # lookups -- and the damage is CLUSTER-WIDE, not confined to the
        # offending session: any backend taking a snapshot while another has
        # overflowed marks its own snapshot overflowed too. Published
        # benchmarks report throughput collapsing from ~7,200 to ~160 TPS.
        #
        # Measured here: 150 queries on one never-committed session grew
        # latency 3.5x (45ms -> 160ms); committing or rolling back between
        # queries held it at 1.3x. Our own eval harness hit this and ran
        # 786x slower (13,514s vs 17.2s for 519 queries) before the fix.
        #
        # One session per request is fine -- that is the production shape --
        # but any caller looping many queries over a single session MUST
        # commit or roll back between them.
        async with db.begin_nested():
            result = await db.execute(text(sql), params)
            rows = result.fetchall()
    except Exception as exc:  # noqa: BLE001
        # One failing arm must not fail the query. Hybrid retrieval degrades
        # to whichever arms did return -- a worse answer beats no answer.
        logger.warning("retrieval arm %s failed: %s: %s", name, type(exc).__name__, exc)
        trace.arm_ms[name] = (time.perf_counter() - t0) * 1000
        trace.arm_counts[name] = -1
        return []

    chunks = []
    for rank, row in enumerate(rows, start=1):
        chunk = _row_to_chunk(row)
        chunk.arms[name] = (rank, float(row.score))
        chunks.append(chunk)

    trace.arm_ms[name] = (time.perf_counter() - t0) * 1000
    trace.arm_counts[name] = len(chunks)
    return chunks


def reciprocal_rank_fusion(
    arms: list[list[RetrievedChunk]],
    k: int = RRF_K,
    limit: int = 20,
    weights: dict[str, float] | None = None,
) -> list[RetrievedChunk]:
    """
    Fuse ranked lists by sum of w_arm / (k + rank).

    Rank-only, not score-normalised: the arms produce cosine distance,
    ts_rank_cd and trigram similarity, which are not on comparable scales.

    WHY WEIGHTS EXIST -- a measured failure of unweighted RRF
    ----------------------------------------------------------
    Unweighted RRF assumes the arms are of comparable quality. Measured on
    the dev split (519 queries, bench/run_retrieval_eval.py), they are not:

        dense    recall@10 0.908
        lexical  recall@10 0.339
        symbol   recall@10 0.285

    With equal weights a chunk ranked first by the lexical arm scores exactly
    as much as one ranked first by dense, so weak-arm results displace strong
    ones from the cut. Unweighted three-arm fusion measured *worse* than
    dense alone -- recall@10 0.813 vs 0.908.

    Weights per arm fix the mechanism. They are tuned on the dev split and
    reported once on test; `DEFAULT_ARM_WEIGHTS` records the tuned values and
    must not be changed without re-running the eval.
    """
    w = weights or DEFAULT_ARM_WEIGHTS
    merged: dict[str, RetrievedChunk] = {}

    for arm in arms:
        for chunk in arm:
            existing = merged.get(chunk.chunk_id)
            if existing is None:
                merged[chunk.chunk_id] = chunk
                existing = chunk
            else:
                existing.arms.update(chunk.arms)
            for name, (rank, _score) in chunk.arms.items():
                existing.rrf_score += w.get(name, 1.0) / (k + rank)

    ranked = sorted(merged.values(), key=lambda c: -c.rrf_score)
    return ranked[:limit]


async def hybrid_search(
    db: AsyncSession,
    repo_id: str,
    query: str,
    query_bits: str,
    query_vector: str,
    *,
    limit: int = 20,
    weights: dict[str, float] | None = None,
    arm_depth: int = DEFAULT_ARM_DEPTH,
    ann_candidates: int = DEFAULT_ANN_CANDIDATES,
    use_dense: bool | None = None,
    use_lexical: bool | None = None,
    use_symbol: bool | None = None,
) -> tuple[list[RetrievedChunk], RetrievalTrace]:
    """
    Run the enabled arms and fuse them.

    `query_bits` is a Postgres bit-string literal and `query_vector` a
    halfvec literal, both produced by the embedder. They are passed in rather
    than computed here so retrieval stays testable without loading a model,
    and so the API process never imports onnxruntime.

    The arm toggles exist for ablation: the eval harness measures each arm
    alone and in combination, which is the only way to claim the third arm
    earns its latency.
    """
    # Route unless the caller pinned the arms explicitly. Ablation runs pass
    # flags directly; production passes none and gets the routed behaviour.
    explicit = any(f is not None for f in (use_dense, use_lexical, use_symbol))
    if explicit:
        # An unspecified flag stays ENABLED. Treating None as False would
        # mean `use_lexical=False` silently disabled the symbol arm too --
        # a partial override quietly becoming a total one.
        decision = RoutingDecision(
            use_dense=True if use_dense is None else use_dense,
            use_lexical=True if use_lexical is None else use_lexical,
            use_symbol=True if use_symbol is None else use_symbol,
            weights=weights or DEFAULT_ARM_WEIGHTS, route="explicit",
        )
    else:
        decision = route_query(query)
        if weights:
            decision = RoutingDecision(
                decision.use_dense, decision.use_lexical, decision.use_symbol,
                weights, decision.route,
            )
    use_dense, use_lexical, use_symbol = (
        decision.use_dense, decision.use_lexical, decision.use_symbol
    )
    weights = decision.weights

    trace = RetrievalTrace(query=query, repo_id=repo_id, route=decision.route)
    t0 = time.perf_counter()

    arms: list[list[RetrievedChunk]] = []

    if use_dense:
        arms.append(
            await _run_arm(
                db, _DENSE_SQL,
                {
                    "repo_id": repo_id,
                    "qbits": query_bits,
                    "qvec": query_vector,
                    "ann_candidates": ann_candidates,
                    "depth": arm_depth,
                },
                "dense", trace,
            )
        )

    if use_lexical:
        tsquery = build_lexical_tsquery(query)
        if tsquery:
            arms.append(
                await _run_arm(
                    db, _LEXICAL_SQL,
                    {"repo_id": repo_id, "tsquery": tsquery, "depth": arm_depth},
                    "lexical", trace,
                )
            )
        else:
            trace.arm_counts["lexical"] = 0

    if use_symbol:
        terms = extract_identifier_terms(
            query, liberal=(decision.route == "identifier")
        )
        if terms:
            arms.append(
                await _run_arm(
                    db, _SYMBOL_SQL,
                    {
                        "repo_id": repo_id,
                        "q": " ".join(terms),
                        "terms": terms,
                        "depth": arm_depth,
                    },
                    "symbol", trace,
                )
            )
        else:
            trace.arm_counts["symbol"] = 0

    fused = reciprocal_rank_fusion(arms, limit=limit, weights=weights)
    trace.fused_count = len(fused)
    trace.total_ms = (time.perf_counter() - t0) * 1000
    trace.log()
    return fused, trace
