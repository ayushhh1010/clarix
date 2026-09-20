"""Hybrid retrieval: dense + lexical + symbol arms, fused with RRF."""

from app.retrieval.hybrid import (
    RetrievalTrace,
    RetrievedChunk,
    RoutingDecision,
    build_lexical_tsquery,
    extract_identifier_terms,
    hybrid_search,
    reciprocal_rank_fusion,
    route_query,
)

__all__ = [
    "RoutingDecision",
    "route_query",
    "build_lexical_tsquery",
    "RetrievalTrace",
    "RetrievedChunk",
    "extract_identifier_terms",
    "hybrid_search",
    "reciprocal_rank_fusion",
]
