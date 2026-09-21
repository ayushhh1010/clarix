"""
Query embedding for the API process.

The indexer embeds chunks with a local ONNX model. The API must embed the
*query* with the same model -- vectors from different models are not
comparable, and mixing them degrades retrieval silently rather than loudly
-- but the API deliberately does not load onnxruntime or 320 MB of weights.

So the API calls the indexer's embedding endpoint over HTTP. One model, two
callers, no model in the request path.

Degradation is a first-class outcome. When no endpoint is configured or it
is unreachable, `embed_query` returns None and the caller drops the dense
arm. Lexical and symbol retrieval run entirely inside Postgres and keep
working: measured on the identifier dev split, symbol-only still reaches
MRR 0.948. A worse answer beats no answer -- provided the degradation is
visible, which is why the result says which arms actually ran.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from app.embedding_id import embedding_identity

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 8.0


@dataclass(frozen=True)
class QueryVectors:
    """A query embedding in the two forms the SQL needs."""

    bits: str      # bit(n) literal, for the Hamming ANN stage
    vector: str    # halfvec literal, for exact rescoring
    model_id: str


EMBED_PATH = "/embed"


def normalise_endpoint(endpoint: str) -> str:
    """
    Accept either a full endpoint URL or the indexer's base URL.

    Render injects a sibling service's address as `RENDER_EXTERNAL_URL`,
    which is an origin with no path -- `fromService` cannot append one. A
    bare origin posted to verbatim hits `/` and returns 405, and because
    this client degrades quietly on any non-2xx, the symptom would be the
    dense arm silently switching off. That is precisely the failure the
    embedding service was added to fix, so it is worth absorbing here.

    A URL that already names a path is left alone, so an endpoint served
    behind a prefix or on a different route still works.
    """
    from urllib.parse import urlparse

    cleaned = endpoint.strip().rstrip("/")
    if not cleaned:
        return cleaned
    path = urlparse(cleaned).path
    if path in ("", "/"):
        return cleaned + EMBED_PATH
    return cleaned


class QueryEmbedder:
    """Base: always unavailable. Used when nothing is configured."""

    available = False

    async def embed_query(self, text: str) -> QueryVectors | None:
        return None


class HttpQueryEmbedder(QueryEmbedder):
    """
    Calls the indexer's embedding endpoint.

    The endpoint returns the model id alongside the vector, and a mismatch
    against the index's model is refused rather than used: querying a
    halfvec index with vectors from a different encoder produces plausible
    nonsense, which is the worst kind of failure.
    """

    available = True

    def __init__(
        self,
        endpoint: str,
        expected_model: str,
        *,
        api_key: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ):
        self.endpoint = normalise_endpoint(endpoint)
        self.expected_model = expected_model
        self.api_key = api_key
        self.timeout = timeout
        self._client = client

    async def embed_query(self, text: str) -> QueryVectors | None:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        owns = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        try:
            response = await client.post(
                self.endpoint, json={"texts": [text]}, headers=headers
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "query embedding unavailable (%s); dense retrieval disabled "
                "for this request", exc,
            )
            return None
        finally:
            if owns:
                await client.aclose()

        if not 200 <= response.status_code < 300:
            logger.warning(
                "embedding endpoint returned %s; dense retrieval disabled "
                "for this request", response.status_code,
            )
            return None

        try:
            data = response.json()
            model_id = data["model_id"]
            bits = data["bits"][0]
            vector = data["vectors"][0]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning("malformed embedding response (%s); dense disabled", exc)
            return None

        if model_id != self.expected_model:
            # Refuse rather than degrade: mixing vector spaces produces
            # confident, wrong rankings with no error anywhere.
            logger.error(
                "embedding model mismatch: endpoint serves %r but the index "
                "was built with %r; dense retrieval disabled",
                model_id, self.expected_model,
            )
            return None

        return QueryVectors(
            bits=bits,
            vector=vector if isinstance(vector, str) else _to_literal(vector),
            model_id=model_id,
        )


def _to_literal(values) -> str:
    return "[" + ",".join(f"{float(v):.6f}" for v in values) + "]"


def build_query_embedder(settings) -> QueryEmbedder:
    endpoint = getattr(settings, "embedding_endpoint", "")
    if not endpoint:
        logger.info(
            "no embedding endpoint configured; retrieval will use the lexical "
            "and symbol arms only"
        )
        return QueryEmbedder()
    # The expected identity is composed the same way the indexer composes
    # it, so a disagreement about the ONNX variant or the truncation cap is
    # caught here rather than showing up as quietly worse rankings.
    return HttpQueryEmbedder(
        endpoint=endpoint,
        expected_model=embedding_identity(
            getattr(settings, "embedding_model_id",
                    "jinaai/jina-embeddings-v2-base-code"),
            getattr(settings, "embedding_onnx_file", "onnx/model_quantized.onnx"),
            getattr(settings, "embedding_max_tokens", 512),
        ),
        api_key=getattr(settings, "embedding_api_key", ""),
    )
