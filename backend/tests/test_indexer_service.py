"""
Tests for the indexer service: the embedding endpoint and its worker thread.

The bug this file exists to prevent
-----------------------------------
`query_embedder.py` was written to call an embedding endpoint, and nothing
served one. Every unit test passed, because both sides were tested in
isolation: the client handled a well-formed response, and the endpoint did
not exist to be tested. The dense retrieval arm was simply off in any real
deployment, and silently -- the client degrades on purpose.

So the central test here is not that `/embed` returns 200. It is that the
real `HttpQueryEmbedder` can consume the real endpoint's real response and
produce the literals the SQL needs. That is the seam that was broken, and
only a test spanning it can catch it.

Most tests use a fake embedder: the contract is about shapes, literals and
concurrency, none of which need 306 MB of weights. The tests that do need
the real model are marked and skip without it.
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

from app.indexing.embedder import EMBED_DIM

httpx = pytest.importorskip("httpx")

from app.indexing.service import (  # noqa: E402
    MAX_CHARS_PER_TEXT,
    MAX_TEXTS_PER_REQUEST,
    SharedEmbedder,
    build_app,
)
from app.retrieval.query_embedder import normalise_endpoint  # noqa: E402

# --- endpoint normalisation ------------------------------------------------

@pytest.mark.parametrize(
    "configured,expected",
    [
        # A bare origin is what Render's RENDER_EXTERNAL_URL gives, and
        # `fromService` cannot append a path. Posting to it verbatim hits
        # `/`, gets 405, and the client degrades silently -- the dense arm
        # switches off with no error anywhere.
        ("https://clarix-indexer.onrender.com", "https://clarix-indexer.onrender.com/embed"),
        ("https://clarix-indexer.onrender.com/", "https://clarix-indexer.onrender.com/embed"),
        ("http://localhost:8081", "http://localhost:8081/embed"),
        # An explicit path is the operator's choice and must be preserved.
        ("http://localhost:8081/embed", "http://localhost:8081/embed"),
        ("https://host/prefix/embed", "https://host/prefix/embed"),
        ("https://host/custom-route", "https://host/custom-route"),
        ("https://host/embed/", "https://host/embed"),
        ("", ""),
    ],
)
def test_endpoint_normalisation(configured, expected):
    assert normalise_endpoint(configured) == expected


class FakeInner:
    """
    A deterministic stand-in for OnnxEmbedder.

    Vectors depend on the text so reordering bugs are detectable: if the
    wrapper returned results in the wrong order, the assertions comparing
    per-text vectors would fail rather than silently pass on identical rows.
    """

    model_id = "fake/model"

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.calls: list[list[str]] = []
        self.max_concurrent = 0
        self._active = 0
        self._guard = threading.Lock()

        class _Stats:
            batches = 0
            texts = 0
            seconds = 0.0

        self.stats = _Stats()

    def embed(self, texts, batch_size=16, sort_by_length=True):
        texts = list(texts)
        with self._guard:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
            self.calls.append(texts)
        try:
            if self.delay:
                time.sleep(self.delay)
            out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
            for i, text in enumerate(texts):
                out[i, 0] = float(len(text))
                out[i, 1] = float(sum(ord(c) for c in text) % 1000)
                out[i, 2] = -1.0  # a negative dim, so bits are not all ones
            return out
        finally:
            with self._guard:
                self._active -= 1


class Settings:
    """Minimal settings object; build_app only reads these."""

    app_env = "test"
    embedding_api_key = ""
    embedding_model_id = "fake/model"
    indexer_run_worker = False
    worker_dir = "./data/work"
    index_version = 1
    index_batch_size = 16


@pytest.fixture
def fake(monkeypatch):
    inner = FakeInner()
    monkeypatch.setattr(
        "app.indexing.service.OnnxEmbedder", lambda **_kw: inner
    )
    return inner


async def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://indexer")


# --- the contract that was broken -----------------------------------------

async def test_query_embedder_consumes_the_real_endpoint(fake):
    """
    The seam test: the shipped client against the shipped server.

    `HttpQueryEmbedder` is used exactly as `app.main` builds it, pointed at
    the real ASGI app. If the endpoint renamed a field, changed the literal
    format, or dropped `model_id`, this fails -- where testing either side
    alone would not.
    """
    from app.retrieval.query_embedder import HttpQueryEmbedder

    app = build_app(Settings())
    async with await _client(app) as http, app.router.lifespan_context(app):
        embedder = HttpQueryEmbedder(
            endpoint="http://indexer/embed",
            expected_model="fake/model",
            client=http,
        )
        vectors = await embedder.embed_query("how does retry work")

    assert vectors is not None, "the client refused a well-formed response"
    assert vectors.model_id == "fake/model"
    # bits must be a bit(768) literal: exactly EMBED_DIM characters of 0/1.
    assert len(vectors.bits) == EMBED_DIM
    assert set(vectors.bits) <= {"0", "1"}
    # vectors must be a halfvec literal parseable back to EMBED_DIM floats.
    assert vectors.vector.startswith("[") and vectors.vector.endswith("]")
    parsed = [float(x) for x in vectors.vector[1:-1].split(",")]
    assert len(parsed) == EMBED_DIM


async def test_model_id_mismatch_is_refused_end_to_end(fake):
    """
    A mismatched model must disable dense retrieval, not silently rank on
    incomparable vectors. Checked against the real endpoint, not a stub.
    """
    from app.retrieval.query_embedder import HttpQueryEmbedder

    app = build_app(Settings())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://indexer"
    ) as http:
        embedder = HttpQueryEmbedder(
            endpoint="http://indexer/embed",
            expected_model="a-completely-different-model",
            client=http,
        )
        async with app.router.lifespan_context(app):
            assert await embedder.embed_query("anything") is None


# --- endpoint behaviour ----------------------------------------------------

async def test_embed_returns_both_literal_forms(fake):
    app = build_app(Settings())
    async with await _client(app) as http, app.router.lifespan_context(app):
        r = await http.post("/embed", json={"texts": ["alpha", "beta beta"]})
    assert r.status_code == 200
    body = r.json()
    assert body["model_id"] == "fake/model"
    assert body["dim"] == EMBED_DIM
    assert len(body["vectors"]) == len(body["bits"]) == 2
    assert all(len(b) == EMBED_DIM for b in body["bits"])
    # Distinct inputs must give distinct vectors -- catches a wrapper that
    # returns the same row for every text.
    assert body["vectors"][0] != body["vectors"][1]


async def test_bits_match_the_vectors_sign_pattern(fake):
    """
    The bit literal must be the binary quantisation of the vector it ships
    with. A mismatch means the Hamming stage would shortlist candidates the
    rescore stage then scores on unrelated vectors.
    """
    app = build_app(Settings())
    async with await _client(app) as http, app.router.lifespan_context(app):
        r = await http.post("/embed", json={"texts": ["alpha"]})
    body = r.json()
    parsed = [float(x) for x in body["vectors"][0][1:-1].split(",")]
    bits = body["bits"][0]
    for i, value in enumerate(parsed):
        assert bits[i] == ("1" if value > 0 else "0"), f"dim {i} disagrees"


async def test_oversized_text_is_rejected(fake):
    app = build_app(Settings())
    async with await _client(app) as http, app.router.lifespan_context(app):
        r = await http.post("/embed", json={"texts": ["x" * (MAX_CHARS_PER_TEXT + 1)]})
    assert r.status_code == 413


async def test_too_many_texts_is_rejected(fake):
    app = build_app(Settings())
    async with await _client(app) as http, app.router.lifespan_context(app):
        r = await http.post(
            "/embed", json={"texts": ["x"] * (MAX_TEXTS_PER_REQUEST + 1)}
        )
    assert r.status_code == 422


async def test_empty_texts_is_rejected(fake):
    app = build_app(Settings())
    async with await _client(app) as http, app.router.lifespan_context(app):
        r = await http.post("/embed", json={"texts": []})
    assert r.status_code == 422


# --- auth ------------------------------------------------------------------

async def test_endpoint_requires_bearer_when_key_configured(fake):
    class Keyed(Settings):
        embedding_api_key = "s3cret-token"

    app = build_app(Keyed())
    async with await _client(app) as http, app.router.lifespan_context(app):
        assert (await http.post("/embed", json={"texts": ["a"]})).status_code == 401
        bad = await http.post(
            "/embed", json={"texts": ["a"]},
            headers={"Authorization": "Bearer wrong"},
        )
        assert bad.status_code == 401
        ok = await http.post(
            "/embed", json={"texts": ["a"]},
            headers={"Authorization": "Bearer s3cret-token"},
        )
        assert ok.status_code == 200


async def test_open_endpoint_when_no_key_configured(fake):
    app = build_app(Settings())
    async with await _client(app) as http, app.router.lifespan_context(app):
        assert (await http.post("/embed", json={"texts": ["a"]})).status_code == 200


def test_docs_are_disabled_in_production(fake):
    """
    The indexer serves one caller, the API, and never a person. Publishing
    the request schema of an authenticated endpoint buys nothing.
    """
    class Prod(Settings):
        app_env = "production"
        embedding_api_key = "a-key"

    prod = build_app(Prod())
    paths = {r.path for r in prod.routes if hasattr(r, "path")}
    assert "/docs" not in paths and "/redoc" not in paths
    assert "/openapi.json" not in paths
    # The endpoint itself is unaffected.
    assert "/embed" in paths and "/health" in paths

    # Still available outside production, where they are useful.
    dev = build_app(Settings())
    dev_paths = {r.path for r in dev.routes if hasattr(r, "path")}
    assert "/docs" in dev_paths


def test_production_without_a_key_refuses_to_start(fake):
    """
    A model endpoint open to the internet is free compute for whoever finds
    it. Failing at startup is the only failure mode that cannot be ignored.
    """
    class Prod(Settings):
        app_env = "production"
        embedding_api_key = ""

    with pytest.raises(RuntimeError, match="EMBEDDING_API_KEY"):
        build_app(Prod())


# --- health ----------------------------------------------------------------

async def test_health_reports_model_and_worker(fake):
    app = build_app(Settings())
    async with await _client(app) as http, app.router.lifespan_context(app):
        body = (await http.get("/health")).json()
    assert body["status"] == "healthy"
    assert body["model_id"] == "fake/model"
    assert body["dim"] == EMBED_DIM
    assert body["worker_running"] is False  # disabled in Settings


# --- SharedEmbedder --------------------------------------------------------

def test_shared_embedder_preserves_input_order():
    """
    Length-sorting reorders internally and must restore input order. A bug
    here misattributes every embedding to the wrong chunk -- catastrophic
    and invisible without this check.
    """
    inner = FakeInner()
    shared = SharedEmbedder(inner)
    texts = ["x" * n for n in (500, 1, 250, 2, 900, 3)]
    got = shared.embed(texts, batch_size=2)
    for i, text in enumerate(texts):
        assert got[i, 0] == float(len(text)), f"row {i} is not its own text"


def test_shared_embedder_matches_unsorted_output():
    inner = FakeInner()
    shared = SharedEmbedder(inner)
    texts = ["alpha", "beta beta", "c", "dddd"]
    sorted_out = shared.embed(texts, batch_size=2, sort_by_length=True)
    plain_out = shared.embed(texts, batch_size=2, sort_by_length=False)
    assert np.array_equal(sorted_out, plain_out)


def test_shared_embedder_handles_empty_input():
    shared = SharedEmbedder(FakeInner())
    assert shared.embed([]).shape == (0, EMBED_DIM)


def test_shared_embedder_serialises_concurrent_callers():
    """
    The reason the lock exists.

    The worker thread and the HTTP thread both reach the model. ONNX
    Runtime's thread-safety for concurrent Run() on one session is
    contested, so access is serialised; this asserts it actually is.
    """
    inner = FakeInner(delay=0.02)
    shared = SharedEmbedder(inner)
    errors: list[BaseException] = []

    def hammer():
        try:
            for _ in range(5):
                shared.embed(["a", "b", "c"], batch_size=2)
        except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, errors
    assert inner.max_concurrent == 1, (
        f"{inner.max_concurrent} concurrent calls reached the model; "
        "the lock is not holding"
    )


def test_shared_embedder_releases_the_lock_between_sub_batches():
    """
    Lock granularity is per sub-batch, not per call. Otherwise one indexing
    flush would block a query for the whole flush instead of one batch.
    """
    inner = FakeInner()
    shared = SharedEmbedder(inner)
    shared.embed([f"text-{i}" for i in range(64)], batch_size=16)
    assert len(inner.calls) == 4, "expected 4 sub-batch calls, not one big one"
    assert all(len(c) == 16 for c in inner.calls)


def test_shared_embedder_defaults_to_its_configured_batch_size():
    """
    The pipeline calls `embed(texts)` with no batch size, so the configured
    default is what the worker actually uses. Inheriting the library's 16
    would multiply the worst-case query wait by 16 (measured: median lock
    hold 0.25s at batch 1 against 4.84s at batch 16).
    """
    inner = FakeInner()
    shared = SharedEmbedder(inner, batch_size=1)
    shared.embed([f"text-{i}" for i in range(8)])
    assert len(inner.calls) == 8, "expected one call per text at batch 1"
    assert all(len(c) == 1 for c in inner.calls)


def test_configured_batch_size_is_overridable_per_call():
    inner = FakeInner()
    shared = SharedEmbedder(inner, batch_size=1)
    shared.embed([f"text-{i}" for i in range(8)], batch_size=4)
    assert len(inner.calls) == 2
    assert all(len(c) == 4 for c in inner.calls)


@pytest.mark.parametrize("configured,per_call", [(0, None), (-5, None), (1, 0), (1, -3)])
def test_batch_size_is_never_below_one(configured, per_call):
    """
    Zero would raise on `range(step=0)` and a negative would loop forever.
    Both the configured and the per-call path are clamped -- clamping only
    __init__ leaves the argument as a way in.
    """
    inner = FakeInner()
    shared = SharedEmbedder(inner, batch_size=configured)
    shared.embed(["a", "b"], batch_size=per_call)
    assert len(inner.calls) == 2
    assert all(len(c) == 1 for c in inner.calls)


async def test_embed_endpoint_does_not_block_the_event_loop(fake, monkeypatch):
    """
    Embedding runs in a thread. If it ran on the loop, a slow embed would
    stall every other request -- including /health, which is what a
    platform uses to decide the instance is dead.
    """
    slow = FakeInner(delay=0.3)
    monkeypatch.setattr("app.indexing.service.OnnxEmbedder", lambda **_kw: slow)
    app = build_app(Settings())
    async with await _client(app) as http, app.router.lifespan_context(app):
        embed_task = asyncio.create_task(
            http.post("/embed", json={"texts": ["a"]})
        )
        await asyncio.sleep(0.05)
        t0 = time.perf_counter()
        health = await http.get("/health")
        elapsed = time.perf_counter() - t0
        await embed_task

    assert health.status_code == 200
    assert elapsed < 0.25, (
        f"/health waited {elapsed:.2f}s behind an embed call; "
        "embedding is blocking the event loop"
    )


# --- worker thread ---------------------------------------------------------

async def test_worker_thread_starts_and_stops(fake, monkeypatch):
    """
    The worker must actually run, and must stop when the server does --
    otherwise a deploy leaves a thread holding a claimed job.
    """
    started = threading.Event()
    stopped = threading.Event()

    async def fake_run_worker(_factory, _config, _embedder, _chunker, *, stop=None,
                              max_jobs=None):
        started.set()
        await stop.wait()
        stopped.set()
        return 0

    monkeypatch.setattr("app.indexing.worker.run_worker", fake_run_worker)
    monkeypatch.setattr(
        "app.indexing.chunker.ASTChunker", lambda **_kw: object()
    )

    class WithWorker(Settings):
        indexer_run_worker = True

    app = build_app(WithWorker())
    async with app.router.lifespan_context(app):
        assert started.wait(timeout=10), "worker thread never started"
        async with await _client(app) as http:
            assert (await http.get("/health")).json()["worker_running"] is True

    assert stopped.wait(timeout=10), "worker thread did not stop on shutdown"
