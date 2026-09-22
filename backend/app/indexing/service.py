"""
The indexer process: embedding endpoint plus indexing worker, together.

Why this exists
---------------
`app/retrieval/query_embedder.py` embeds the query by POSTing to an
embedding endpoint, because the API container must not load a model.
Nothing served that endpoint, so `embedding_endpoint` was always empty,
`embed_query` always returned None, and the dense arm was disabled in any
real deployment -- which made the entire two-stage vector search
unreachable. This is the missing half.

Why one process and not two
---------------------------
Measured (bench/bench_colocated_rss.py, fp16, arena off):

    app.main (API) alone                 59.1 MB
    model loaded and serving              439 MB peak
    both in one process                   485 MB against a 512 MB cap

Colocating the API with the model leaves 27 MB of headroom, which is not
enough to serve requests under load. Splitting them and colocating the
*embedder* with the *worker* instead costs nothing extra: the worker
already holds the model, so serving queries from it is free. That is the
whole argument for this file's shape.

Concurrency
-----------
Two things want the model: HTTP query embedding (latency-sensitive, one
short text) and the indexing worker (throughput-sensitive, large batches).

The worker cannot share the server's event loop. `index_repository` calls
`embedder.embed()` synchronously (pipeline.py), which would block the loop
for the length of a batch -- the endpoint would stop answering for seconds
at a time while a repository indexed. So the worker runs on its own thread
with its own loop.

That makes the model genuinely shared across threads, and ONNX Runtime's
own documentation and issue tracker disagree about whether concurrent
`Run()` on one session is safe. `SharedEmbedder` serialises instead, which
is correct under either reading, and it takes the lock per *sub-batch* so a
query waits at most one batch rather than a whole flush.

The batch size is therefore a latency control, not a throughput knob.
Measured on real chunks (bench/bench_embed_batch.py), throughput is flat
across a 16x range while the lock hold scales linearly:

    batch  1    2.43 chunks/s    median hold 0.25 s    p95  1.15 s
    batch 16    2.32 chunks/s    median hold 4.97 s    p95 16.74 s

So `INDEXER_EMBED_BATCH` defaults to 1. Batching exists to amortise
padding, and a batch of one is never padded.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.indexing.embedder import (
    EMBED_DIM,
    MAX_SEQUENCE_TOKENS,
    ONNX_FILE,
    OnnxEmbedder,
    binary_quantize,
    bits_to_sql,
    vector_to_sql,
)

logger = logging.getLogger(__name__)

# A query is one short string. These bound what a single caller can ask the
# model to do, so one request cannot monopolise the lock the worker needs.
MAX_TEXTS_PER_REQUEST = 32
MAX_CHARS_PER_TEXT = 32_000

# How long to wait for the worker thread to publish its loop at startup,
# and how long to let it finish the current job at shutdown.
WORKER_START_SECONDS = 10.0
WORKER_SHUTDOWN_SECONDS = 60.0


class SharedEmbedder:
    """
    Thread-safe wrapper that serialises model access at batch granularity.

    Length-sorting happens here rather than inside the inner embedder so it
    still applies across the whole call -- it is worth 2.88x on padded
    tokens (see OnnxEmbedder.embed) -- while the lock is still released
    between sub-batches.

    At the default batch size of 1 the sorting is a no-op, because a batch
    of one is never padded. It stays because the batch size is configurable
    and anything above 1 needs it.
    """

    def __init__(
        self,
        inner: OnnxEmbedder,
        lock: threading.Lock | None = None,
        batch_size: int = 1,
    ):
        self._inner = inner
        self._lock = lock or threading.Lock()
        self._batch_size = max(1, batch_size)
        self.model_id = inner.model_id
        self.identity = inner.identity
        self.stats = inner.stats

    def embed(
        self,
        texts,
        batch_size: int | None = None,
        sort_by_length: bool = True,
    ) -> np.ndarray:
        """
        `batch_size` defaults to the configured one, not the library's 16.

        The pipeline calls `embed(texts)` with no batch size, so the default
        is what the worker actually uses -- and it decides how long a query
        can be stuck behind indexing.
        """
        # Clamped on the per-call path too, not just in __init__: a zero
        # would make `range(0, n, 0)` raise, and a negative one would loop
        # forever.
        batch_size = max(1, self._batch_size if batch_size is None else batch_size)
        texts = list(texts)
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)

        order = list(range(len(texts)))
        if sort_by_length and len(texts) > batch_size:
            order.sort(key=lambda i: len(texts[i]))

        out = np.empty((len(texts), EMBED_DIM), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            with self._lock:
                vecs = self._inner.embed(
                    [texts[i] for i in idx],
                    batch_size=len(idx),
                    sort_by_length=False,
                )
            for slot, i in enumerate(idx):
                out[i] = vecs[slot]
        return out

    def embed_iter(self, texts, batch_size: int | None = None):
        batch_size = self._batch_size if batch_size is None else batch_size
        buffer: list[str] = []
        for text in texts:
            buffer.append(text)
            if len(buffer) >= batch_size:
                yield self.embed(buffer, batch_size)
                buffer = []
        if buffer:
            yield self.embed(buffer, batch_size)


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_TEXTS_PER_REQUEST)


class EmbedResponse(BaseModel):
    model_id: str
    dim: int
    vectors: list[str]
    bits: list[str]


def _constant_time_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def _require_key(settings):
    """
    Bearer auth, enforced whenever a key is configured.

    Compared in constant time: a plain `==` on a shared secret leaks its
    prefix through timing, and this one is long-lived.
    """

    async def dependency(authorization: str = Header(default="")):
        expected = settings.embedding_api_key
        if not expected:
            return
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not _constant_time_eq(token, expected):
            raise HTTPException(
                status_code=401, detail="invalid or missing bearer token"
            )

    return dependency


def build_app(settings=None) -> FastAPI:
    settings = settings or get_settings()

    # An open endpoint that runs a model on caller-supplied text is a free
    # compute service for whoever finds it. Refuse to start rather than
    # quietly expose one.
    if settings.app_env == "production" and not settings.embedding_api_key:
        raise RuntimeError(
            "EMBEDDING_API_KEY must be set in production: the embedding "
            "endpoint runs a model on caller-supplied input."
        )

    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("indexer starting: loading %s", settings.embedding_model_id)
        inner = OnnxEmbedder(
            model_id=settings.embedding_model_id,
            onnx_file=getattr(settings, "embedding_onnx_file", ONNX_FILE),
            max_tokens=getattr(settings, "embedding_max_tokens", MAX_SEQUENCE_TOKENS),
            threads=getattr(settings, "embedding_threads", 0) or None,
            enable_mem_arena=getattr(settings, "embedding_mem_arena", False),
        )
        shared = SharedEmbedder(
            inner, batch_size=getattr(settings, "indexer_embed_batch", 1)
        )
        state["embedder"] = shared

        worker = _start_worker_thread(settings, shared)
        state["worker"] = worker
        logger.info(
            "indexer ready: model=%s, worker=%s",
            shared.model_id,
            "running" if worker is not None else "disabled",
        )
        try:
            yield
        finally:
            if worker is not None:
                logger.info("indexer stopping; waiting for the current job")
                worker.stop(timeout=WORKER_SHUTDOWN_SECONDS)

    # No interactive docs in production. This service talks to one caller
    # -- the API -- and never to a person, so /docs and /redoc are surface
    # that publishes the request schema of an authenticated endpoint and
    # buy nothing. They stay on outside production, where they are useful
    # for poking at the endpoint by hand.
    in_production = settings.app_env == "production"

    app = FastAPI(
        title="Clarix indexer",
        description=(
            "Embedding endpoint for the API's query path, and the indexing "
            "worker. Holds the model so the API does not have to."
        ),
        version="2.0.0",
        lifespan=lifespan,
        docs_url=None if in_production else "/docs",
        redoc_url=None if in_production else "/redoc",
        openapi_url=None if in_production else "/openapi.json",
    )

    @app.post(
        "/embed",
        response_model=EmbedResponse,
        dependencies=[Depends(_require_key(settings))],
    )
    async def embed(payload: EmbedRequest) -> EmbedResponse:
        """
        Embed texts and return them in both forms the SQL needs.

        Both literals come from the same helpers the ingestion path uses
        (`vector_to_sql`, `bits_to_sql`), because a query formatted
        differently from the rows it is compared against is a ranking bug
        that no test of either side alone would catch.
        """
        for text in payload.texts:
            if len(text) > MAX_CHARS_PER_TEXT:
                raise HTTPException(
                    status_code=413,
                    detail=f"text exceeds {MAX_CHARS_PER_TEXT} characters",
                )

        embedder: SharedEmbedder = state["embedder"]
        # Off the event loop: embedding is CPU-bound and would otherwise
        # block every other request for its duration.
        matrix = await asyncio.to_thread(
            embedder.embed, payload.texts, len(payload.texts)
        )
        packed = binary_quantize(matrix)
        return EmbedResponse(
            # The composite identity, not the repository id: the client
            # must be able to tell fp16 vectors from int8 ones.
            model_id=embedder.identity,
            dim=EMBED_DIM,
            vectors=[vector_to_sql(row) for row in matrix],
            bits=[bits_to_sql(packed[i], EMBED_DIM) for i in range(len(matrix))],
        )

    @app.api_route("/health", methods=["GET", "HEAD"])
    async def health():
        worker = state.get("worker")
        embedder = state.get("embedder")
        return {
            "status": "healthy" if embedder is not None else "starting",
            "service": "clarix-indexer",
            "model_id": getattr(embedder, "identity", None),
            "dim": EMBED_DIM,
            "worker_running": bool(worker is not None and worker.is_alive()),
        }

    return app


class WorkerThread:
    """
    The worker running on its own thread, with a stop that crosses threads.

    The worker waits on an `asyncio.Event` owned by *its* loop, which the
    server's loop must not touch directly. `call_soon_threadsafe` is the
    supported way across that boundary; polling a `threading.Event` from
    the worker loop would also work but wakes it constantly and delays
    shutdown by up to the poll interval.
    """

    def __init__(self, thread: threading.Thread, ready: threading.Event, holder: dict):
        self._thread = thread
        self._ready = ready
        self._holder = holder

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self, timeout: float = 60.0) -> None:
        # The loop and event only exist once the thread has started them.
        # Without this wait, a shutdown immediately after startup would
        # find nothing to signal and leave the thread running.
        self._ready.wait(timeout=WORKER_START_SECONDS)
        loop = self._holder.get("loop")
        event = self._holder.get("stop")
        if loop is not None and event is not None:
            loop.call_soon_threadsafe(event.set)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning("worker thread did not stop within %.0fs", timeout)


def _start_worker_thread(settings, embedder) -> WorkerThread | None:
    """
    Run the indexing worker on its own thread and event loop.

    A separate loop, not a task: the pipeline's embed call is synchronous
    and would block the HTTP loop for the length of a batch.
    """
    if not getattr(settings, "indexer_run_worker", True):
        logger.info("worker disabled (INDEXER_RUN_WORKER=false); serving /embed only")
        return None

    from app.indexing.chunker import ASTChunker
    from app.indexing.tokens import get_token_counter
    from app.indexing.worker import WorkerConfig, run_worker

    config = WorkerConfig(
        workdir=Path(settings.worker_dir),
        model_id=settings.embedding_model_id,
        index_version=settings.index_version,
        batch_size=settings.index_batch_size,
    )
    chunker = ASTChunker(count_tokens=get_token_counter().count)

    holder: dict = {}
    ready = threading.Event()

    def target() -> None:
        from app.database import async_session_factory

        async def main() -> None:
            holder["loop"] = asyncio.get_running_loop()
            holder["stop"] = asyncio.Event()
            ready.set()
            await run_worker(
                async_session_factory, config, embedder, chunker,
                stop=holder["stop"],
            )

        try:
            asyncio.run(main())
        except Exception:  # noqa: BLE001 - a dead worker must be visible, not silent
            logger.exception("worker thread died")
        finally:
            # Unblock anyone waiting on startup even if the loop never ran.
            ready.set()

    thread = threading.Thread(target=target, name="clarix-worker", daemon=True)
    thread.start()
    return WorkerThread(thread, ready, holder)
