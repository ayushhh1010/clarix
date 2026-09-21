"""
Local ONNX embedder for `jinaai/jina-embeddings-v2-base-code` (Apache-2.0).

Replaces the v1 HuggingFace Inference API embedder, which needed five retries
with ten-second backoff and explicit 503 cold-start handling -- that retry
loop is an accurate description of how reliable a serverless inference
endpoint is as a pipeline dependency.

Running locally is also a quality change, not just a reliability one: v1 used
`bge-small-en-v1.5`, a general English text model, for code retrieval.

Runs under onnxruntime with no torch and no transformers (~26 MB resident,
measured in bench/bench_import_rss.py), inside the indexer process. The API
server never imports this module.

Model notes
-----------
  * Mean pooling over the attention mask, then L2 normalisation -- this is
    what the model's sentence-transformers config specifies. Getting pooling
    wrong silently degrades every downstream metric, so it is asserted in
    tests against known-good reference vectors.
  * 768 dimensions, fixed. This model is not Matryoshka-trained, so vectors
    cannot be truncated to save space; storage is reduced by quantisation
    instead (see bench/bench_quantization.py for the measured recall cost).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from app.embedding_id import embedding_identity

logger = logging.getLogger(__name__)

MODEL_ID = "jinaai/jina-embeddings-v2-base-code"
# fp16 by default -- for MEMORY, not speed.
#
# CORRECTION. This previously read "free 1.36x" on the strength of an
# earlier run recording fp32 0.90 and fp16 1.22 chunks/s. Re-measured at the
# same settings (bench/bench_embed_model.py, 400 chunks, batch 16):
#
#   fp32  5.21 chunks/s  recall@10 100.0%  cos 1.0000  (reference)
#   fp16  3.94 chunks/s  recall@10 100.0%  cos 1.0000  <- 0.76x, SLOWER
#   int8  9.70 chunks/s  recall@10  90.5%  cos 0.9831  <- 1.86x for -9.5%
#
# fp16 is 24% slower than fp32 here, not 36% faster: the old claim was
# inverted. The ratio is machine-dependent -- CPUs have no native fp16
# kernels, so ORT inserts Cast nodes, and fp16 only wins where memory
# bandwidth, not compute, is the limit. The original run was ~5.8x slower in
# absolute terms, which is consistent with a bandwidth-bound machine. int8's
# ratio reproduced (1.86x vs 1.89x, recall 90.5% vs 91.0%), so the harness
# is sound; it is the fp16 comparison specifically that was wrong.
#
# fp16 still ships, because the binding constraint is RAM, not throughput
# (bench/bench_colocated_rss.py, arena off): fp16 peaks at 439 MB against
# fp32's 734 MB, and only fp16 fits a 512 MB instance.
#
# int8 remains a lever to pull only if throughput is critical AND the ~9.5%
# recall loss is shown not to matter end to end. It is not free.
# NOTE: this module default is the *benchmark reference*, not what the
# application runs. The app builds the embedder from `Settings`
# (`embedding_onnx_file`, `embedding_max_tokens`), which default to
# int8 at a 512-token cap because that is what fits a 512 MiB instance.
#
# The two differ on purpose. Committed artifacts under bench/results were
# measured against fp16 here, and changing this default would silently
# re-point every benchmark that calls `OnnxEmbedder()` with no arguments
# at a different model, invalidating numbers the checker verifies.
ONNX_FILE = "onnx/model_fp16.onnx"
EMBED_DIM = 768

# Chunks are capped at 1,024 tokens by the chunker, so 2,048 leaves headroom
# for the split-part signature headers without ever hitting the encoder's own
# limit. Longer inputs are truncated rather than dropped.
MAX_SEQUENCE_TOKENS = 2048

DEFAULT_BATCH_SIZE = 16


@dataclass
class EmbedStats:
    texts: int = 0
    batches: int = 0
    seconds: float = 0.0


class OnnxEmbedder:
    """Batched local embedding. Thread-unsafe; one instance per worker."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        onnx_file: str = ONNX_FILE,
        max_tokens: int = MAX_SEQUENCE_TOKENS,
        threads: int | None = None,
        enable_mem_arena: bool = False,
    ):
        """
        `enable_mem_arena` is the difference between deployable and not.

        ONNX Runtime's CPU arena pre-allocates and never returns memory, and
        on this model it dominates the process. Measured, fp16, 200 real
        chunks at batch 16 (bench/bench_colocated_rss.py):

            arena on    peak RSS 1,809 MB    3.40 chunks/s    55 ms/query
            arena off   peak RSS   439 MB    2.33 chunks/s    99 ms/query

        There is no middle setting: `arena_extend_strategy` changed nothing
        and disabling `mem_pattern` only reached 835 MB. So it is 439 MB or
        it does not fit a 512 MB instance, which makes the 1.46x throughput
        the price of deploying at all.

        It defaults to off for that reason. Benchmarks that want to measure
        the arena pass True explicitly.
        """
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.model_id = model_id
        self.onnx_file = onnx_file
        self.max_tokens = max_tokens
        self.identity = embedding_identity(model_id, onnx_file, max_tokens)
        self.stats = EmbedStats()

        model_path = hf_hub_download(model_id, onnx_file)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_cpu_mem_arena = enable_mem_arena
        if threads:
            opts.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}

        self._tok = Tokenizer.from_pretrained(model_id)
        self._tok.enable_truncation(max_length=max_tokens)
        self._tok.enable_padding(pad_id=0, pad_token="<pad>")  # noqa: S106 - a tokenizer pad token, not a credential

        logger.info(
            "ONNX embedder ready: %s (inputs=%s, max_tokens=%d)",
            model_id,
            sorted(self._input_names),
            max_tokens,
        )

    # -- core --------------------------------------------------------------

    def _forward(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self._tok.encode_batch(list(texts))
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        feeds: dict[str, np.ndarray] = {}
        if "input_ids" in self._input_names:
            feeds["input_ids"] = input_ids
        if "attention_mask" in self._input_names:
            feeds["attention_mask"] = attention
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self._session.run(None, feeds)
        hidden = outputs[0]  # (batch, seq, hidden)

        # Mean pooling over non-padding positions. Summing over padded
        # positions instead -- the common mistake -- shifts every vector
        # toward the pad embedding and quietly flattens the similarity space.
        mask = attention[..., None].astype(np.float32)
        summed = (hidden * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        pooled = summed / counts

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-12, None)).astype(np.float32)

    def embed(
        self,
        texts: Sequence[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        sort_by_length: bool = True,
    ) -> np.ndarray:
        """
        Embed texts, returning an (n, 768) float32 array in input order.

        A numpy array, not `list[list[float]]`: the Python-list representation
        was measured at 10.2x the f32 payload and is what exhausted the 512 MB
        container during v1 ingestion (bench/bench_ingest_memory.py).

        Length-sorted batching (`sort_by_length`, on by default)
        -------------------------------------------------------
        Every sequence in a batch is padded to the longest one in it. Chunk
        lengths are heavily skewed -- measured on the benchmark corpus:
        p50 117 tokens, p99 935, max 1,065 -- so a batch that happens to
        contain one long chunk pads fifteen short ones up to its length.

        Measured on that corpus at batch 16: arrival-order batching processes
        1,731,831 padded tokens to embed 593,384 real ones (2.92x the
        necessary work). Sorting by length first brings that to 601,401
        (1.01x) -- a 2.88x reduction in compute for identical output.

        Identical because padding invariance is a verified property of the
        pooling implementation, not an assumption: see
        tests/test_embedder.py::test_padding_does_not_change_an_embedding.
        Results are restored to input order before returning.
        """
        import time

        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)

        order = list(range(len(texts)))
        if sort_by_length and len(texts) > batch_size:
            # Character length is a good proxy for token count and costs
            # nothing; tokenising twice just to sort would eat the win.
            order.sort(key=lambda i: len(texts[i]))

        out = np.empty((len(texts), EMBED_DIM), dtype=np.float32)
        t0 = time.perf_counter()
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            vecs = self._forward([texts[i] for i in idx])
            for slot, i in enumerate(idx):
                out[i] = vecs[slot]
            self.stats.batches += 1
        self.stats.seconds += time.perf_counter() - t0
        self.stats.texts += len(texts)
        return out

    def embed_iter(
        self, texts: Iterable[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> Iterable[np.ndarray]:
        """
        Stream batches instead of materialising the whole matrix.

        This is the ingestion path: holding every embedding alive at once is
        precisely the shape that OOMed v1.
        """
        buffer: list[str] = []
        for text in texts:
            buffer.append(text)
            if len(buffer) >= batch_size:
                yield self.embed(buffer, batch_size)
                buffer = []
        if buffer:
            yield self.embed(buffer, batch_size)


def binary_quantize(vectors: np.ndarray, centroid: np.ndarray | None = None) -> np.ndarray:
    """
    Pack vectors to one bit per dimension, thresholding at `centroid`.

    MEASURED NEGATIVE RESULT. This function was written on the assumption
    that pgvector's zero-threshold `binary_quantize()` would lose signal,
    because encoder outputs generally carry a per-dimension bias and
    thresholding a biased dimension at zero sends every vector to the same
    bit. bench/bench_quantization.py tested that on 3,601 real code
    embeddings and it is false for this model: per-dimension |mean| averages
    0.0114 (max 0.0711), and centroid thresholding is indistinguishable from
    zero thresholding. From fetch=100 upward both stay in 0.9995-1.0000
    recall@10, and the largest gap between them at any over-fetch factor is
    0.0005 -- a single slot out of 200 queries x 10 -- with the sign going
    both ways (centroid leads at fetch=500, trails at fetch=2000). That is
    sampling noise, not a difference. For reference, exact halfvec scores
    0.9995 against fp32, so binary-plus-rescore is already at the ceiling
    set by the storage format rather than by the quantisation.

    So the ingestion path uses pgvector's SQL `binary_quantize()` directly
    and never passes a centroid. This stays because it is the right hedge if
    the embedding model is ever swapped: re-run that benchmark before
    assuming a new model is as well-centred as this one.

    Returns a uint8 array of shape (n, ceil(dim/8)), MSB-first per byte, which
    is the layout Postgres' `bit(n)` expects.
    """
    if centroid is None:
        centroid = np.zeros(vectors.shape[1], dtype=np.float32)
    return np.packbits(vectors > centroid, axis=1, bitorder="big")


def bits_to_sql(packed_row: np.ndarray, dim: int) -> str:
    """Render one packed row as a Postgres bit-string literal."""
    return "".join(f"{byte:08b}" for byte in packed_row)[:dim]


def vector_to_sql(vec: np.ndarray) -> str:
    """
    Render one vector as a Postgres halfvec literal.

    Indexing and query embedding must format vectors identically: the query
    is compared against stored rows, so a difference in precision here is a
    difference in ranking. Both paths call this.
    """
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
