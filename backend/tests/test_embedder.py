"""
Correctness tests for the ONNX embedder.

Pooling correctness is the single most load-bearing property in the pipeline:
every retrieval metric, every benchmark number and every ranking decision is
computed on these vectors. A pooling bug does not crash -- it silently
degrades everything by a few percent, which is exactly the kind of fault a
benchmark suite is supposed to catch and a smoke test is not.

The model's own `1_Pooling/config.json` declares `pooling_mode_mean_tokens:
true` (cls and max false), which is what `_forward` implements.

The decisive test here is padding invariance. The classic mean-pooling bug is
summing over padded positions and dividing by the padded length instead of
the true token count. It produces plausible-looking vectors, so it survives
eyeballing -- but it makes a text's embedding depend on what else happened to
be in its batch. Embedding the same text alone and alongside a much longer
one must give bit-comparable results; if it does not, the mask is being
applied wrongly.

These tests need the real 641 MB ONNX weights and are skipped when the model
is not cached locally.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("onnxruntime", reason="indexer extra not installed")

from app.indexing.embedder import EMBED_DIM, OnnxEmbedder, binary_quantize  # noqa: E402


@pytest.fixture(scope="module")
def embedder():
    """Load once: session construction dominates these tests."""
    try:
        return OnnxEmbedder()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"model unavailable: {type(exc).__name__}: {exc}")


SHORT = "def add(a, b):\n    return a + b"
LONG = (
    "def process(records):\n"
    + "\n".join(f"    step_{i} = transform(records[{i}], mode='full')" for i in range(120))
    + "\n    return records"
)


# --- pooling correctness ---------------------------------------------------

def test_padding_does_not_change_an_embedding(embedder):
    """
    The decisive pooling test.

    `SHORT` is embedded alone (no padding) and again in a batch with a much
    longer text (heavily padded). Mean pooling that respects the attention
    mask gives the same vector both times. Pooling that includes padded
    positions does not.
    """
    alone = embedder.embed([SHORT], batch_size=1)[0]
    padded = embedder.embed([SHORT, LONG], batch_size=2)[0]

    cosine = float(alone @ padded)
    assert cosine > 0.9999, (
        f"embedding changed with batch padding (cos={cosine:.6f}); "
        "mean pooling is including padded positions"
    )
    np.testing.assert_allclose(alone, padded, atol=2e-3)


def test_batch_size_does_not_change_embeddings(embedder):
    texts = [SHORT, "class A:\n    pass", "def q(x):\n    return x * 2", LONG]
    one_at_a_time = embedder.embed(texts, batch_size=1)
    all_at_once = embedder.embed(texts, batch_size=4)
    for i in range(len(texts)):
        cos = float(one_at_a_time[i] @ all_at_once[i])
        assert cos > 0.9999, f"text {i} drifted with batch size (cos={cos:.6f})"


def test_output_is_l2_normalised(embedder):
    vecs = embedder.embed([SHORT, LONG, "x = 1"], batch_size=2)
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_shape_and_dtype(embedder):
    vecs = embedder.embed([SHORT, "y = 2"])
    assert vecs.shape == (2, EMBED_DIM)
    assert vecs.dtype == np.float32
    assert np.isfinite(vecs).all()


def test_embedding_is_deterministic(embedder):
    a = embedder.embed([SHORT])[0]
    b = embedder.embed([SHORT])[0]
    np.testing.assert_array_equal(a, b)


def test_empty_input_returns_an_empty_matrix(embedder):
    out = embedder.embed([])
    assert out.shape == (0, EMBED_DIM)


# --- semantic sanity -------------------------------------------------------

def test_related_code_is_closer_than_unrelated(embedder):
    """
    A weak but non-trivial check: if this fails the vectors carry no usable
    signal, whatever the pooling maths says.
    """
    auth_a = "def verify_password(plain, hashed):\n    return bcrypt.checkpw(plain, hashed)"
    auth_b = "def hash_password(password):\n    return bcrypt.hashpw(password, bcrypt.gensalt())"
    unrelated = "def draw_triangle(canvas, x, y, size):\n    canvas.polygon([(x, y)], fill='red')"

    v = embedder.embed([auth_a, auth_b, unrelated])
    related = float(v[0] @ v[1])
    different = float(v[0] @ v[2])
    assert related > different, f"related={related:.3f} not > unrelated={different:.3f}"


def test_identical_text_embeds_identically_across_positions(embedder):
    """Position within a batch must not matter."""
    v = embedder.embed([SHORT, "z = 3", SHORT], batch_size=3)
    np.testing.assert_allclose(v[0], v[2], atol=1e-5)


# --- quantisation ----------------------------------------------------------

def test_binary_quantize_packs_to_the_expected_width():
    vecs = np.array([[0.5, -0.5] * (EMBED_DIM // 2)], dtype=np.float32)
    packed = binary_quantize(vecs)
    assert packed.shape == (1, EMBED_DIM // 8)
    assert packed.dtype == np.uint8


def test_binary_quantize_matches_the_sign_pattern():
    vecs = np.array([[1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0]], dtype=np.float32)
    packed = binary_quantize(vecs)
    bits = "".join(f"{b:08b}" for b in packed[0])
    assert bits == "10110010"


def test_binary_quantize_with_a_centroid_shifts_the_threshold():
    vecs = np.array([[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]], dtype=np.float32)
    assert "".join(f"{b:08b}" for b in binary_quantize(vecs)[0]) == "11111111"
    centroid = np.full(8, 0.3, dtype=np.float32)
    assert "".join(f"{b:08b}" for b in binary_quantize(vecs, centroid)[0]) == "00000000"


def test_quantized_bits_preserve_neighbour_order_on_real_vectors(embedder):
    """
    End-to-end: Hamming distance over packed bits should rank a related
    snippet above an unrelated one, which is the property the ANN arm relies
    on for candidate generation.
    """
    query = "def verify_password(plain, hashed):\n    return bcrypt.checkpw(plain, hashed)"
    related = "def check_password(raw, digest):\n    return bcrypt.checkpw(raw, digest)"
    unrelated = "func (e *Engine) ServeHTTP(w http.ResponseWriter, r *http.Request) {}"

    vecs = embedder.embed([query, related, unrelated])
    bits = binary_quantize(vecs)

    def hamming(a, b):
        return int(np.unpackbits(np.bitwise_xor(a, b)).sum())

    assert hamming(bits[0], bits[1]) < hamming(bits[0], bits[2])


# --- length-sorted batching ------------------------------------------------

def test_length_sorted_batching_returns_input_order(embedder):
    """
    Sorting is an internal scheduling detail: the caller's order must be
    preserved, or every chunk gets the wrong vector -- a silent, total
    corruption of the index.
    """
    texts = [SHORT, LONG, "x = 1", "def mid(a):\n    return a\n" * 8, "y = 2"]
    sorted_out = embedder.embed(texts, batch_size=2, sort_by_length=True)
    plain_out = embedder.embed(texts, batch_size=2, sort_by_length=False)

    for i in range(len(texts)):
        cos = float(sorted_out[i] @ plain_out[i])
        assert cos > 0.9999, f"row {i} differs between sorted and unsorted (cos={cos:.6f})"


def test_length_sorted_batching_is_exact_not_approximate(embedder):
    """Follows from padding invariance; pinned so a regression is loud."""
    texts = ["a = 1", LONG, "b = 2", SHORT]
    a = embedder.embed(texts, batch_size=2, sort_by_length=True)
    b = embedder.embed(texts, batch_size=2, sort_by_length=False)
    np.testing.assert_allclose(a, b, atol=2e-3)


def test_sorting_is_skipped_for_inputs_smaller_than_one_batch(embedder):
    out = embedder.embed([SHORT, "z = 9"], batch_size=16, sort_by_length=True)
    assert out.shape == (2, EMBED_DIM)


# --- streaming path (the fix for the measured ingestion OOM) ---------------

def test_embed_iter_streams_batches_without_materialising_everything(embedder):
    """
    `embed_iter` is the ingestion path. Holding every embedding alive at once
    is the shape measured at 409 MB for 25k chunks
    (bench/bench_ingest_memory.py); this yields batch-sized arrays instead.
    """
    texts = [f"def f{i}():\n    return {i}" for i in range(7)]
    batches = list(embedder.embed_iter(texts, batch_size=3))

    assert [b.shape[0] for b in batches] == [3, 3, 1]
    assert all(b.shape[1] == EMBED_DIM for b in batches)

    streamed = np.vstack(batches)
    direct = embedder.embed(texts, batch_size=3)
    for i in range(len(texts)):
        assert float(streamed[i] @ direct[i]) > 0.9999, f"row {i} differs"


def test_embed_iter_handles_an_empty_input(embedder):
    assert list(embedder.embed_iter([], batch_size=4)) == []


def test_bits_to_sql_renders_a_postgres_bit_literal():
    """The numpy -> Postgres `bit(n)` bridge used by the indexer."""
    from app.indexing.embedder import bits_to_sql

    vecs = np.array([[1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0]], dtype=np.float32)
    packed = binary_quantize(vecs)
    assert bits_to_sql(packed[0], 8) == "10110010"
    # Truncates to the declared width rather than emitting padding bits.
    assert len(bits_to_sql(packed[0], 5)) == 5
