"""
The identity that the index and the query path must agree on.

Why this is its own module
--------------------------
Both sides need this function, and they must compute it the same way --
duplicating it is exactly the drift that a shared helper exists to
prevent.

But the obvious home, `app/indexing/embedder.py`, imports numpy, and
`app/indexing/__init__.py` imports the tree-sitter chunker. The API
process deliberately loads neither: its import-time RSS is the binding
constraint on the deployment (59 MB against the indexer's 439 MB, see
bench/bench_colocated_rss.py). Importing the identity from there would
quietly pull numpy into the web process and undo that.

So it lives here, in a leaf module with no imports at all.
"""

from __future__ import annotations


def embedding_identity(model_id: str, onnx_file: str, max_tokens: int) -> str:
    """
    What has to match between the index and the query, in one string.

    The Hugging Face repository id alone is not enough.
    `model_fp16.onnx` and `model_quantized.onnx` are the same model and
    produce *different vectors*, and so does the same file under a
    different truncation cap. Comparing only the repository id would let
    an API configured one way query an index built the other way, which
    yields confident, wrong rankings and raises nothing -- precisely the
    failure the mismatch check is supposed to catch.

    So the identity carries all three things that change the vectors.

    >>> embedding_identity("jinaai/x", "onnx/model_quantized.onnx", 512)
    'jinaai/x|model_quantized|512'
    """
    stem = onnx_file.rsplit("/", 1)[-1]
    if stem.endswith(".onnx"):
        stem = stem[: -len(".onnx")]
    return f"{model_id}|{stem}|{max_tokens}"
