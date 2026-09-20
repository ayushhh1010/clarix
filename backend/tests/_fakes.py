"""
Test doubles shared across test modules.

Why this file exists
--------------------
`test_worker.py` used to do `from tests.test_pipeline import FakeEmbedder`.
That resolved locally and failed in CI, and the difference was not the
test code -- it was how the package happened to be installed.

An editable install writes a `.pth` that puts `backend/` on `sys.path`,
which makes `tests` importable as an implicit namespace package. CI does a
plain `pip install .`, so `backend/` is not on the path and `tests` does
not exist as a module. The suite was quietly depending on a developer
convenience.

Living here fixes it for both. `tests/` has no `__init__.py`, so pytest
prepends that directory to `sys.path` when it collects, and a plain
`from _fakes import ...` resolves regardless of install mode or whether
pytest was started as `pytest` or `python -m pytest`.

The leading underscore keeps it out of collection -- it does not match
`test_*.py`, so pytest will not try to run it as a test module.
"""

from __future__ import annotations

import hashlib

import numpy as np

# The embedding width the pipeline expects. Kept local so importing this
# module does not require the indexer extra.
DIM = 768


class FakeEmbedder:
    """
    Deterministic unit vectors derived from the text; no model needed.

    Deterministic *per text*, so a chunk embedded twice gets the same
    vector and cache-hit assertions mean something. Unit length, because
    the retrieval SQL compares with cosine distance and a non-normalised
    vector would make those comparisons meaningless.

    The counters exist so tests can assert how much work the pipeline
    actually asked for -- notably that it never holds more than one batch
    at a time.
    """

    def __init__(self):
        self.calls = 0
        self.texts_embedded = 0
        self.max_batch_seen = 0

    def embed(self, texts, batch_size=16, sort_by_length=True):
        self.calls += 1
        self.texts_embedded += len(texts)
        self.max_batch_seen = max(self.max_batch_seen, len(texts))
        out = np.empty((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            v = rng.normal(size=DIM).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out
