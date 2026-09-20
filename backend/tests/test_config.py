"""
Configuration tests.

`.env.example` is documentation that rots silently. The v1 copy of it still
advertised OLLAMA_MODEL, EMBEDDING_MODEL, REDIS_URL and CHROMA_PERSIST_DIR
long after every one of those was deleted, and listed none of the settings
the application had gained. Nothing caught it, because nothing read the file
except a human following the README.

These tests make the file checkable: every key it documents must be a real
setting, and every setting a deployment cannot work without must be
documented.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from app.config import Settings

BACKEND_DIR = pathlib.Path(__file__).resolve().parents[1]
ENV_EXAMPLE = BACKEND_DIR / ".env.example"

# Settings that are internal knobs rather than deployment configuration.
# Leaving these out of .env.example is deliberate, not an omission.
UNDOCUMENTED_BY_DESIGN = {
    "google_api_key",   # legacy alias; gemini_api_key is the documented one
}


def _example_keys() -> set[str]:
    keys = set()
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Z0-9_]+)=", line)
        if match:
            keys.add(match.group(1).lower())
    return keys


def test_env_example_exists():
    assert ENV_EXAMPLE.is_file(), "the file the README tells people to copy"


def test_every_documented_key_is_a_real_setting():
    """
    A key here that Settings does not define is silently ignored at runtime
    -- the operator sets it, nothing reads it, and the symptom shows up
    somewhere unrelated.
    """
    declared = set(Settings.model_fields)
    unknown = sorted(_example_keys() - declared)
    assert not unknown, (
        f".env.example documents settings that do not exist: {unknown}. "
        "Either add them to Settings or remove them from the example."
    )


def test_every_setting_is_documented():
    declared = set(Settings.model_fields) - UNDOCUMENTED_BY_DESIGN
    missing = sorted(declared - _example_keys())
    assert not missing, (
        f"settings missing from .env.example: {missing}. An operator cannot "
        "configure what is not written down."
    )


def test_no_deleted_v1_settings_remain():
    """
    Named explicitly so the failure says what happened rather than just
    'unknown key'. These were real settings once.
    """
    gone = {"redis_url", "chroma_persist_dir", "ollama_model", "embedding_model"}
    present = sorted(gone & _example_keys())
    assert not present, f".env.example still advertises removed settings: {present}"


def test_no_real_secrets_are_committed():
    """
    The example file is committed. Placeholder values only.
    """
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    for pattern, what in (
        (r"sk-[A-Za-z0-9]{20,}", "an OpenAI-style key"),
        (r"gsk_[A-Za-z0-9]{20,}", "a Groq key"),
        (r"AIza[A-Za-z0-9_\-]{30,}", "a Google API key"),
        (r"ghp_[A-Za-z0-9]{20,}", "a GitHub token"),
    ):
        assert not re.search(pattern, text), f".env.example appears to contain {what}"


@pytest.mark.parametrize(
    "name",
    ["database_url", "embedding_endpoint", "embedding_api_key", "secret_key"],
)
def test_deployment_critical_settings_are_documented(name):
    """
    The four a deployment is broken without. EMBEDDING_ENDPOINT in
    particular: leaving it unset disables dense retrieval silently, which is
    exactly how the dense arm came to be off in every real deployment.
    """
    assert name in _example_keys()


def test_indexer_embed_batch_defaults_to_one():
    """
    Measured, not preferred: throughput is flat across batch sizes with the
    ONNX arena off, while the lock hold a query waits behind scales
    linearly (0.25 s at batch 1 against 4.84 s at batch 16).
    """
    assert Settings.model_fields["indexer_embed_batch"].default == 1
