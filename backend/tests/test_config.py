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

from app.config import LOCAL_DEV_DATABASE_URL, Settings

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


def test_the_deployable_embedding_config_is_the_default():
    """
    The defaults are what a fresh deploy runs, so they must be the
    configuration that actually fits the target instance.

    fp16 was the default until Render killed the service with
    "Out of memory (used over 512Mi)". Measured on Linux with the worker
    running and a cold download, fp16 peaks at 1,135 MB and cannot fit at
    any truncation cap; int8 at a 384-token cap peaks at 432 MB against
    the 537 MB limit. See bench/bench_service_memory.py.
    """
    fields = Settings.model_fields
    assert fields["embedding_onnx_file"].default == "onnx/model_quantized.onnx"
    assert fields["embedding_max_tokens"].default == 384


def test_index_version_is_ahead_of_the_fp16_index():
    """
    The embedding variant changed, so the vectors changed. Anything
    indexed at version 1 holds fp16 vectors and must be rebuilt rather
    than queried with int8 ones.
    """
    from app.models_v2 import INDEX_VERSION

    assert Settings.model_fields["index_version"].default >= 2
    assert INDEX_VERSION >= 2


def test_indexer_embed_batch_defaults_to_one():
    """
    Measured, not preferred: throughput is flat across batch sizes with the
    ONNX arena off, while the lock hold a query waits behind scales
    linearly (0.25 s at batch 1 against 4.84 s at batch 16).
    """
    assert Settings.model_fields["indexer_embed_batch"].default == 1


# --- the production guard --------------------------------------------------
#
# A deploy with DATABASE_URL unset fell back to the docker-compose default
# and spent its startup dialling 127.0.0.1:5433 inside a container. The
# error said "Connection refused", which describes the symptom and hides
# the cause: the variable was simply never set.


def test_production_refuses_the_local_dev_database():
    with pytest.raises(ValueError, match="DATABASE_URL is not set"):
        Settings(app_env="production", database_url=LOCAL_DEV_DATABASE_URL)


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
def test_production_refuses_any_localhost_database(host):
    """
    Not just the exact default: any localhost URL in production is a
    misconfiguration, since nothing is listening inside the container.
    """
    with pytest.raises(ValueError, match="localhost"):
        Settings(
            app_env="production",
            database_url=f"postgresql+asyncpg://u:p@{host}:5432/db",
        )


def test_development_still_accepts_the_local_default():
    """The guard must not make local development harder."""
    s = Settings(app_env="development", database_url=LOCAL_DEV_DATABASE_URL)
    assert s.database_url == LOCAL_DEV_DATABASE_URL


def test_production_accepts_a_managed_database():
    s = Settings(
        app_env="production",
        database_url="postgresql+asyncpg://u:p@ep-x.neon.tech/db?ssl=require",
    )
    assert "neon.tech" in s.database_url


def test_the_guard_does_not_leak_the_password():
    """
    The message names the host so the operator can see what is wrong,
    and must not echo the credentials back into a deploy log.
    """
    try:
        Settings(
            app_env="production",
            database_url="postgresql+asyncpg://user:sup3rs3cret@localhost:5432/db",
        )
    except ValueError as exc:
        assert "sup3rs3cret" not in str(exc)
        assert "localhost" in str(exc)
    else:
        raise AssertionError("expected the guard to reject a localhost URL")
