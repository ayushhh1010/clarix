"""
Application configuration using Pydantic Settings.
Loads from environment variables and .env file.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The docker-compose database from the README. Declared up here so the
# production guard can recognise it.
LOCAL_DEV_DATABASE_URL = (
    "postgresql+asyncpg://clarix:clarix_secret@localhost:5433/clarix_db"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── LLM providers ────────────────────────────────────────
    #
    # Every provider is optional. Those without a key are not constructed,
    # so the router's candidate list is exactly what is configured. With
    # none configured, retrieval still works and generation degrades to
    # returning citations (see app/routes/chat.py).
    #
    # Ordered roughly by the free allowance measured in 2026-09:
    # Cerebras 1M tokens/day, Groq 200K/day, Gemini 1K requests/day,
    # OpenRouter 50/day. See app/llm/budget.py.
    cerebras_api_key: str = ""
    cerebras_model: str = "llama-3.3-70b"

    groq_api_key: str = ""
    # Groq retired the Llama 3.3 models. This is the free-tier model the
    # context budget in app/retrieval/context.py is sized against, and it
    # is on the account's model list; `llama-3.3-70b-versatile` returns
    # `model_not_found`, which degraded every answer to citations.
    groq_model: str = "openai/gpt-oss-120b"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"

    openrouter_api_key: str = ""
    openrouter_model: str = "deepseek/deepseek-chat:free"

    # ── Query embedding ──────────────────────────────────────
    #
    # The API does not load the embedding model; it calls the indexer's
    # endpoint so both sides use the same one. Unset means the dense
    # retrieval arm is skipped and lexical + symbol carry the query.
    embedding_endpoint: str = ""
    embedding_api_key: str = ""
    embedding_model_id: str = "jinaai/jina-embeddings-v2-base-code"

    # Which ONNX export, and the encoder truncation cap. Both change the
    # vectors, so both are part of the identity the API and the indexer
    # must agree on -- see embedder.embedding_identity.
    #
    # int8 at a 384-token cap is what fits a 512 MiB instance with room to
    # spare. Measured on Linux with the worker running and a cold model
    # download, peak RSS against the 537 MB cap
    # (bench/bench_service_memory.py):
    #
    #     fp16 @ 512    1,135 MB   does not fit -- not at any cap
    #     int8 @ 1024     619 MB   does not fit
    #     int8 @ 512      469 MB   fits, 68 MB spare (503 MB on a rerun)
    #     int8 @ 384      432 MB   fits, 105 MB spare
    #
    # fp16 cannot be made to fit: it needs ~1 GB merely to load, because
    # CPUs have no fp16 kernels and ONNX Runtime upcasts every weight to
    # fp32. The rest of the peak is attention, which is O(sequence^2),
    # which is why the truncation cap is the second lever.
    #
    # 384 over 512 because the same measurement varies by ~30 MB between
    # runs, and 512 leaves too little margin for that, while costing the
    # same in quality: semantic ROUTED MRR is 0.696 at 384 against 0.693
    # at 512 -- indistinguishable.
    #
    # The quality cost is measured end to end, not inferred from the
    # "90.5% agreement with fp32" proxy, which badly overstated it:
    # semantic ROUTED MRR 0.707 -> 0.696, and identifier queries are
    # unchanged. Raise both on a host with more memory, and bump
    # index_version when you do -- the stored vectors must be rebuilt.
    embedding_onnx_file: str = "onnx/model_quantized.onnx"
    embedding_max_tokens: int = 384

    # ONNX Runtime intra-op threads. 0 inherits the core count.
    #
    # Pinned to 1, but NOT for memory: measured, it makes no difference
    # (469 MB peak at 1 thread against 464 MB at the default, which is
    # inside the ~30 MB run-to-run variance). It is pinned because a free
    # instance reports far more cores than it is actually scheduled, and
    # oversubscribing them costs latency rather than buying throughput.
    embedding_threads: int = 1

    # ONNX Runtime's CPU memory arena. ON, and this was the fix for an
    # out-of-memory kill rather than a cause of one.
    #
    # Measured on a real indexing run of 1,385 chunks
    # (bench/bench_index_memory.py), peak RSS against the 537 MB cap:
    #
    #     arena off    574 MB    343 s    killed by the platform
    #     arena ON     430 MB    301 s    fits, 107 MB spare, and faster
    #
    # The arena allocates a pool once and reuses it. Without it every one
    # of those 1,385 inferences allocates and frees, and the high-water
    # mark of that churn is ~200 MB above the model.
    #
    # This was off for a while because the arena was measured on *fp16*,
    # where it looked catastrophic (1,809 MB). That reading was about the
    # model, not the arena: CPUs have no fp16 kernels, so ONNX Runtime
    # upcasts every weight to fp32 and the model alone needs ~1 GB. The
    # conclusion was carried over to int8 without re-measuring, which is
    # the actual mistake.
    embedding_mem_arena: bool = True

    # ── Indexer process (python -m app.indexing) ─────────────
    #
    # Serves the endpoint above and runs the indexing worker. Separate from
    # the API because it holds the model: measured 439 MB peak against the
    # API's 59 MB, and the two together exceed a 512 MB instance.
    indexer_port: int = 8081
    indexer_run_worker: bool = True

    # How many chunks the worker embeds per lock acquisition. The indexer
    # shares one model between the worker and the query endpoint, so this
    # sets the worst case a query can wait behind indexing.
    #
    # 1, because batching buys nothing here. Measured on 200 real chunks
    # with the arena off (bench/bench_embed_batch.py):
    #
    #     batch  1   2.45 chunks/s   median hold 0.25s   p95  1.11s
    #     batch 16   2.41 chunks/s   median hold 4.84s   p95 16.17s
    #
    # Throughput is flat and hold time scales linearly: a batch of one has
    # no padding at all, which is the only thing batching was buying back.
    # Raise it only if a measurement on the target machine says otherwise.
    indexer_embed_batch: int = 1

    # ── Google OAuth only (Gemini uses gemini_api_key above) ──
    google_api_key: str = ""

    # ── PostgreSQL ──────────────────────────────────────────
    # The local docker-compose database. Convenient in development and
    # never right in production -- see the validator below.
    database_url: str = LOCAL_DEV_DATABASE_URL

    # ── Repos ───────────────────────────────────────────────
    repos_dir: str = "./data/repos"

    # ── Application ─────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"

    # ── Auth / JWT ──────────────────────────────────────────
    secret_key: str = "change-me-in-production-use-a-long-random-string"
    frontend_url: str = "http://localhost:3000"
    backend_url: str = "http://localhost:8000"

    # ── GitHub OAuth ────────────────────────────────────────
    github_client_id: str = ""
    github_client_secret: str = ""

    # ── Google OAuth ────────────────────────────────────────
    google_client_id: str = ""
    google_client_secret: str = ""

    # ── Indexing ─────────────────────────────────────────────
    index_version: int = 2
    index_batch_size: int = 64
    worker_dir: str = "./data/work"

    # ── LLM ─────────────────────────────────────────────────
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # ── Retrieval ────────────────────────────────────────────
    #
    # 4,000 tokens, not v1's 12,000: Cerebras caps free-tier context at
    # 8,192 and Groq's free gpt-oss-120b allows 8,000 tokens/minute, so an
    # over-budget request fails rather than costing more.
    rag_top_k: int = 20
    context_budget_tokens: int = 4000

    @model_validator(mode="after")
    def _refuse_dev_defaults_in_production(self):
        """
        Fail loudly rather than quietly reaching for localhost.

        With DATABASE_URL unset, a production deploy fell back to the
        docker-compose default and spent its startup trying to open a
        connection to 127.0.0.1:5433 inside a container where nothing is
        listening. The error it produced -- "Connection refused" against
        localhost -- describes the symptom and hides the cause, which is
        simply that the variable was never set.

        The same reasoning already applies to SECRET_KEY
        (security._check_secret); this closes the matching hole for the
        database, and for the embedding endpoint, where the failure is
        worse because it is silent: an unset endpoint disables dense
        retrieval and the service still answers, just less well.
        """
        if self.app_env != "production":
            return self

        if self.database_url == LOCAL_DEV_DATABASE_URL:
            raise ValueError(
                "DATABASE_URL is not set: the application would fall back "
                "to the local docker-compose database at localhost:5433, "
                "which does not exist in production. Set DATABASE_URL to "
                "your managed Postgres connection string."
            )
        if "localhost" in self.database_url or "127.0.0.1" in self.database_url:
            raise ValueError(
                f"DATABASE_URL points at localhost in production: "
                f"{self.database_url.split('@')[-1]}"
            )

        # OAuth redirect URIs are built from backend_url, and the provider
        # sends the user wherever it says. Left at its default in
        # production, the consent screen bounces the visitor to
        # http://localhost:8000 -- their own machine -- or the provider
        # rejects the request outright for not matching a registered
        # callback. Neither says what is actually wrong.
        #
        # Only checked when a provider is configured: OAuth is optional,
        # and a deployment using email and password alone has no reason to
        # care what backend_url says.
        oauth_configured = bool(self.github_client_id or self.google_client_id)
        if oauth_configured:
            for field, value in (
                ("BACKEND_URL", self.backend_url),
                ("FRONTEND_URL", self.frontend_url),
            ):
                if "localhost" in value or "127.0.0.1" in value:
                    raise ValueError(
                        f"{field} points at localhost in production "
                        f"({value}) while OAuth is configured. The provider "
                        f"would redirect users there. Set it to the public "
                        f"URL of the service."
                    )
        return self

    @property
    def repos_path(self) -> Path:
        path = Path(self.repos_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def worker_path(self) -> Path:
        path = Path(self.worker_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()