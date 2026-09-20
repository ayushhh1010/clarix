"""
Application configuration using Pydantic Settings.
Loads from environment variables and .env file.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    database_url: str = "postgresql+asyncpg://clarix:clarix_secret@localhost:5433/clarix_db"

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
    index_version: int = 1
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