"""
Application configuration using Pydantic Settings.
Loads from environment variables and .env file.
"""

from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Google (OAuth only — Gemini removed) ──────────────────
    google_api_key: str = ""

    # ── Groq API (production LLM) ────────────────────────────
    groq_api_key: str = ""

    # ── Ollama (local development LLM) ───────────────────────
    ollama_model: str = "llama3.1"

    # ── PostgreSQL ──────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://clarix:clarix_secret@localhost:5433/clarix_db"

    # ── Redis ───────────────────────────────────────────────
    redis_url: str = "redis://localhost:6380/0"

    # ── ChromaDB ────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma"

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

    # ── Embedding (HuggingFace Inference API, 384-dim) ─────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    huggingface_api_token: str = ""

    # ── LLM ─────────────────────────────────────────────────
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # ── RAG ─────────────────────────────────────────────────
    rag_top_k: int = 10
    rag_context_max_tokens: int = 12000

    # ── Chunking ────────────────────────────────────────────
    chunk_size_lines: int = 50
    chunk_overlap_lines: int = 10

    @property
    def repos_path(self) -> Path:
        path = Path(self.repos_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def chroma_path(self) -> Path:
        path = Path(self.chroma_persist_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache()
def get_settings() -> Settings:
    return Settings()