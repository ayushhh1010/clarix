"""
FastAPI application entry point.
Production-grade setup with CORS, lifespan events, structured logging,
rate limiting, and health checks.
"""

# Must happen before ANY import that transitively pulls in gitpython.
# Render's runtime has git installed but it may not be on the PATH that
# Python resolves at import time. This silences the module-level check;
# the actual git binary is located explicitly inside clone_repo().
import os
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.database import init_db, close_db
from app.rate_limit import limiter

settings = get_settings()

# ── Structured Logging ──────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("clarix")


# ── Lifespan Events ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("🚀 Starting Clarix")
    logger.info("   Environment : %s", settings.app_env)
    logger.info("   LLM Model   : %s", settings.llm_model)
    logger.info("   Embed Model : %s", settings.embedding_model)

    # Initialize database tables
    await init_db()
    logger.info("✅ Database initialized")

    # Redis configured lazily — connects on first use
    logger.info("✅ Redis configured at %s", settings.redis_url)

    # Embeddings via HuggingFace Inference API — no local model loading
    logger.info("✅ Embeddings via HF Inference API (%s)", settings.embedding_model)

    yield

    # Shutdown
    logger.info("🛑 Shutting down...")
    await close_db()

    from app.memory.long_term import close_redis
    await close_redis()

    logger.info("👋 Shutdown complete")


# ── Application ─────────────────────────────────────────────

app = FastAPI(
    title="Clarix",
    description=(
        "Agentic AI backend for understanding codebases, answering technical questions, "
        "debugging issues, and suggesting code modifications. "
        "Powered by LangGraph, RAG, and GPT-4o."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Attach rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ─────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────

from app.routes.repo import router as repo_router
from app.routes.chat import router as chat_router
from app.routes.agent import router as agent_router
from app.routes.auth import router as auth_router

app.include_router(auth_router)
app.include_router(repo_router)
app.include_router(chat_router)
app.include_router(agent_router)


# ── Health Check ─────────────────────────────────────────────

@app.api_route("/health", methods=["GET", "HEAD"], tags=["System"])
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "Clarix",
        "version": "1.0.0",
        "environment": settings.app_env,
    }


@app.api_route("/", methods=["GET", "HEAD"], tags=["System"])
async def root():
    """API root — redirect to docs."""
    return {
        "message": "Clarix API",
        "docs": "/docs",
        "health": "/health",
    }