"""
FastAPI application entry point.
Production-grade setup with CORS, lifespan events, structured logging,
rate limiting, and health checks.
"""

# The GIT_PYTHON_REFRESH workaround that used to live here is gone with
# GitPython: app/indexing/source.py shells out to git directly, so there is
# no module-level binary discovery to silence.
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.database import close_db, init_db
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
    logger.info("   Index ver.  : %s", settings.index_version)

    # Initialize database tables
    await init_db()
    logger.info("✅ Database initialized")

    # LLM providers: only those with a key are constructed, so the
    # router's candidate list is exactly what is configured.
    from app.llm import BreakerRegistry, LLMRouter, build_providers
    from app.retrieval.query_embedder import build_query_embedder

    providers = build_providers(settings)
    app.state.llm_router = LLMRouter(providers, BreakerRegistry())
    app.state.query_embedder = build_query_embedder(settings)
    logger.info(
        "LLM providers: %s",
        ", ".join(p.name for p in providers) or "none (generation will degrade "
        "to returning citations)",
    )
    logger.info(
        "Query embedding: %s",
        settings.embedding_endpoint or "not configured (dense arm disabled)",
    )

    yield

    # Shutdown
    logger.info("Shutting down")
    await close_db()
    logger.info("Shutdown complete")


# ── Application ─────────────────────────────────────────────

app = FastAPI(
    title="Clarix",
    description=(
        "Repository-aware code intelligence: AST-grounded indexing, routed "
        "hybrid retrieval over pgvector, and quota-aware generation across "
        "free-tier providers."
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

from app.routes.agent import router as agent_router
from app.routes.auth import router as auth_router
from app.routes.chat import router as chat_router
from app.routes.repo import router as repo_router

app.include_router(auth_router)
app.include_router(repo_router)
app.include_router(chat_router)
app.include_router(agent_router)


# ── Health Check ─────────────────────────────────────────────

@app.api_route("/health", methods=["GET", "HEAD"], tags=["System"])
async def health_check():
    """
    Liveness plus the state of every dependency that can degrade.

    Circuit breaker state is included because a provider being tripped is
    the difference between a healthy service and one silently answering
    from its last fallback.
    """
    return {
        "status": "healthy",
        "service": "Clarix",
        "version": "2.0.0",
        "environment": settings.app_env,
        "llm": app.state.llm_router.health(),
        "query_embedding": getattr(app.state.query_embedder, "available", False),
    }


@app.api_route("/", methods=["GET", "HEAD"], tags=["System"])
async def root():
    """API root — redirect to docs."""
    return {
        "message": "Clarix API",
        "docs": "/docs",
        "health": "/health",
    }