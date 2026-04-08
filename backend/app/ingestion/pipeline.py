"""
Ingestion pipeline orchestrator — clone → parse → chunk → embed → store.
"""

import asyncio
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.ingestion.cloner import clone_repo
from app.ingestion.parser import parse_repository
from app.ingestion.chunker import chunk_repository
from app.ingestion.embedder import embed_chunks
from app.ingestion.vectorstore import store_chunks

logger = logging.getLogger(__name__)
settings = get_settings()


class IngestionResult:
    """Result of an ingestion pipeline run."""

    def __init__(self) -> None:
        self.repo_name: str = ""
        self.local_path: str = ""
        self.file_count: int = 0
        self.chunk_count: int = 0
        self.success: bool = False
        self.error: Optional[str] = None


async def _mark_repo_failed(repo_id: str, error_msg: str) -> None:
    """
    Use a completely fresh DB session to mark the repo as failed.
    This guarantees the status update is persisted even when the
    caller's session is corrupted (e.g. overlapping commits).
    """
    from app.database import async_session_factory
    from app.models import Repository

    try:
        async with async_session_factory() as fresh_db:
            repo = await fresh_db.get(Repository, repo_id)
            if repo:
                repo.status = "failed"
                repo.error_message = error_msg[:2000]
                repo.ingestion_progress = 0
                await fresh_db.commit()
                logger.info("[%s] Marked repo as failed via fresh session", repo_id)
    except Exception as mark_exc:
        logger.error(
            "[%s] CRITICAL: Could not mark repo as failed: %s", repo_id, mark_exc
        )


async def run_ingestion_pipeline(
    repo_id: str,
    url: str,
    db: AsyncSession,
) -> IngestionResult:
    """
    Full ingestion pipeline:
    1. Clone the repository
    2. Parse all source files
    3. Chunk source files
    4. Generate embeddings (async concurrent + cached)
    5. Store in vector database
    6. Update repository record in PostgreSQL

    This is run as a background task from the API.
    """
    from app.models import Repository

    result = IngestionResult()

    # Update status to ingesting
    repo = await db.get(Repository, repo_id)
    if not repo:
        result.error = f"Repository {repo_id} not found in DB"
        return result

    repo.status = "ingesting"
    repo.ingestion_progress = 0
    repo.ingestion_total_chunks = 0
    repo.ingestion_cached_chunks = 0
    repo.ingestion_phase = "clone"
    await db.commit()

    # Lock to serialise progress_callback DB commits —
    # embed_chunks fires many concurrent batches via asyncio.gather,
    # and each one calls progress_callback. Without the lock multiple
    # commit() calls race on the same session, causing
    # IllegalStateChangeError.
    _progress_lock = asyncio.Lock()

    try:
        # 1. Clone
        logger.info("[%s] Step 1/5: Cloning %s", repo_id, url)
        repo_name, local_path = clone_repo(url, repo_id)
        result.repo_name = repo_name
        result.local_path = local_path

        repo.name = repo_name
        repo.local_path = local_path
        repo.ingestion_phase = "parse"
        await db.commit()

        # 2. Parse
        logger.info("[%s] Step 2/5: Parsing files", repo_id)
        parsed_files = list(parse_repository(local_path))
        result.file_count = len(parsed_files)
        logger.info("[%s] Parsed %d files", repo_id, result.file_count)

        # 3. Chunk
        logger.info("[%s] Step 3/5: Chunking code", repo_id)
        chunks = chunk_repository(parsed_files, repo_id)
        result.chunk_count = len(chunks)
        logger.info("[%s] Produced %d chunks", repo_id, result.chunk_count)

        # Update total chunks count in DB for frontend progress display
        repo.ingestion_total_chunks = result.chunk_count
        repo.ingestion_phase = "embed"
        await db.commit()

        # 4. Embed (async concurrent with caching)
        logger.info("[%s] Step 4/5: Generating embeddings", repo_id)

        async def progress_callback(embedded_so_far: int, total: int, cache_hits: int):
            """Update progress in DB so frontend can poll it."""
            async with _progress_lock:
                pct = int((embedded_so_far / total) * 100) if total > 0 else 0
                repo.ingestion_progress = min(pct, 99)  # cap at 99 until fully stored
                repo.ingestion_cached_chunks = cache_hits
                await db.commit()

        embeddings = await embed_chunks(
            chunks,
            repo_id=repo_id,
            progress_callback=progress_callback,
        )

        # 5. Store
        logger.info("[%s] Step 5/5: Storing in vector DB", repo_id)
        repo.ingestion_phase = "store"
        await db.commit()

        stored = store_chunks(repo_id, chunks, embeddings)

        # Update repo record — completed
        repo.status = "ready"
        repo.file_count = result.file_count
        repo.chunk_count = stored
        repo.ingestion_progress = 100
        repo.ingestion_phase = "done"
        await db.commit()

        result.success = True
        logger.info(
            "[%s] Ingestion complete — %d files, %d chunks",
            repo_id,
            result.file_count,
            stored,
        )

    except Exception as exc:
        logger.exception("[%s] Ingestion failed: %s", repo_id, exc)
        result.error = str(exc)
        result.success = False

        # Try updating status on the current session first
        try:
            repo.status = "failed"
            repo.error_message = str(exc)[:2000]
            repo.ingestion_progress = 0
            await db.commit()
        except Exception:
            logger.warning(
                "[%s] Main session commit failed, using fresh session", repo_id
            )
            # Session is corrupted — use a fresh one to guarantee the update
            await _mark_repo_failed(repo_id, str(exc))

    return result
