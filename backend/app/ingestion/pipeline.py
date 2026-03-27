"""
Ingestion pipeline orchestrator — clone → parse → chunk → embed → store.
"""

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
    4. Generate embeddings
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
    await db.commit()

    try:
        # 1. Clone
        logger.info("[%s] Step 1/5: Cloning %s", repo_id, url)
        repo_name, local_path = clone_repo(url, repo_id)
        result.repo_name = repo_name
        result.local_path = local_path

        repo.name = repo_name
        repo.local_path = local_path
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

        # 4. Embed
        logger.info("[%s] Step 4/5: Generating embeddings", repo_id)
        embeddings = embed_chunks(chunks)

        # 5. Store
        logger.info("[%s] Step 5/5: Storing in vector DB", repo_id)
        stored = store_chunks(repo_id, chunks, embeddings)

        # Update repo record
        repo.status = "ready"
        repo.file_count = result.file_count
        repo.chunk_count = stored
        await db.commit()

        result.success = True
        logger.info(
            "[%s] Ingestion complete — %d files, %d chunks",
            repo_id, result.file_count, stored,
        )

    except Exception as exc:
        logger.exception("[%s] Ingestion failed: %s", repo_id, exc)
        result.error = str(exc)
        result.success = False
        repo.status = "failed"
        repo.error_message = str(exc)[:2000]
        await db.commit()

    return result
