"""
Repository routes — upload / clone repos, check status, list files.
"""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Repository
from app.schemas import RepoUploadRequest, RepoResponse, RepoFileNode

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/repo", tags=["Repository"])


@router.post("/upload", response_model=RepoResponse, status_code=202)
async def upload_repo(
    request: RepoUploadRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Clone a Git repository and start the ingestion pipeline.
    Returns immediately with status 'pending'. Ingestion runs in background.
    """
    repo_id = str(uuid.uuid4())

    repo = Repository(
        id=repo_id,
        name=request.url.split("/")[-1].replace(".git", ""),
        url=request.url,
        local_path="",
        status="pending",
    )
    db.add(repo)
    await db.commit()
    await db.refresh(repo)

    # Schedule background ingestion
    background_tasks.add_task(_run_background_ingestion, repo_id, request.url)

    logger.info("Repo %s queued for ingestion: %s", repo_id, request.url)
    return repo


async def _run_background_ingestion(repo_id: str, url: str):
    """Run ingestion in background with its own DB session."""
    from app.database import async_session_factory
    from app.ingestion.pipeline import run_ingestion_pipeline

    async with async_session_factory() as db:
        try:
            await run_ingestion_pipeline(repo_id, url, db)
            await db.commit()
        except Exception as exc:
            logger.exception("Background ingestion failed for repo %s: %s", repo_id, exc)
            await db.rollback()


@router.get("/{repo_id}", response_model=RepoResponse)
async def get_repo(
    repo_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get repository details and ingestion status."""
    repo = await db.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    return repo


@router.get("/{repo_id}/status")
async def get_repo_status(
    repo_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Check ingestion status for a repository."""
    repo = await db.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    return {
        "id": repo.id,
        "status": repo.status,
        "file_count": repo.file_count,
        "chunk_count": repo.chunk_count,
        "error_message": repo.error_message,
    }


@router.get("/{repo_id}/files", response_model=list[RepoFileNode])
async def get_repo_files(
    repo_id: str,
    path: str = "",
    db: AsyncSession = Depends(get_db),
):
    """List files and directories in the ingested repository."""
    repo = await db.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    if repo.status != "ready":
        raise HTTPException(status_code=400, detail=f"Repository not ready (status: {repo.status})")

    base = Path(repo.local_path) / path
    if not base.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")

    skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv"}
    nodes: list[RepoFileNode] = []

    try:
        for entry in sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if entry.name in skip_dirs:
                continue
            rel = str(entry.relative_to(Path(repo.local_path))).replace("\\", "/")
            nodes.append(RepoFileNode(
                name=entry.name,
                path=rel,
                type="directory" if entry.is_dir() else "file",
            ))
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")

    return nodes


@router.get("/", response_model=list[RepoResponse])
async def list_repos(
    db: AsyncSession = Depends(get_db),
):
    """List all repositories."""
    result = await db.execute(
        select(Repository).order_by(Repository.created_at.desc())
    )
    return result.scalars().all()


@router.delete("/{repo_id}")
async def delete_repo(
    repo_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a repository and its associated data."""
    repo = await db.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    # Clean up vector store
    from app.ingestion.vectorstore import delete_collection
    delete_collection(repo_id)

    # Clean up files on disk
    from app.ingestion.cloner import remove_repo
    remove_repo(repo_id)

    await db.delete(repo)
    await db.commit()
    return {"message": f"Repository {repo_id} deleted successfully"}
