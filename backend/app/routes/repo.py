"""
Repository routes.

Rewritten onto the v2 indexing path. Three behavioural changes, each fixing
something measured or observed:

  Queued, not backgrounded. v1 used FastAPI `BackgroundTasks`, which dies
  with the process: a spin-down mid-ingest left the repository in
  `status='ingesting'` for ever, with no retry. The row and its job are now
  inserted in one transaction, so neither can exist without the other, and
  a worker that dies has its job reclaimed.

  The URL is validated at submission. v1 accepted any string and discovered
  problems inside the background task, where the user never saw them.
  `ext::sh -c ...` is a remote code execution primitive, so it is rejected
  at the API boundary with a 400 rather than deep in a worker.

  File content comes from the forge, pinned to the indexed commit. v1 read
  from `local_path` on the worker's disk; on an ephemeral filesystem that
  path was dangling after every restart, so the viewer 404'd for
  repositories the API simultaneously reported as `ready`. Nothing is
  stored, and what is displayed is exactly what was indexed.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.content import ContentError, build_tree, fetch_file
from app.database import get_db
from app.indexing import queue
from app.indexing.source import UnsafeSourceError, repo_name_from_url, validate_url
from app.models import Repository, User
from app.schemas import (
    PaginatedResponse,
    RepoFileNode,
    RepoResponse,
    RepoUploadRequest,
)
from app.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/repo", tags=["Repository"])


async def _get_user_repo(db: AsyncSession, repo_id: str, user: User) -> Repository:
    """
    Fetch a repository the caller owns.

    A repository belonging to someone else returns 404, not 403: a 403
    confirms the id exists, which is an enumeration oracle.
    """
    repo = await db.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    if repo.user_id is not None and repo.user_id != user.id:
        raise HTTPException(status_code=404, detail="Repository not found")
    return repo


@router.post("/upload", response_model=RepoResponse, status_code=202)
async def upload_repo(
    request: RepoUploadRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Register a repository and queue it for indexing."""
    try:
        url = validate_url(request.url)
    except UnsafeSourceError as exc:
        # Fail here, where the user can read it, not inside a worker.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    existing = await db.execute(
        select(Repository).where(
            Repository.user_id == user.id, Repository.url == url
        )
    )
    already = existing.scalars().first()
    if already:
        await queue.enqueue(db, already.id, "full_index", {"ref": None})
        await db.commit()
        logger.info("repo %s re-queued for user %s", already.id, user.id)
        return already

    repo = Repository(
        id=str(uuid.uuid4()),
        user_id=user.id,
        name=repo_name_from_url(url),
        url=url,
        local_path="",  # v2 keeps no checkout; retained for schema compatibility
        status="pending",
    )
    db.add(repo)
    await db.flush()

    # Same transaction as the row above: a repository can never exist
    # without its job, and a job can never reference a rolled-back
    # repository.
    await queue.enqueue(db, repo.id, "full_index", {"ref": None})
    await db.commit()
    await db.refresh(repo)

    logger.info("repo %s queued by user %s: %s", repo.id, user.id, url)
    return repo


@router.post("/{repo_id}/reindex", status_code=202)
async def reindex_repo(
    repo_id: str,
    force: bool = Query(False, description="Reindex even if the commit is unchanged"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Queue a re-index. Deduplicated against any job already pending."""
    repo = await _get_user_repo(db, repo_id, user)
    job_id = await queue.enqueue(db, repo.id, "full_index", {"force": force})
    await db.commit()
    return {
        "repo_id": repo.id,
        "queued": job_id is not None,
        "detail": "queued" if job_id else "a job is already pending for this repository",
    }


@router.get("/{repo_id}", response_model=RepoResponse)
async def get_repo(
    repo_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _get_user_repo(db, repo_id, user)


@router.get("/{repo_id}/status")
async def get_repo_status(
    repo_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Indexing status, including the queue state.

    v1 reported only a progress percentage written by the in-process task,
    which stopped updating the moment that process died and left no way to
    tell a running ingest from a dead one. The job row is the truth.
    """
    repo = await _get_user_repo(db, repo_id, user)
    job = (
        await db.execute(
            text(
                "SELECT status, attempts, max_attempts, last_error, run_after "
                "FROM ingest_jobs WHERE repo_id = :i "
                "ORDER BY created_at DESC LIMIT 1"
            ),
            {"i": repo_id},
        )
    ).first()

    return {
        "repo_id": repo.id,
        "status": repo.status,
        "progress": repo.ingestion_progress,
        "phase": repo.ingestion_phase,
        "chunk_count": repo.chunk_count,
        "indexed_commit": repo.indexed_commit_sha,
        "last_indexed_at": repo.last_indexed_at,
        "error_message": repo.error_message,
        "job": None if job is None else {
            "status": job.status,
            "attempts": job.attempts,
            "max_attempts": job.max_attempts,
            "last_error": job.last_error,
            "next_attempt_at": job.run_after,
        },
    }


@router.get("/{repo_id}/files", response_model=list[RepoFileNode])
async def get_repo_files(
    repo_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    The indexed file tree.

    Built from the chunks that exist, so the viewer can only offer files
    that are genuinely searchable -- v1 listed the checkout, which could
    disagree with the index in either direction.
    """
    await _get_user_repo(db, repo_id, user)
    rows = await db.execute(
        text("SELECT DISTINCT file_path FROM chunks WHERE repo_id = :i "
             "ORDER BY file_path"),
        {"i": repo_id},
    )
    return build_tree([r.file_path for r in rows])


@router.get("/{repo_id}/file-content")
async def get_file_content(
    repo_id: str,
    path: str = Query(..., description="Repository-relative file path"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Fetch one file at the exact commit that was indexed."""
    repo = await _get_user_repo(db, repo_id, user)
    if not repo.url:
        raise HTTPException(status_code=409, detail="Repository has no source URL")
    if not repo.indexed_commit_sha:
        raise HTTPException(
            status_code=409,
            detail="Repository has not finished indexing yet",
        )

    # Only serve paths the index knows about: the path is user-supplied and
    # goes into an outbound URL, and this makes the index the allowlist.
    known = (
        await db.execute(
            text("SELECT 1 FROM chunks WHERE repo_id = :i AND file_path = :p LIMIT 1"),
            {"i": repo_id, "p": path},
        )
    ).first()
    if known is None:
        raise HTTPException(status_code=404, detail=f"{path} is not in the index")

    try:
        content = await fetch_file(repo.url, repo.indexed_commit_sha, path)
    except ContentError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "path": content.path,
        "content": content.text,
        "truncated": content.truncated,
        "ref": content.ref,
    }


@router.get("/")
async def list_repos(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[RepoResponse]:
    """List the caller's repositories."""
    total = (
        await db.execute(
            select(func.count(Repository.id)).where(Repository.user_id == user.id)
        )
    ).scalar() or 0

    offset = (page - 1) * per_page
    result = await db.execute(
        select(Repository)
        .where(Repository.user_id == user.id)
        .order_by(Repository.created_at.desc())
        .offset(offset)
        .limit(per_page)
    )
    items = result.scalars().all()

    return PaginatedResponse(
        items=[RepoResponse.model_validate(r) for r in items],
        total=total,
        page=page,
        per_page=per_page,
        has_more=(offset + len(items)) < total,
    )


@router.delete("/{repo_id}")
async def delete_repo(
    repo_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a repository and everything indexed from it.

    Chunks, jobs and conversations are removed by `ON DELETE CASCADE` in one
    transaction. v1 deleted a ChromaDB collection and a file-based cache
    separately from the row, so a partial failure left orphans in whichever
    store the error missed.
    """
    repo = await _get_user_repo(db, repo_id, user)
    await db.delete(repo)
    await db.commit()
    logger.info("repo %s deleted by user %s", repo_id, user.id)
    return {"deleted": repo_id}
