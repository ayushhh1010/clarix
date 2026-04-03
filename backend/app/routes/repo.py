"""
Repository routes — upload / clone repos, check status, list files.
All routes are scoped to the currently authenticated user.
"""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Repository, User
from app.schemas import RepoUploadRequest, RepoResponse, RepoFileNode, PaginatedResponse
from app.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/repo", tags=["Repository"])


# ── Helper: fetch repo owned by user ─────────────────────────

async def _get_user_repo(
    db: AsyncSession, repo_id: str, user: User
) -> Repository:
    """Fetch a repo and verify it belongs to the current user."""
    repo = await db.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    # Allow access if repo has no owner (legacy) or belongs to user
    if repo.user_id is not None and repo.user_id != user.id:
        raise HTTPException(status_code=404, detail="Repository not found")
    return repo


@router.post("/upload", response_model=RepoResponse, status_code=202)
async def upload_repo(
    request: RepoUploadRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Clone a Git repository and start the ingestion pipeline.
    Returns immediately with status 'pending'. Ingestion runs in background.
    """
    repo_id = str(uuid.uuid4())

    repo = Repository(
        id=repo_id,
        user_id=user.id,
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

    logger.info("Repo %s queued for ingestion by user %s: %s", repo_id, user.id, request.url)
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
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get repository details and ingestion status."""
    return await _get_user_repo(db, repo_id, user)


@router.get("/{repo_id}/status")
async def get_repo_status(
    repo_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Check ingestion status for a repository."""
    repo = await _get_user_repo(db, repo_id, user)
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
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List files and directories in the ingested repository."""
    repo = await _get_user_repo(db, repo_id, user)
    if repo.status != "ready":
        raise HTTPException(status_code=400, detail=f"Repository not ready (status: {repo.status})")

    # Gracefully handle missing local_path (e.g. after deploy/restart)
    if not repo.local_path:
        return []

    base = Path(repo.local_path) / path
    if not base.exists() or not base.is_dir():
        # Directory doesn't exist on disk — return empty instead of 404
        return []

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
    except OSError:
        # Filesystem error — return empty gracefully
        return []

    return nodes


# Language detection by extension
LANGUAGE_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".jsx": "javascript", ".java": "java", ".go": "go", ".rs": "rust",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp", ".cs": "csharp",
    ".rb": "ruby", ".php": "php", ".swift": "swift", ".kt": "kotlin",
    ".scala": "scala", ".r": "r", ".R": "r", ".sql": "sql",
    ".html": "html", ".css": "css", ".scss": "scss", ".less": "less",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".xml": "xml",
    ".md": "markdown", ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".ps1": "powershell", ".dockerfile": "dockerfile", ".toml": "toml",
}


@router.get("/{repo_id}/file-content")
async def get_file_content(
    repo_id: str,
    path: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the content of a specific file in the repository."""
    repo = await _get_user_repo(db, repo_id, user)
    if repo.status != "ready":
        raise HTTPException(status_code=400, detail=f"Repository not ready (status: {repo.status})")
    
    if not repo.local_path:
        raise HTTPException(status_code=404, detail="Repository files not available")
    
    # Security: Prevent path traversal
    from pathlib import Path as PathLib
    base_path = PathLib(repo.local_path).resolve()
    file_path = (base_path / path).resolve()
    
    if not str(file_path).startswith(str(base_path)):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Limit file size (5MB)
    if file_path.stat().st_size > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")
    
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")
    
    # Detect language from extension
    suffix = file_path.suffix.lower()
    language = LANGUAGE_MAP.get(suffix, "plaintext")
    
    return {"content": content, "language": language}


@router.get("/")
async def list_repos(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[RepoResponse]:
    """List all repositories owned by the current user with pagination."""
    # Get total count
    count_result = await db.execute(
        select(func.count(Repository.id))
        .where(Repository.user_id == user.id)
    )
    total = count_result.scalar() or 0

    # Get paginated results
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
    """Delete a repository and its associated data."""
    repo = await _get_user_repo(db, repo_id, user)

    # Clean up vector store
    from app.ingestion.vectorstore import delete_collection
    delete_collection(repo_id)

    # Clean up files on disk
    from app.ingestion.cloner import remove_repo
    remove_repo(repo_id)

    await db.delete(repo)
    await db.commit()
    return {"message": f"Repository {repo_id} deleted successfully"}
