"""
Chat routes: retrieve, pack, generate.

Rewritten onto the v2 pipeline. The API surface is unchanged -- the
frontend is not touched -- but everything behind it is different:

  Retrieval is routed. `hybrid_search` picks its configuration per query.
  Measured on held-out test splits (BENCHMARKS.md section 7): identifier
  lookups go from recall@1 0.680 to 0.937 against the dense-only baseline
  v1 used.

  The context budget is real. v1 packed to 12,000 estimated tokens using
  `len(text)//4`. That exceeds Cerebras' 8,192-token free-tier cap and
  Groq's 8,000 tokens/minute, so it does not merely cost more -- it fails.
  Packing is now counted with the real tokenizer against a 4,000-token
  budget.

  Generation fails over. One provider's daily allowance cannot carry the
  service, so `LLMRouter` tracks each provider's remaining quota and routes
  around the exhausted ones.

  Degradation is visible. If no provider answers, the endpoint returns the
  retrieved code with citations and `degraded: true` rather than a 500: the
  citations are the part the user can still act on.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import conversations as convo
from app.database import get_db
from app.llm.types import AllProvidersFailed, ContextTooLong
from app.models import Conversation, Message, Repository, User
from app.retrieval import hybrid_search
from app.retrieval.context import build_messages, pack
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    MessageResponse,
    PaginatedResponse,
)
from app.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["Chat"])

RETRIEVE_LIMIT = 20


async def _get_user_repo(db: AsyncSession, repo_id: str, user: User) -> Repository:
    repo = await db.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    if repo.user_id is not None and repo.user_id != user.id:
        raise HTTPException(status_code=404, detail="Repository not found")
    return repo


async def _retrieve(request: Request, db: AsyncSession, repo_id: str, question: str):
    """Embed the query if possible, then run routed retrieval."""
    embedder = request.app.state.query_embedder
    vectors = await embedder.embed_query(question)

    # No embedding endpoint, or it is down: drop the dense arm rather than
    # the whole request. Lexical and symbol run inside Postgres alone.
    if vectors is None:
        return await hybrid_search(
            db, repo_id, question,
            query_bits="0", query_vector="[0]",
            limit=RETRIEVE_LIMIT, use_dense=False,
        )
    return await hybrid_search(
        db, repo_id, question,
        query_bits=vectors.bits, query_vector=vectors.vector,
        limit=RETRIEVE_LIMIT,
    )


async def _prepare(request: Request, db: AsyncSession, repo: Repository,
                   user: User, payload: ChatRequest):
    """Shared setup for the buffered and streaming endpoints."""
    if repo.status != "ready":
        raise HTTPException(
            status_code=409,
            detail=f"Repository is not ready (status: {repo.status})",
        )

    try:
        conversation = await convo.get_or_create_conversation(
            db, repo.id, user.id, payload.conversation_id, title=payload.message[:100]
        )
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc

    await convo.save_message(db, conversation.id, "user", payload.message)
    history = [
        (m.role, m.content)
        for m in await convo.load_history(db, conversation.id)
        if m.role in ("user", "assistant")
    ][:-1]  # the message just saved is the question, not history

    chunks, trace = await _retrieve(request, db, repo.id, payload.message)
    packed = pack(
        chunks,
        question_tokens=len(payload.message) // 3,
        history_tokens=sum(len(c) for _, c in history) // 3,
    )
    messages = build_messages(payload.message, packed, history)
    await db.commit()
    return conversation, packed, messages, trace


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    repo = await _get_user_repo(db, payload.repo_id, user)
    conversation, packed, messages, trace = await _prepare(
        request, db, repo, user, payload
    )

    router_ = request.app.state.llm_router
    degraded_reason = ""
    try:
        completion, llm_route = await router_.complete(db, messages)
        answer = completion.text
        provider = completion.provider
        attempted = llm_route.summary()
    except ContextTooLong as exc:
        raise HTTPException(
            status_code=413,
            detail=f"The question plus retrieved code exceeds the model's limit: {exc}",
        ) from exc
    except AllProvidersFailed as exc:
        # Return what retrieval found. The citations are the part the user
        # can still act on, and a 500 would throw them away.
        logger.error("generation unavailable: %s", exc)
        answer = _no_generation_message(packed)
        provider = "none"
        attempted = str(exc)
        degraded_reason = str(exc)

    message = await convo.save_message(
        db, conversation.id, "assistant", answer,
        metadata={
            "sources": packed.citations,
            "route": trace.route,
            "provider": provider,
            "llm_attempts": attempted,
            "degraded": bool(degraded_reason),
        },
    )
    await db.commit()

    return ChatResponse(
        conversation_id=conversation.id,
        message_id=message.id,
        content=answer,
        sources=packed.citations,
    )


def _no_generation_message(packed) -> str:
    if packed.empty:
        return (
            "No language model is available right now, and retrieval found no "
            "relevant code for this question."
        )
    lines = [
        "No language model is available right now, so this answer could not be "
        "written. These are the most relevant places in the code:",
        "",
    ]
    lines += [
        f"- `{c.citation}`" + (f" — `{c.symbol}`" if c.symbol else "")
        for c in packed.chunks[:8]
    ]
    return "\n".join(lines)


@router.post("/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Server-sent events.

    Sources are emitted before the answer so the UI can show citations
    while generation is still running -- and so they survive a generation
    failure, which is the case v1 handled by sending nothing.
    """
    repo = await _get_user_repo(db, payload.repo_id, user)
    conversation, packed, messages, trace = await _prepare(
        request, db, repo, user, payload
    )
    router_ = request.app.state.llm_router

    async def events():
        yield _sse("sources", {
            "conversation_id": conversation.id,
            "sources": packed.citations,
            "route": trace.route,
            "context_tokens": packed.used_tokens,
        })

        degraded = False
        try:
            completion, _route = await router_.complete(db, messages)
            answer = completion.text
            provider = completion.provider
            if completion.degraded:
                yield _sse("notice", {
                    "message": f"answered by {provider} after "
                               f"{', '.join(completion.fallbacks)} were unavailable",
                })
        except ContextTooLong as exc:
            answer = f"The question plus retrieved code exceeds the model's limit: {exc}"
            provider, degraded = "none", True
        except AllProvidersFailed as exc:
            logger.error("generation unavailable: %s", exc)
            answer = _no_generation_message(packed)
            provider, degraded = "none", True

        # Chunked rather than token-by-token: the providers are called
        # without streaming, so pretending to stream tokens would be
        # theatre. This keeps the client's incremental rendering working.
        for i in range(0, len(answer), 120):
            yield _sse("token", {"text": answer[i : i + 120]})

        message = await convo.save_message(
            db, conversation.id, "assistant", answer,
            metadata={"sources": packed.citations, "route": trace.route,
                      "provider": provider, "degraded": degraded},
        )
        await db.commit()
        yield _sse("done", {
            "conversation_id": conversation.id,
            "message_id": message.id,
            "provider": provider,
            "degraded": degraded,
        })

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# --- conversation management (unchanged behaviour) -------------------------

@router.get("/{conversation_id}/history")
async def get_chat_history(
    conversation_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[MessageResponse]:
    conversation = await db.get(Conversation, conversation_id)
    if not conversation or (
        conversation.user_id is not None and conversation.user_id != user.id
    ):
        raise HTTPException(status_code=404, detail="Conversation not found")

    total = await convo.count_messages(db, conversation_id)
    offset = (page - 1) * per_page
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
        .offset(offset)
        .limit(per_page)
    )
    items = result.scalars().all()
    return PaginatedResponse(
        items=[MessageResponse.model_validate(m) for m in items],
        total=total, page=page, per_page=per_page,
        has_more=(offset + len(items)) < total,
    )


@router.get("/conversations/{repo_id}")
async def list_conversations(
    repo_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[ConversationResponse]:
    await _get_user_repo(db, repo_id, user)

    total = (
        await db.execute(
            select(func.count(Conversation.id)).where(
                Conversation.repo_id == repo_id, Conversation.user_id == user.id
            )
        )
    ).scalar() or 0

    offset = (page - 1) * per_page
    result = await db.execute(
        select(Conversation)
        .where(Conversation.repo_id == repo_id, Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .offset(offset)
        .limit(per_page)
    )
    items = result.scalars().all()
    return PaginatedResponse(
        items=[ConversationResponse.model_validate(c) for c in items],
        total=total, page=page, per_page=per_page,
        has_more=(offset + len(items)) < total,
    )


@router.patch("/conversations/{conversation_id}")
async def rename_conversation(
    conversation_id: str,
    title: str = Query(..., min_length=1, max_length=512),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    conversation = await db.get(Conversation, conversation_id)
    if not conversation or (
        conversation.user_id is not None and conversation.user_id != user.id
    ):
        raise HTTPException(status_code=404, detail="Conversation not found")
    conversation.title = title
    await db.commit()
    return {"id": conversation.id, "title": conversation.title}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    conversation = await db.get(Conversation, conversation_id)
    if not conversation or (
        conversation.user_id is not None and conversation.user_id != user.id
    ):
        raise HTTPException(status_code=404, detail="Conversation not found")
    await db.delete(conversation)
    await db.commit()
    return {"deleted": conversation_id}
