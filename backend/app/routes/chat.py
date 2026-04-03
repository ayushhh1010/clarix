"""
Chat routes — conversational RAG-powered chat with streaming support.
All routes are scoped to the currently authenticated user.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Repository, Conversation, Message, User
from app.schemas import ChatRequest, ChatResponse, ConversationResponse, MessageResponse, PaginatedResponse
from app.memory.manager import MemoryManager
from app.rag.retriever import retrieve_context
from app.rag.prompt_builder import build_messages
from app.rag.llm import generate, generate_stream
from app.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["Chat"])


# ── Helper: verify repo belongs to user ──────────────────────

async def _get_user_repo(db: AsyncSession, repo_id: str, user: User) -> Repository:
    """Fetch a repo and verify it belongs to the current user."""
    repo = await db.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    if repo.user_id is not None and repo.user_id != user.id:
        raise HTTPException(status_code=404, detail="Repository not found")
    return repo


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Ask a question about a repository. Returns a complete response with sources.
    For streaming, use POST /api/chat/stream instead.
    """
    # Validate repo ownership
    repo = await _get_user_repo(db, request.repo_id, user)
    if repo.status != "ready":
        raise HTTPException(status_code=400, detail=f"Repository not ready (status: {repo.status})")

    memory = MemoryManager(db, request.repo_id)

    # Get or create conversation (scoped to user)
    conv = await memory.get_or_create_conv(
        conversation_id=request.conversation_id,
        title=request.message[:100],
        user_id=user.id,
    )

    # Save user message
    await memory.save_user_message(conv.id, request.message)

    # Load conversation history for context
    history = await memory.load_conversation(conv.id, limit=20)

    # Retrieve relevant code chunks
    chunks = retrieve_context(request.message, request.repo_id)

    # Load long-term facts
    facts = await memory.recall_facts()
    additional = ""
    if facts:
        additional = "Previously learned facts about this codebase:\n" + "\n".join(f"- {f}" for f in facts[:10])

    # Build prompt and generate
    messages = build_messages(request.message, chunks, history, additional)
    response_text = generate(messages)

    # Build sources list
    sources = [c.to_dict() for c in chunks[:5]]

    # Save assistant response
    msg = await memory.save_assistant_message(conv.id, response_text, sources)
    await db.commit()

    return ChatResponse(
        conversation_id=conv.id,
        message_id=msg.id,
        content=response_text,
        sources=sources,
    )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Ask a question with Server-Sent Events (SSE) streaming response.
    """
    # Validate repo ownership
    repo = await _get_user_repo(db, request.repo_id, user)
    if repo.status != "ready":
        raise HTTPException(status_code=400, detail=f"Repository not ready (status: {repo.status})")

    memory = MemoryManager(db, request.repo_id)

    conv = await memory.get_or_create_conv(
        conversation_id=request.conversation_id,
        title=request.message[:100],
        user_id=user.id,
    )

    await memory.save_user_message(conv.id, request.message)
    history = await memory.load_conversation(conv.id, limit=20)

    chunks = retrieve_context(request.message, request.repo_id)
    sources = [c.to_dict() for c in chunks[:5]]

    facts = await memory.recall_facts()
    additional = ""
    if facts:
        additional = "Previously learned facts about this codebase:\n" + "\n".join(f"- {f}" for f in facts[:10])

    messages = build_messages(request.message, chunks, history, additional)

    async def event_generator():
        full_response = []

        # Send conversation ID first
        yield f"data: {json.dumps({'type': 'metadata', 'conversation_id': conv.id, 'sources': sources})}\n\n"

        # Stream LLM response chunks
        async for chunk in generate_stream(messages):
            full_response.append(chunk)
            yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

        # Save the complete response
        complete = "".join(full_response)
        await memory.save_assistant_message(conv.id, complete, sources)
        await db.commit()

        # Send completion signal
        yield f"data: {json.dumps({'type': 'done', 'message_id': 'saved'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{conversation_id}/history")
async def get_chat_history(
    conversation_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=100, description="Items per page"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[MessageResponse]:
    """Get the message history of a conversation with pagination."""
    conv = await db.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    # Verify ownership
    if conv.user_id is not None and conv.user_id != user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Get total count
    count_result = await db.execute(
        select(func.count(Message.id))
        .where(Message.conversation_id == conversation_id)
    )
    total = count_result.scalar() or 0

    # Get paginated results
    offset = (page - 1) * per_page
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .offset(offset)
        .limit(per_page)
    )
    items = result.scalars().all()

    return PaginatedResponse(
        items=[MessageResponse.model_validate(m) for m in items],
        total=total,
        page=page,
        per_page=per_page,
        has_more=(offset + len(items)) < total,
    )


@router.get("/conversations/{repo_id}")
async def list_conversations(
    repo_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[ConversationResponse]:
    """List all conversations for a repository, scoped to current user."""
    # Verify repo ownership first
    await _get_user_repo(db, repo_id, user)

    # Get total count
    count_result = await db.execute(
        select(func.count(Conversation.id))
        .where(
            Conversation.repo_id == repo_id,
            Conversation.user_id == user.id,
        )
    )
    total = count_result.scalar() or 0

    # Get paginated results
    offset = (page - 1) * per_page
    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.repo_id == repo_id,
            Conversation.user_id == user.id,
        )
        .order_by(Conversation.updated_at.desc())
        .offset(offset)
        .limit(per_page)
    )
    items = result.scalars().all()

    return PaginatedResponse(
        items=[ConversationResponse.model_validate(c) for c in items],
        total=total,
        page=page,
        per_page=per_page,
        has_more=(offset + len(items)) < total,
    )


@router.patch("/conversations/{conversation_id}")
async def rename_conversation(
    conversation_id: str,
    request: dict,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Rename a conversation."""
    conv = await db.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv.user_id is not None and conv.user_id != user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    title = request.get("title", "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")
    
    conv.title = title[:512]
    await db.flush()
    await db.commit()
    return {"message": "Conversation renamed"}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a conversation and all its messages."""
    conv = await db.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv.user_id is not None and conv.user_id != user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    await db.delete(conv)
    await db.commit()
    return {"message": "Conversation deleted"}
