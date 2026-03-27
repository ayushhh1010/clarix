"""
Chat routes — conversational RAG-powered chat with streaming support.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Repository, Conversation, Message
from app.schemas import ChatRequest, ChatResponse, ConversationResponse, MessageResponse
from app.memory.manager import MemoryManager
from app.rag.retriever import retrieve_context
from app.rag.prompt_builder import build_messages
from app.rag.llm import generate, generate_stream

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Ask a question about a repository. Returns a complete response with sources.
    For streaming, use POST /api/chat/stream instead.
    """
    # Validate repo
    repo = await db.get(Repository, request.repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    if repo.status != "ready":
        raise HTTPException(status_code=400, detail=f"Repository not ready (status: {repo.status})")

    memory = MemoryManager(db, request.repo_id)

    # Get or create conversation
    conv = await memory.get_or_create_conv(
        conversation_id=request.conversation_id,
        title=request.message[:100],
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
    db: AsyncSession = Depends(get_db),
):
    """
    Ask a question with Server-Sent Events (SSE) streaming response.
    """
    # Validate repo
    repo = await db.get(Repository, request.repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    if repo.status != "ready":
        raise HTTPException(status_code=400, detail=f"Repository not ready (status: {repo.status})")

    memory = MemoryManager(db, request.repo_id)

    conv = await memory.get_or_create_conv(
        conversation_id=request.conversation_id,
        title=request.message[:100],
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


@router.get("/{conversation_id}/history", response_model=list[MessageResponse])
async def get_chat_history(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get the full message history of a conversation."""
    conv = await db.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    return result.scalars().all()


@router.get("/conversations/{repo_id}", response_model=list[ConversationResponse])
async def list_conversations(
    repo_id: str,
    db: AsyncSession = Depends(get_db),
):
    """List all conversations for a repository."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.repo_id == repo_id)
        .order_by(Conversation.updated_at.desc())
    )
    return result.scalars().all()
