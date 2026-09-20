"""
Agent routes.

THE v1 AGENT WAS REMOVED, NOT PORTED.

Its LangGraph pipeline was planner -> retrieval -> tool agent -> executor,
three sequential LLM calls around one retrieval. Reading it showed the tool
path was unreachable: `needs_tools` was a substring match on the model's own
prose (`"need to read" in analysis.lower()`) gated behind `context_str == ""`,
and retrieval almost always returned something, so the six registered tools
never fired. The tool agent's own prompt said "Do NOT attempt to call any
tools". The graph had no cycles either -- `_route_after_planner` returned a
constant.

So it cost three LLM calls and several seconds of added latency to produce
what one call produces, and on free-tier quotas three calls per question is
the difference between answering ~200 questions a day and ~65. Porting a
design measured as net-negative would have been the wrong kind of
faithfulness.

These endpoints are kept because the frontend calls them. They now run the
same routed retrieval and generation as `/api/chat`, and the reported steps
describe what actually happened -- which arms ran, how much context was
packed, which provider answered -- rather than narrating agent roles that
no longer exist.

A real agent loop, with cycles, a relevance grader and genuine tool calls,
is worth building. It should be justified by the evaluation harness against
this single-pass baseline before it ships, which is exactly what v1 never
did.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app import conversations as convo
from app.database import get_db
from app.llm.types import AllProvidersFailed, ContextTooLong
from app.models import Repository, User
from app.retrieval import hybrid_search
from app.retrieval.context import build_messages, pack
from app.schemas import AgentRunRequest, AgentRunResponse, AgentStepResponse
from app.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agent", tags=["Agent"])

RETRIEVE_LIMIT = 20


async def _get_user_repo(db: AsyncSession, repo_id: str, user: User) -> Repository:
    repo = await db.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    if repo.user_id is not None and repo.user_id != user.id:
        raise HTTPException(status_code=404, detail="Repository not found")
    if repo.status != "ready":
        raise HTTPException(
            status_code=409, detail=f"Repository is not ready (status: {repo.status})"
        )
    return repo


async def _run(request: Request, db: AsyncSession, repo: Repository,
               user: User, payload: AgentRunRequest):
    """Execute the pipeline, returning (conversation, steps, answer)."""
    conversation = await convo.get_or_create_conversation(
        db, repo.id, user.id, payload.conversation_id, title=payload.task[:100]
    )
    await convo.save_message(db, conversation.id, "user", payload.task)
    await db.commit()

    embedder = request.app.state.query_embedder
    vectors = await embedder.embed_query(payload.task)
    if vectors is None:
        chunks, trace = await hybrid_search(
            db, repo.id, payload.task, query_bits="0", query_vector="[0]",
            limit=RETRIEVE_LIMIT, use_dense=False,
        )
    else:
        chunks, trace = await hybrid_search(
            db, repo.id, payload.task,
            query_bits=vectors.bits, query_vector=vectors.vector,
            limit=RETRIEVE_LIMIT,
        )

    steps = [
        AgentStepResponse(
            step="route",
            agent="router",
            content=(
                f"Classified as a {trace.route} query; ran "
                f"{', '.join(k for k, v in trace.arm_counts.items() if v > 0) or 'no'} "
                f"retrieval arm(s)."
            ),
        ),
        AgentStepResponse(
            step="retrieve",
            agent="retrieval",
            content=(
                f"Fused {trace.fused_count} candidates in "
                f"{trace.total_ms:.0f} ms. Arm yields: {trace.arm_counts}."
            ),
        ),
    ]

    packed = pack(chunks, question_tokens=len(payload.task) // 3)
    steps.append(AgentStepResponse(
        step="pack",
        agent="context",
        content=(
            f"Packed {len(packed.chunks)} of {len(chunks)} chunks into "
            f"{packed.used_tokens} tokens ({packed.dropped} did not fit)."
        ),
    ))

    messages = build_messages(payload.task, packed)
    router_ = request.app.state.llm_router
    try:
        completion, _ = await router_.complete(db, messages, max_tokens=1500)
        answer = completion.text
        detail = f"Answered by {completion.provider} in {completion.latency_ms:.0f} ms"
        if completion.degraded:
            detail += f" (after {', '.join(completion.fallbacks)} were unavailable)"
    except ContextTooLong as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except AllProvidersFailed as exc:
        logger.error("generation unavailable: %s", exc)
        answer = _citations_only(packed)
        detail = f"No provider available: {exc}"

    steps.append(AgentStepResponse(step="generate", agent="llm", content=detail))

    await convo.save_message(
        db, conversation.id, "assistant", answer,
        metadata={"sources": packed.citations, "route": trace.route},
    )
    await db.commit()
    return conversation, steps, answer, packed


def _citations_only(packed) -> str:
    if packed.empty:
        return "No language model is available, and no relevant code was found."
    lines = ["No language model is available. Most relevant code:", ""]
    lines += [f"- `{c.citation}`" for c in packed.chunks[:8]]
    return "\n".join(lines)


@router.post("/run", response_model=AgentRunResponse)
async def run_agent(
    payload: AgentRunRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    repo = await _get_user_repo(db, payload.repo_id, user)
    conversation, steps, answer, _ = await _run(request, db, repo, user, payload)
    return AgentRunResponse(
        conversation_id=conversation.id, steps=steps, final_answer=answer
    )


@router.post("/run/stream")
async def run_agent_stream(
    payload: AgentRunRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    repo = await _get_user_repo(db, payload.repo_id, user)

    async def events():
        conversation, steps, answer, packed = await _run(
            request, db, repo, user, payload
        )
        for step in steps:
            yield _sse("step", step.model_dump())
        yield _sse("sources", {"sources": packed.citations})
        for i in range(0, len(answer), 120):
            yield _sse("token", {"text": answer[i : i + 120]})
        yield _sse("done", {"conversation_id": conversation.id})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
