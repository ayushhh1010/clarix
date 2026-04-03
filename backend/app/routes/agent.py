"""
Agent routes — run the full multi-agent workflow via LangGraph.
All routes are scoped to the currently authenticated user.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Repository, User
from app.schemas import AgentRunRequest, AgentRunResponse, AgentStepResponse
from app.memory.manager import MemoryManager
from app.agents.graph import get_compiled_graph
from app.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agent", tags=["Agent"])


# ── Helper: verify repo belongs to user ──────────────────────

async def _get_user_repo(db: AsyncSession, repo_id: str, user: User) -> Repository:
    """Fetch a repo and verify it belongs to the current user."""
    repo = await db.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    if repo.user_id is not None and repo.user_id != user.id:
        raise HTTPException(status_code=404, detail="Repository not found")
    return repo


@router.post("/run", response_model=AgentRunResponse)
async def run_agent(
    request: AgentRunRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Run the full multi-agent workflow (Planner → Retrieval → Tool → Executor).
    Returns the final answer along with all intermediate steps.
    """
    # Validate repo ownership
    repo = await _get_user_repo(db, request.repo_id, user)
    if repo.status != "ready":
        raise HTTPException(status_code=400, detail=f"Repository not ready (status: {repo.status})")

    memory = MemoryManager(db, request.repo_id)
    conv = await memory.get_or_create_conv(
        conversation_id=request.conversation_id,
        title=f"Agent: {request.task[:80]}",
        user_id=user.id,
    )

    # Save the user task as a message
    await memory.save_user_message(conv.id, request.task)

    # Build initial state
    initial_state = {
        "messages": [],
        "user_query": request.task,
        "repo_id": request.repo_id,
        "repo_path": repo.local_path,
        "plan": [],
        "retrieved_context": [],
        "tool_results": [],
        "final_answer": "",
        "current_step": 0,
        "needs_tools": False,
        "error": None,
        "steps_log": [],
    }

    # Run the compiled LangGraph
    graph = get_compiled_graph()

    try:
        result = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("Agent execution failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(exc)}")

    final_answer = result.get("final_answer", "No answer generated.")
    steps_log = result.get("steps_log", [])

    # Save assistant response
    sources = [
        {"file_path": c.get("file_path", ""), "relevance_score": c.get("relevance_score", 0)}
        for c in result.get("retrieved_context", [])[:5]
    ]
    await memory.save_assistant_message(conv.id, final_answer, sources)
    await db.commit()

    # Format steps for response
    agent_steps = []
    for step in steps_log:
        agent_steps.append(AgentStepResponse(
            step=str(step.get("agent", "unknown")),
            agent=str(step.get("agent", "unknown")),
            content=json.dumps(
                {k: v for k, v in step.items() if k != "agent"},
                default=str,
            ),
        ))

    return AgentRunResponse(
        conversation_id=conv.id,
        steps=agent_steps,
        final_answer=final_answer,
    )


@router.post("/run/stream")
async def run_agent_stream(
    request: AgentRunRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Run the multi-agent workflow with SSE streaming.
    Streams intermediate steps and the final answer.
    """
    repo = await _get_user_repo(db, request.repo_id, user)
    if repo.status != "ready":
        raise HTTPException(status_code=400, detail=f"Repository not ready (status: {repo.status})")

    memory = MemoryManager(db, request.repo_id)
    conv = await memory.get_or_create_conv(
        conversation_id=request.conversation_id,
        title=f"Agent: {request.task[:80]}",
        user_id=user.id,
    )
    await memory.save_user_message(conv.id, request.task)

    initial_state = {
        "messages": [],
        "user_query": request.task,
        "repo_id": request.repo_id,
        "repo_path": repo.local_path,
        "plan": [],
        "retrieved_context": [],
        "tool_results": [],
        "final_answer": "",
        "current_step": 0,
        "needs_tools": False,
        "error": None,
        "steps_log": [],
    }

    graph = get_compiled_graph()

    async def event_generator():
        yield f"data: {json.dumps({'type': 'metadata', 'conversation_id': conv.id})}\n\n"

        try:
            async for event in graph.astream(initial_state):
                for node_name, node_output in event.items():
                    # Stream each agent step
                    step_data = {
                        "type": "agent_step",
                        "agent": node_name,
                    }

                    if node_name == "planner" and "plan" in node_output:
                        step_data["plan"] = node_output["plan"]
                        step_data["needs_tools"] = node_output.get("needs_tools", False)

                    elif node_name == "retrieval" and "retrieved_context" in node_output:
                        step_data["chunks_found"] = len(node_output["retrieved_context"])
                        step_data["top_files"] = list({
                            c["file_path"] for c in node_output["retrieved_context"][:5]
                        })

                    elif node_name == "tool_agent" and "tool_results" in node_output:
                        step_data["tools_called"] = [
                            r["tool"] for r in node_output["tool_results"]
                        ]

                    elif node_name == "executor" and "final_answer" in node_output:
                        step_data["type"] = "final_answer"
                        step_data["content"] = node_output["final_answer"]

                        # Save the answer
                        await memory.save_assistant_message(
                            conv.id, node_output["final_answer"]
                        )
                        await db.commit()

                    yield f"data: {json.dumps(step_data, default=str)}\n\n"

        except Exception as exc:
            logger.exception("Agent stream error: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
