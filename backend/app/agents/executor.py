"""
Execution Agent — synthesizes all gathered context into a final, well-structured answer.
"""

import logging
from typing import Any

from app.rag.llm import _get_llm
from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.state import AgentState
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

EXECUTOR_SYSTEM_PROMPT = """You are the Execution Agent — the final synthesizer in an AI code assistant system.

You have been given:
1. The user's original question.
2. A plan that was executed.
3. Code chunks retrieved from semantic search.
4. Results from tool invocations (file reads, code analysis, etc.).

Your job is to produce a **comprehensive, well-structured, and actionable** final answer.

Guidelines:
- Reference specific file paths and line numbers.
- Use markdown formatting with appropriate code blocks.
- If code modifications were requested, show exact code changes.
- If debugging, explain the root cause and suggest fixes.
- If explaining architecture, describe the flow clearly with references.
- Be thorough but concise. Avoid unnecessary repetition.
- If information is incomplete, state what's missing rather than guessing.
- Always provide production-quality, actionable advice.
"""


async def executor_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Execution Agent.
    Produces the final answer by synthesizing all available context.
    """
    query = state["user_query"]
    plan = state.get("plan", [])
    context = state.get("retrieved_context", [])
    tool_results = state.get("tool_results", [])
    tool_analysis = state.get("tool_analysis", "")

    logger.info("[Executor] Synthesizing answer from %d context chunks and %d tool results",
                len(context), len(tool_results))

    # Build comprehensive context for the final LLM call
    context_section = ""
    if context:
        context_section = "\n## Retrieved Code\n\n"
        for i, chunk in enumerate(context, 1):
            name_str = f" ({chunk['name']})" if chunk.get('name') else ""
            context_section += (
                f"### {chunk['file_path']}{name_str} "
                f"(L{chunk['start_line']}–L{chunk['end_line']}) "
                f"[{chunk['chunk_type']}] — Relevance: {chunk['relevance_score']:.2%}\n"
                f"```{chunk['language']}\n{chunk['content']}\n```\n\n"
            )

    tools_section = ""
    if tool_results:
        tools_section = "\n## Tool Investigation Results\n\n"
        for i, tr in enumerate(tool_results, 1):
            tools_section += (
                f"### Tool Call {i}: `{tr['tool']}`\n"
                f"**Args:** `{tr['args']}`\n\n"
                f"{tr['result']}\n\n"
            )

    analysis_section = ""
    if tool_analysis:
        analysis_section = f"\n## Initial Analysis\n\n{tool_analysis}\n\n"

    plan_section = "\n## Execution Plan\n\n"
    for i, step in enumerate(plan, 1):
        plan_section += f"{i}. {step}\n"

    full_context = (
        f"## User Question\n\n{query}\n\n"
        f"{plan_section}\n"
        f"{context_section}\n"
        f"{analysis_section}"
        f"{tools_section}"
    )

    llm = _get_llm()

    messages = [
        SystemMessage(content=EXECUTOR_SYSTEM_PROMPT),
        HumanMessage(content=full_context),
    ]

    response = await llm.ainvoke(messages)
    final_answer = response.content

    step_log = {
        "agent": "executor",
        "answer_length": len(final_answer),
    }

    logger.info("[Executor] Generated answer: %d chars", len(final_answer))

    return {
        "final_answer": final_answer,
        "steps_log": [step_log],
    }
