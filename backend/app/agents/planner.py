"""
Planner Agent — analyzes the user query and generates a step-by-step execution plan.
"""

import json
import logging
from typing import Any

from app.rag.llm import _get_llm
from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.state import AgentState
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

PLANNER_SYSTEM_PROMPT = """You are the Planner Agent in an AI code assistant system.

Your job is to analyze a user's question about a codebase and produce:
1. A clear step-by-step plan to answer their question or complete their task.
2. A decision on whether external tools (file reading, test running, code modification) are needed beyond semantic search.

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{
    "plan": [
        "Step 1: <description>",
        "Step 2: <description>",
        ...
    ],
    "needs_tools": true/false,
    "reasoning": "Brief explanation of why you chose this plan"
}

Guidelines:
- Keep plans concise: 2-6 steps maximum.
- Always include a retrieval step (semantic code search) as one of the first steps.
- Set needs_tools=true if the task requires reading specific files, running tests, modifying code, or listing directories.
- Set needs_tools=false if the question can be answered purely from semantic search results.
- For debugging tasks, include steps to trace the execution flow.
- For feature requests, include steps to identify relevant files and suggest modifications.
"""


async def planner_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Planner Agent.
    Produces a plan and decides whether tools are needed.
    """
    logger.info("[Planner] Processing query: %.100s...", state["user_query"])

    llm = _get_llm()

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"User query: {state['user_query']}"),
    ]

    response = await llm.ainvoke(messages)
    content = response.content.strip()

    # Parse JSON response
    try:
        # Handle markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        parsed = json.loads(content)
        plan = parsed.get("plan", ["Search codebase", "Analyze results", "Respond"])
        needs_tools = parsed.get("needs_tools", False)
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("[Planner] Failed to parse JSON response; using defaults. Error: %s", exc)
        plan = [
            "Search the codebase for relevant code",
            "Analyze the retrieved code",
            "Generate a comprehensive response",
        ]
        needs_tools = False
        reasoning = "Fallback plan due to parsing failure."

    step_log = {
        "agent": "planner",
        "plan": plan,
        "needs_tools": needs_tools,
        "reasoning": reasoning,
    }

    logger.info("[Planner] Plan: %s | needs_tools=%s", plan, needs_tools)

    return {
        "plan": plan,
        "needs_tools": needs_tools,
        "current_step": 0,
        "steps_log": [step_log],
    }
