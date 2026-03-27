"""
Tool Agent — uses LangChain tool-calling to dynamically select and invoke tools.
"""

import logging
from typing import Any

from app.rag.llm import _get_llm
from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.state import AgentState
from app.tools import get_all_tools
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

TOOL_AGENT_PROMPT = """You are the Tool Agent in an AI code assistant system.

You have access to tools that let you inspect and manipulate a codebase. Use these tools to gather
the information needed to answer the user's question or complete their task.

Relevant code context has already been retrieved from semantic search. Review it and decide
if you need to use any tools to:
- Read specific files for more detail
- List directory structures
- Find specific function/class definitions
- Run tests
- Modify code (only if the user explicitly requested changes)

Guidelines:
- Only use tools when the retrieved context is insufficient.
- Be targeted — read specific files rather than listing everything.
- For debugging, trace the execution path by reading relevant files.
- For code modifications, always confirm what exists before changing.
- Stop after gathering enough information (don't over-investigate).

Retrieved code context:
{context}

User's original question: {query}

Plan step being executed: {plan_step}
"""


async def tool_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Tool Agent.
    Uses LLM with tool-calling to inspect the codebase as needed.
    """
    query = state["user_query"]
    repo_id = state["repo_id"]
    repo_path = state["repo_path"]
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    context = state.get("retrieved_context", [])

    # Format context for the prompt
    context_str = ""
    for i, chunk in enumerate(context[:5], 1):
        context_str += (
            f"\n### Chunk {i}: {chunk['file_path']} "
            f"(L{chunk['start_line']}-L{chunk['end_line']})\n"
            f"```{chunk['language']}\n{chunk['content']}\n```\n"
        )

    plan_step = plan[current_step] if current_step < len(plan) else "Investigate further"

    logger.info("[ToolAgent] Executing with %d context chunks, step: %s", len(context), plan_step)

    # Create LLM with tools bound
    tools = get_all_tools(repo_id, repo_path)
    llm = _get_llm()
    llm_with_tools = llm.bind_tools(tools)

    formatted_prompt = TOOL_AGENT_PROMPT.format(
        context=context_str if context_str else "No context retrieved yet.",
        query=query,
        plan_step=plan_step,
    )

    messages = [
        SystemMessage(content=formatted_prompt),
        HumanMessage(content=f"Execute the plan step: {plan_step}"),
    ]

    # Run the LLM — it may make tool calls
    tool_results: list[dict] = []
    max_iterations = 5  # prevent infinite loops

    for iteration in range(max_iterations):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        # Check if the LLM wants to call tools
        if not response.tool_calls:
            break

        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            logger.info("[ToolAgent] Calling tool: %s(%s)", tool_name, tool_args)

            # Find and invoke the tool
            tool_fn = next((t for t in tools if t.name == tool_name), None)
            if tool_fn:
                try:
                    result = tool_fn.invoke(tool_args)
                except Exception as exc:
                    result = f"Error calling {tool_name}: {exc}"
                    logger.error("[ToolAgent] Tool error: %s", exc)
            else:
                result = f"Unknown tool: {tool_name}"

            tool_results.append({
                "tool": tool_name,
                "args": tool_args,
                "result": str(result)[:4000],  # truncate very long results
            })

            # Feed tool result back to the LLM
            from langchain_core.messages import ToolMessage
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            ))

    step_log = {
        "agent": "tool",
        "tools_called": [r["tool"] for r in tool_results],
        "iteration_count": iteration + 1 if tool_results else 0,
    }

    logger.info("[ToolAgent] Called %d tools in %d iterations", len(tool_results), step_log["iteration_count"])

    return {
        "tool_results": tool_results,
        "current_step": current_step + 1,
        "steps_log": [step_log],
    }
