"""
Tool Agent — uses LangChain tool-calling to dynamically select and invoke tools.
"""

import logging
from typing import Any

from app.rag.llm import _get_llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.agents.state import AgentState
from app.tools import get_all_tools
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

TOOL_AGENT_PROMPT = """You are the Tool Agent in an AI code assistant system.

You have access to tools that let you inspect and manipulate a codebase. Based on the retrieved 
context below, analyze and provide insights to answer the user's question.

**IMPORTANT**: You already have the relevant code context below. Analyze it directly and provide 
a helpful response. Do NOT attempt to call any tools unless absolutely necessary - the context 
has already been retrieved for you.

Retrieved code context:
{context}

User's original question: {query}

Plan step being executed: {plan_step}

Provide a detailed analysis based on the retrieved context above. If the context is sufficient, 
give your answer directly. Only describe what tools you WOULD use if the context is insufficient.
"""


async def tool_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Tool Agent.
    Analyzes retrieved context and provides insights. Uses tools only when necessary.
    """
    query = state["user_query"]
    repo_id = state["repo_id"]
    repo_path = state["repo_path"]
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    context = state.get("retrieved_context", [])

    # Format context for the prompt
    context_str = ""
    for i, chunk in enumerate(context[:8], 1):
        context_str += (
            f"\n### Chunk {i}: {chunk['file_path']} "
            f"(L{chunk['start_line']}-L{chunk['end_line']})\n"
            f"```{chunk['language']}\n{chunk['content']}\n```\n"
        )

    plan_step = plan[current_step] if current_step < len(plan) else "Analyze the retrieved context"

    logger.info("[ToolAgent] Executing with %d context chunks, step: %s", len(context), plan_step)

    # Get LLM without tools for initial analysis (more reliable with Groq)
    llm = _get_llm()

    formatted_prompt = TOOL_AGENT_PROMPT.format(
        context=context_str if context_str else "No context retrieved yet.",
        query=query,
        plan_step=plan_step,
    )

    messages = [
        SystemMessage(content=formatted_prompt),
        HumanMessage(content=f"Analyze the code context and execute the plan step: {plan_step}"),
    ]

    # Run initial analysis without tool binding (avoids Groq tool call issues)
    response = await llm.ainvoke(messages)
    analysis = response.content

    # Check if we need to use tools (only for specific operations like file reading)
    tool_results: list[dict] = []
    needs_tools = any(phrase in analysis.lower() for phrase in [
        "need to read", "need to examine", "let me check", "i'll look at",
        "need more information", "insufficient context"
    ])

    if needs_tools and context_str == "":
        # Only use tools if we have no context at all
        tools = get_all_tools(repo_id, repo_path)
        llm_with_tools = llm.bind_tools(tools)
        
        tool_prompt = f"""Based on your analysis, you indicated you need more information.
        You have these tools available:
        - search_code: Search the codebase semantically
        - read_file: Read a specific file
        - list_files: List directory contents
        
        What specific tool call would help? The user asked: {query}"""
        
        try:
            tool_response = await llm_with_tools.ainvoke([
                SystemMessage(content="You are a helpful assistant. Call the appropriate tool."),
                HumanMessage(content=tool_prompt),
            ])
            
            if tool_response.tool_calls:
                for tool_call in tool_response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    logger.info("[ToolAgent] Calling tool: %s(%s)", tool_name, tool_args)
                    
                    tool_fn = next((t for t in tools if t.name == tool_name), None)
                    if tool_fn:
                        try:
                            result = tool_fn.invoke(tool_args)
                            tool_results.append({
                                "tool": tool_name,
                                "args": tool_args,
                                "result": str(result)[:4000],
                            })
                        except Exception as exc:
                            logger.error("[ToolAgent] Tool error: %s", exc)
        except Exception as e:
            logger.warning("[ToolAgent] Tool calling failed, continuing with analysis: %s", e)

    step_log = {
        "agent": "tool",
        "analysis": analysis[:1000],
        "tools_called": [r["tool"] for r in tool_results],
    }

    logger.info("[ToolAgent] Analysis complete, called %d tools", len(tool_results))

    return {
        "tool_results": tool_results,
        "tool_analysis": analysis,
        "current_step": current_step + 1,
        "steps_log": [step_log],
    }
