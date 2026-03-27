"""
LangGraph Workflow — defines the state machine that orchestrates all agents.

Flow:
    START → planner → retrieval → (conditional) tool_agent → executor → END

The Planner decides whether tools are needed.
If needs_tools=False, the workflow skips the Tool Agent.
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, START, END

from app.agents.state import AgentState
from app.agents.planner import planner_node
from app.agents.retrieval import retrieval_node
from app.agents.tool_agent import tool_node
from app.agents.executor import executor_node

logger = logging.getLogger(__name__)


def _route_after_planner(state: AgentState) -> Literal["retrieval"]:
    """After planning, always go to retrieval first."""
    return "retrieval"


def _route_after_retrieval(state: AgentState) -> Literal["tool_agent", "executor"]:
    """After retrieval, decide whether to use tools or go straight to execution."""
    if state.get("needs_tools", False):
        logger.info("[Router] Routing to tool_agent (needs_tools=True)")
        return "tool_agent"
    else:
        logger.info("[Router] Skipping tools, routing to executor")
        return "executor"


def _route_after_tools(state: AgentState) -> Literal["executor"]:
    """After tools, always proceed to executor."""
    return "executor"


def build_agent_graph() -> StateGraph:
    """
    Construct the LangGraph workflow.

    Graph topology:

        ┌─────────┐
        │  START   │
        └────┬────┘
             │
        ┌────▼────┐
        │ planner │  ← generates plan, decides needs_tools
        └────┬────┘
             │
        ┌────▼─────┐
        │ retrieval │  ← semantic search / RAG
        └────┬─────┘
             │
        ┌────▼────────────┐
        │ conditional     │
        │ needs_tools?    │
        └──┬──────────┬───┘
           │          │
      ┌────▼────┐     │
      │  tool   │     │
      │  agent  │     │
      └────┬────┘     │
           │          │
        ┌──▼──────────▼──┐
        │    executor    │  ← final synthesis
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │      END       │
        └────────────────┘
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("tool_agent", tool_node)
    graph.add_node("executor", executor_node)

    # Define edges
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retrieval")

    # Conditional routing after retrieval
    graph.add_conditional_edges(
        "retrieval",
        _route_after_retrieval,
        {
            "tool_agent": "tool_agent",
            "executor": "executor",
        },
    )

    graph.add_edge("tool_agent", "executor")
    graph.add_edge("executor", END)

    return graph


def compile_agent_graph():
    """Build and compile the agent graph for execution."""
    graph = build_agent_graph()
    compiled = graph.compile()
    logger.info("Agent graph compiled successfully")
    return compiled


# Module-level compiled graph (lazy singleton)
_compiled_graph = None


def get_compiled_graph():
    """Get or create the compiled agent graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = compile_agent_graph()
    return _compiled_graph
