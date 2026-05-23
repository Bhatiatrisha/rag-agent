# graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from state import AgentState
from nodes import (
    safety_check, call_llm, run_tools,
    generate_with_hallucination_check, block_response
)
from typing import Literal

def route_after_safety(state: AgentState) -> Literal["llm", "block"]:
    return "llm" if state["guardrail_passed"] else "block"

def route_after_llm(state: AgentState) -> Literal["tools", "hallucination_check", "end"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    # If docs were retrieved, run hallucination check
    if state.get("retrieved_docs"):
        return "hallucination_check"
    return "end"

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("safety",              safety_check)
    builder.add_node("llm",                 call_llm)
    builder.add_node("tools",               run_tools)
    builder.add_node("hallucination_check", generate_with_hallucination_check)
    builder.add_node("block",               block_response)

    builder.add_edge(START, "safety")
    builder.add_conditional_edges("safety", route_after_safety, {
        "llm":   "llm",
        "block": "block"
    })
    builder.add_conditional_edges("llm", route_after_llm, {
        "tools":               "tools",
        "hallucination_check": "hallucination_check",
        "end":                 END
    })
    builder.add_edge("tools", "llm")
    builder.add_edge("hallucination_check", END)
    builder.add_edge("block", END)

    return builder.compile(checkpointer=MemorySaver())

graph = build_graph()