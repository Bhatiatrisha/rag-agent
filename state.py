# state.py
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages:          Annotated[list, add_messages]
    retrieved_docs:    list
    guardrail_passed:  bool
    pii_detected:      list      # NEW — PII types found in query
    hallucination_ok:  bool      # NEW — faithfulness check result
    thread_id:         str       # NEW — for audit trail