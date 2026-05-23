
# nodes.py
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from state import AgentState
from tools import tools
from safety import (
    detect_and_mask_pii,
    check_prompt_injection,
    check_abuse,
    check_hallucination,
    audit_log
)
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
llm_with_tools = llm.bind_tools(tools)
tool_map = {t.name: t for t in tools}

def safety_check(state: AgentState) -> AgentState:
    """
    Full safety check:
    1. Mask PII in the incoming query
    2. Check for prompt injection
    3. Check for abuse
    4. Audit log the event
    """
    raw_query = state["messages"][-1].content
    thread_id = state.get("thread_id", "unknown")

    # PII masking
    masked_query, pii_found = detect_and_mask_pii(raw_query)

    # Log PII detection
    if pii_found:
        audit_log("pii_detected", thread_id, {
            "pii_types": pii_found,
            "original_length": len(raw_query)
        })

    # Injection + abuse check
    injection = check_prompt_injection(masked_query)
    abuse = check_abuse(masked_query)
    passed = not injection and not abuse

    # Audit log the safety decision
    audit_log("safety_check", thread_id, {
        "passed": passed,
        "injection_detected": injection,
        "abuse_detected": abuse,
        "pii_found": pii_found
    })

    # Replace the last message content with the PII-masked version
    from langchain_core.messages import HumanMessage
    masked_messages = list(state["messages"][:-1]) + [
        HumanMessage(content=masked_query)
    ]

    return {
        "messages": masked_messages,
        "guardrail_passed": passed,
        "pii_detected": pii_found,
        "thread_id": thread_id
    }

def call_llm(state: AgentState) -> AgentState:
    system = SystemMessage(content=(
        "You are a helpful assistant with access to tools. "
        "When the user asks about telecom, networking, LTE, 5G, SDN, NFV, or RAG, "
        "always use the search_documents tool. "
        "For math calculations, use the calculate tool. "
        "For everything else, answer directly."
    ))
    messages = [system] + list(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def run_tools(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    results = []
    retrieved = state.get("retrieved_docs", [])

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name in tool_map:
            result = tool_map[tool_name].invoke(tool_args)
        else:
            result = f"Tool '{tool_name}' not found."

        results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"],
            name=tool_name
        ))

        if tool_name == "search_documents":
            retrieved = [str(result)]

    return {"messages": results, "retrieved_docs": retrieved}

def generate_with_hallucination_check(state: AgentState) -> AgentState:
    """
    After tool execution, do a final hallucination check
    on the LLM's response before returning it to the user.
    """
    # Get the last AI response
    last = state["messages"][-1]
    response_text = last.content if hasattr(last, "content") else str(last)
    docs = state.get("retrieved_docs", [])
    thread_id = state.get("thread_id", "unknown")

    # Faithfulness check
    hall_result = check_hallucination(response_text, docs)

    # Audit log
    audit_log("hallucination_check", thread_id, {
        "score": hall_result["score"],
        "grounded": hall_result["grounded"],
        "reason": hall_result["reason"]
    })

    # If response is not grounded, append a disclaimer
    if not hall_result["grounded"] and docs:
        disclaimer = "\n\n⚠️ Note: This response may not be fully grounded in retrieved documents."
        from langchain_core.messages import AIMessage
        grounded_response = AIMessage(content=response_text + disclaimer)
        return {
            "messages": [grounded_response],
            "hallucination_ok": False
        }

    return {"hallucination_ok": True}

def block_response(state: AgentState) -> AgentState:
    thread_id = state.get("thread_id", "unknown")
    audit_log("request_blocked", thread_id, {
        "reason": "safety_check_failed"
    })
    return {"messages": [AIMessage(content="I can't process that request.")]}