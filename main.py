# main.py
import time
from fastapi import FastAPI
from pydantic import BaseModel
from graph import graph
from observability import record_request, get_metrics_summary
from eval_pipeline import score_response

app = FastAPI(title="RAG Agent — Deutsche Telekom")

class QueryRequest(BaseModel):
    message:   str
    thread_id: str = "default"

class QueryResponse(BaseModel):
    response:   str
    docs_used:  list[str]
    eval_scores: dict      # NEW — quality scores returned per request
    latency_ms:  float     # NEW — latency visible in response

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    start  = time.time()
    error  = False

    try:
        result = graph.invoke(
            {
                "messages":  [{"role": "user", "content": req.message}],
                "thread_id": req.thread_id
            },
            config=config
        )
    except Exception as e:
        error = True
        raise e
    finally:
        latency_ms = round((time.time() - start) * 1000, 2)

    response_text = result["messages"][-1].content
    docs_used     = result.get("retrieved_docs", [])
    pii_found     = result.get("pii_detected", [])
    blocked       = not result.get("guardrail_passed", True)
    hallucination = not result.get("hallucination_ok", True)

    # Extract tools used from message history
    tools_used = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tools_used += [tc["name"] for tc in msg.tool_calls]

    # Count tokens (Claude returns usage metadata)
    last_msg = result["messages"][-1]
    input_tokens  = getattr(last_msg, "usage_metadata", {}).get("input_tokens", 0) if hasattr(last_msg, "usage_metadata") else 0
    output_tokens = getattr(last_msg, "usage_metadata", {}).get("output_tokens", 0) if hasattr(last_msg, "usage_metadata") else 0

    # Record metrics
    record_request(
        thread_id     = req.thread_id,
        latency_ms    = latency_ms,
        input_tokens  = input_tokens,
        output_tokens = output_tokens,
        blocked       = blocked,
        hallucination = hallucination,
        pii_found     = pii_found,
        tools_used    = tools_used,
        error         = error,
    )

    # Eval scoring
    scores = score_response(
        query         = req.message,
        response      = response_text,
        retrieved_docs= docs_used,
        latency_ms    = latency_ms,
    )

    return QueryResponse(
        response    = response_text,
        docs_used   = docs_used,
        eval_scores = scores,
        latency_ms  = latency_ms,
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/metrics")
async def metrics():
    """Live observability dashboard — latency, cost, token usage, quality rates."""
    return get_metrics_summary()