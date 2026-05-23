# observability.py
import time
import json
import logging
from datetime import datetime
from collections import defaultdict
from typing import Optional

# ── Metrics logger (separate from audit log) ──────────────
metrics_logger = logging.getLogger("metrics")
metrics_logger.setLevel(logging.INFO)
handler = logging.FileHandler("metrics.log")
handler.setFormatter(logging.Formatter("%(message)s"))
metrics_logger.addHandler(metrics_logger.handlers[0] if metrics_logger.handlers else handler)

# ── In-memory metrics store ───────────────────────────────
# In production this would be Prometheus/Grafana or CloudWatch
_metrics = {
    "total_requests":     0,
    "blocked_requests":   0,
    "total_latency_ms":   0.0,
    "total_input_tokens": 0,
    "total_output_tokens":0,
    "hallucination_flags":0,
    "pii_detections":     0,
    "tool_calls":         defaultdict(int),   # tool_name → count
    "latency_history":    [],                 # last 100 latencies
    "errors":             0,
}

# Approximate cost per 1000 tokens (Claude Haiku pricing)
COST_PER_1K_INPUT  = 0.00025   # USD
COST_PER_1K_OUTPUT = 0.00125   # USD

def record_request(
    thread_id:       str,
    latency_ms:      float,
    input_tokens:    int,
    output_tokens:   int,
    blocked:         bool      = False,
    hallucination:   bool      = False,
    pii_found:       list      = [],
    tools_used:      list      = [],
    error:           bool      = False,
):
    """Record metrics for a single request."""
    _metrics["total_requests"]      += 1
    _metrics["total_latency_ms"]    += latency_ms
    _metrics["total_input_tokens"]  += input_tokens
    _metrics["total_output_tokens"] += output_tokens

    if blocked:        _metrics["blocked_requests"]    += 1
    if hallucination:  _metrics["hallucination_flags"] += 1
    if pii_found:      _metrics["pii_detections"]      += 1
    if error:          _metrics["errors"]              += 1

    for tool in tools_used:
        _metrics["tool_calls"][tool] += 1

    # Keep last 100 latencies for p95 calculation
    _metrics["latency_history"].append(latency_ms)
    if len(_metrics["latency_history"]) > 100:
        _metrics["latency_history"].pop(0)

    # Write to metrics log
    entry = {
        "timestamp":    datetime.utcnow().isoformat(),
        "thread_id":    thread_id,
        "latency_ms":   round(latency_ms, 2),
        "input_tokens": input_tokens,
        "output_tokens":output_tokens,
        "blocked":      blocked,
        "hallucination":hallucination,
        "pii_found":    pii_found,
        "tools_used":   tools_used,
        "error":        error,
    }
    metrics_logger.info(json.dumps(entry))

def get_metrics_summary() -> dict:
    """Return aggregated metrics — exposed via /metrics endpoint."""
    total = _metrics["total_requests"] or 1  # avoid div by zero

    avg_latency = round(_metrics["total_latency_ms"] / total, 2)

    # p95 latency
    history = sorted(_metrics["latency_history"])
    p95_idx  = int(len(history) * 0.95)
    p95_latency = round(history[p95_idx], 2) if history else 0.0

    # Cost estimate
    cost = round(
        (_metrics["total_input_tokens"]  / 1000 * COST_PER_1K_INPUT) +
        (_metrics["total_output_tokens"] / 1000 * COST_PER_1K_OUTPUT),
        6
    )

    return {
        "total_requests":          _metrics["total_requests"],
        "blocked_requests":        _metrics["blocked_requests"],
        "error_rate":              round(_metrics["errors"] / total, 4),
        "block_rate":              round(_metrics["blocked_requests"] / total, 4),
        "hallucination_flag_rate": round(_metrics["hallucination_flags"] / total, 4),
        "pii_detection_rate":      round(_metrics["pii_detections"] / total, 4),
        "avg_latency_ms":          avg_latency,
        "p95_latency_ms":          p95_latency,
        "total_input_tokens":      _metrics["total_input_tokens"],
        "total_output_tokens":     _metrics["total_output_tokens"],
        "estimated_cost_usd":      cost,
        "tool_call_counts":        dict(_metrics["tool_calls"]),
    }