# Multi-Agent RAG Platform

Production-grade agentic AI system built with LangGraph, ChromaDB, and FastAPI.
Designed to demonstrate real-world AI engineering practices relevant to
intelligent networking and telecom product features.

## Architecture

```
User Query
    │
    ▼
┌─────────────┐     blocked      ┌──────────────┐
│ Safety Node  │ ───────────────► │ Block Response│
│ PII masking  │                  └──────────────┘
│ Injection    │
│ Abuse check  │
└──────┬──────┘
       │ passed
       ▼
┌─────────────┐    tool call     ┌──────────────┐
│  LLM Node   │ ───────────────► │  Tools Node  │
│  (Claude)   │ ◄─────────────── │  RAG/Calc    │
└──────┬──────┘    tool result   └──────────────┘
       │ final answer
       ▼
┌─────────────────┐
│ Hallucination   │
│ Check Node      │
└──────┬──────────┘
       │
       ▼
   Response +
   Eval Scores +
   Latency
```

## Features

- **Multi-agent orchestration** via LangGraph with conditional routing and ReAct loop
- **RAG pipeline** — ChromaDB vector store, sentence-transformers embeddings, cosine similarity retrieval
- **Tool use** — document search, calculator, fallback handler
- **Safety layer** — PII masking (email, phone, Aadhaar, PAN), prompt injection defence, abuse detection
- **Hallucination controls** — faithfulness scoring against retrieved documents
- **Observability** — per-request latency, token usage, cost tracking, p95 metrics
- **Eval pipeline** — inline quality scoring (relevance, groundedness, conciseness, latency)
- **Audit trail** — structured JSON logs for every safety and quality event
- **Production ready** — Dockerised, health check endpoint, environment-based secrets

## Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph, LangChain |
| LLM | Anthropic Claude (provider-agnostic) |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| API | FastAPI |
| Container | Docker, docker-compose |

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/query` | POST | Submit a query, get response + eval scores |
| `/metrics` | GET | Live observability dashboard |
| `/health` | GET | Health check |

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/Bhatiatrisha/rag-agent.git
cd rag-agent
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

# 2. Ingest documents
pip install -r requirements.txt
python ingest.py

# 3. Run with Docker
docker compose up

# 4. Test
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"message": "What is SDN?", "thread_id": "demo"}'
```

## Sample Response

```json
{
  "response": "SDN (Software Defined Networking) separates the network control plane from the data plane...",
  "docs_used": ["SDN allows network administrators to manage..."],
  "latency_ms": 1243.5,
  "eval_scores": {
    "relevance": 0.72,
    "groundedness": 0.81,
    "conciseness": 1.0,
    "latency_score": 0.8,
    "overall": 0.83
  }
}
```

## Project Structure

```
rag-agent/
├── main.py            # FastAPI app + metrics wiring
├── graph.py           # LangGraph graph definition
├── nodes.py           # Node functions
├── state.py           # AgentState TypedDict
├── tools.py           # Tool definitions
├── retriever.py       # ChromaDB query
├── ingest.py          # Document ingestion pipeline
├── safety.py          # PII, injection, abuse, audit trail
├── observability.py   # Metrics, cost, latency tracking
├── eval_pipeline.py   # Response quality scoring
├── Dockerfile
├── docker-compose.yml
└── docs/              # Your document corpus
```