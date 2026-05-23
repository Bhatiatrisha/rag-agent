# eval_pipeline.py

def score_response(
    query:         str,
    response:      str,
    retrieved_docs:list,
    latency_ms:    float,
) -> dict:
    """
    Score a response across 4 dimensions.
    Returns scores 0.0–1.0 per dimension + overall.
    """

    # 1. Relevance — does the response address the query?
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())
    overlap = query_words & response_words
    relevance = round(min(len(overlap) / max(len(query_words), 1), 1.0), 2)

    # 2. Groundedness — are response claims in the docs?
    if retrieved_docs and retrieved_docs != ["No relevant documents found."]:
        doc_text = " ".join(retrieved_docs).lower()
        meaningful = [
            w for w in response.lower().split()
            if len(w) > 4
        ]
        if meaningful:
            grounded_words = sum(1 for w in meaningful if w in doc_text)
            groundedness = round(grounded_words / len(meaningful), 2)
        else:
            groundedness = 1.0
    else:
        groundedness = 1.0  # no docs = direct LLM answer, not penalised

    # 3. Conciseness — penalise very long responses
    word_count = len(response.split())
    if word_count <= 100:
        conciseness = 1.0
    elif word_count <= 200:
        conciseness = 0.8
    elif word_count <= 400:
        conciseness = 0.6
    else:
        conciseness = 0.4

    # 4. Latency score — penalise slow responses
    if latency_ms < 1000:
        latency_score = 1.0
    elif latency_ms < 3000:
        latency_score = 0.8
    elif latency_ms < 5000:
        latency_score = 0.6
    else:
        latency_score = 0.4

    overall = round(
        (relevance * 0.3) +
        (groundedness * 0.4) +
        (conciseness * 0.2) +
        (latency_score * 0.1),
        2
    )

    return {
        "relevance":     relevance,
        "groundedness":  groundedness,
        "conciseness":   conciseness,
        "latency_score": latency_score,
        "overall":       overall,
    }