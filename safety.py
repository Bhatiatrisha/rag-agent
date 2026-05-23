# safety.py
import re
import json
import logging
from datetime import datetime

# ── Logging setup (this IS your audit trail) ─────────────
logging.basicConfig(
    filename="audit.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── PII patterns ──────────────────────────────────────────
PII_PATTERNS = {
    "email":       r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone":       r"\b[\+]?[0-9]{10,13}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "aadhaar":     r"\b[2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b",  # India-specific
    "pan":         r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",                 # India-specific
}

# ── Prompt injection patterns ─────────────────────────────
INJECTION_PATTERNS = [
    "ignore previous",
    "ignore all previous",
    "disregard instructions",
    "jailbreak",
    "you are now",
    "forget your instructions",
    "system prompt",
    "act as if",
    "pretend you are",
    "bypass",
    "override",
]

# ── Abuse patterns ────────────────────────────────────────
ABUSE_PATTERNS = [
    "how to hack",
    "how to attack",
    "ddos",
    "exploit vulnerability",
    "steal credentials",
    "phishing",
]

def detect_and_mask_pii(text: str) -> tuple[str, list[str]]:
    """
    Detects PII in text, masks it, and returns
    the masked text + list of PII types found.
    """
    found_pii = []
    masked = text

    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, masked)
        if matches:
            found_pii.append(pii_type)
            masked = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", masked)

    return masked, found_pii

def check_prompt_injection(text: str) -> bool:
    """Returns True if injection attempt detected."""
    text_lower = text.lower()
    return any(p in text_lower for p in INJECTION_PATTERNS)

def check_abuse(text: str) -> bool:
    """Returns True if abusive intent detected."""
    text_lower = text.lower()
    return any(p in text_lower for p in ABUSE_PATTERNS)

def check_hallucination(response: str, retrieved_docs: list[str]) -> dict:
    """
    Lightweight faithfulness check.
    Checks if key nouns in the response appear in retrieved docs.
    Returns a score and a flag.
    """
    if not retrieved_docs or retrieved_docs == ["No relevant documents found."]:
        # No docs used — LLM answered from general knowledge, skip check
        return {"score": 1.0, "grounded": True, "reason": "no_docs_used"}

    combined_docs = " ".join(retrieved_docs).lower()

    # Extract meaningful words from response (length > 4, not stopwords)
    stopwords = {"this", "that", "with", "from", "have", "been", "they",
                 "their", "will", "which", "about", "these", "there"}
    words = [
        w.lower().strip(".,!?") for w in response.split()
        if len(w) > 4 and w.lower() not in stopwords
    ]

    if not words:
        return {"score": 1.0, "grounded": True, "reason": "no_checkable_words"}

    matched = sum(1 for w in words if w in combined_docs)
    score = round(matched / len(words), 2)
    grounded = score >= 0.25  # at least 25% of response words appear in docs

    return {
        "score": score,
        "grounded": grounded,
        "reason": "faithfulness_check"
    }

def audit_log(event: str, thread_id: str, details: dict):
    """Write a structured audit entry to audit.log."""
    entry = {
        "event": event,
        "thread_id": thread_id,
        "timestamp": datetime.utcnow().isoformat(),
        **details
    }
    logger.info(json.dumps(entry))