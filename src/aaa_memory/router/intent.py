"""
Intent classifier — routes queries to hot/warm/cold/all tiers.

Uses Nemotron 3 Super (free) via OpenRouter, with rule-based fallback.
"""

import os
import re
import openai
from enum import Enum
from dataclasses import dataclass

BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "nvidia/nemotron-3-super-12b"


class Intent(str, Enum):
    RECENT = "recent"  # Hot tier — current session
    RELATIONSHIP = "relationship"  # Warm tier — Graphiti traversal
    ARCHIVAL = "archival"  # Cold tier — long-term
    FACTUAL = "factual"  # Fact lookup (hot + warm)
    AMBIGUOUS = "ambiguous"  # Search all tiers


SYSTEM_PROMPT = """You are a query intent classifier.

Given a user query, classify it into exactly ONE intent category:

- recent: Questions about something that happened in the last few hours/days (e.g., "what did we discuss today?")
- relationship: Questions about how things relate, patterns, or connections (e.g., "how does auth tie into token system?", "what patterns emerged?")
- archival: Questions about something from months ago or deep history (e.g., "that conversation from June about WebSockets")
- factual: Simple fact lookup that likely exists in indexed knowledge (e.g., "what VRAM does EmbeddingGemma use?")
- ambiguous: Cannot determine from query alone; likely needs multi-tier search

Output ONLY the category name (lowercase).
"""

RULE_FALLBACK = {
    "recent": [
        r"\b(?:today|yesterday|this week|last few days|recent)\b",
        r"\b(?:just now|earlier|earlier today)\b",
        r"\bsession\b",
    ],
    "relationship": [
        r"\bhow does .* relate",
        r"\bconnection between",
        r"\bhow are .* and .* connected",
        r"\bwhat.*pattern",
        r"\bcorrelation",
    ],
    "archival": [
        r"\b(?:6 months ago|last year|months? ago|old|previous)\b",
        r"\bArchival\b",
        r"\bfrom .* history\b",
    ],
    "factual": [
        r"^(?:what|which|who|where) (?:is|are|does|use)",
    ],
}


def classify_intent_llm(query: str, api_key: str = None) -> Intent:
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY required")

    client = openai.OpenAI(api_key=api_key, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=10,
    )
    raw = resp.choices[0].message.content.strip().lower()
    try:
        return Intent(raw)
    except ValueError:
        return Intent.AMBIGUOUS


def classify_intent_rule(query: str) -> Intent:
    """Fast regex-based fallback."""
    q = query.lower()
    for intent, patterns in RULE_FALLBACK.items():
        for pat in patterns:
            if re.search(pat, q, re.IGNORECASE):
                return Intent(intent)
    return Intent.AMBIGUOUS


def classify_intent(query: str, use_llm: bool = True) -> Intent:
    """
    Two-tier classification.

    Priority:
    1. Rule fallback for high-confidence patterns (recent/time keywords) -> return immediately
    2. If rule match weak → LLM classification (if available)
    3. Else ambiguous
    """
    rule_result = classify_intent_rule(query)
    if rule_result != Intent.AMBIGUOUS:
        return rule_result

    if use_llm:
        try:
            return classify_intent_llm(query)
        except Exception:
            pass

    return Intent.AMBIGUOUS


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "What did we talk about today?"
    print(classify_intent(q))
