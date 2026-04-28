"""
LLM-based document classifier using Nemotron 3 Super (free on OpenRouter).

Used as fallback when rule-based classifier confidence is low (<0.7).
"""

import os
import json
from typing import Optional
import openai  # OpenRouter-compatible client

# OpenRouter endpoint
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "nvidia/nemotron-3-super-12b"  # or 120B variant — whichever is free tier

# Categories we support
CATEGORIES = [
    "prd",  # Product requirements / spec
    "transcript",  # LLM conversation transcript
    "research_paper",  # Academic paper / research
    "knowledge_extract",  # Notes, wiki-style extracts, markdown notes
    "unknown",  # None of the above
]

SYSTEM_PROMPT = """You are a document classifier.
Given a document excerpt, classify it into exactly ONE of these categories:

- prd: Product requirements document, specifications, architecture, user stories
- transcript: Conversation between human and AI, chat log, dialogue turns
- research_paper: Academic-style paper with abstract/methods/results, citations
- knowledge_extract: Notes, wiki articles, markdown notes, Obsidian-style vault notes
- unknown: Does not match any category

Respond with ONLY the category name (lowercase). Do not add explanation.
"""


def classify(content: str, api_key: Optional[str] = None) -> str:
    """
    Classify document content using LLM.

    Parameters
    ----------
    content : str
        Document text (first 8k chars recommended)
    api_key : str | None
        OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.

    Returns
    -------
    str
        Category string (one of CATEGORIES)
    """
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OpenRouter API key required — set OPENROUTER_API_KEY")

    client = openai.OpenAI(api_key=api_key, base_url=BASE_URL)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content[:8000]},
        ],
        temperature=0.0,
        max_tokens=10,
    )
    raw = response.choices[0].message.content.strip().lower()

    # Validate
    if raw not in CATEGORIES:
        return "unknown"
    return raw


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m aaa_memory.classifier.llm <file>")
        sys.exit(1)

    content = Path(sys.argv[1]).read_text(errors="replace")[:8000]
    cat = classify(content)
    print(cat)
