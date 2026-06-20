"""
Element extractor — LLM-based structured extraction from transcripts.

Uses Nemotron 3 Super (free) via OpenRouter to extract knowledge elements:
- Decision
- Pattern
- Code snippet
- Working prompt
- Fact
- Concept
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import json
import openai
import os

BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "nvidia/nemotron-3-super-12b"  # free tier

# ── Schemas ────────────────────────────────────────────────────────────────────

ELEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "decision",
                            "pattern",
                            "code",
                            "prompt",
                            "fact",
                            "concept",
                            "noise",
                        ],
                    },
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "quote": {"type": "string"},  # exact excerpt from source
                    "line_range": {"type": "string"},  # e.g. "23-45" if available
                },
                "required": ["type", "title", "content", "confidence"],
            },
        }
    },
    "required": ["elements"],
}

SYSTEM_PROMPT = """You are a knowledge extraction engine.

Given a transcript of a human↔AI conversation, extract discrete knowledge elements.

Element types:
- decision: A choice made, with reasoning (e.g., "We will use SQLite because...")
- pattern: Reusable pattern or best practice (e.g., "Always validate embeddings before storing")
- code: Code snippets worth remembering (function, command, config)
- prompt: Working LLM prompt that produced good results
- fact: Atomic fact worth remembering (e.g., "EmbeddingGemma-300M uses 0.4GB VRAM")
- concept: Abstract idea or definition (e.g., "Reciprocal Rank Fusion combines ranked lists")
- noise: Boilerplate, greetings, chit-chat — discard

For each element:
- title: one-line summary (5–12 words)
- content: full extracted text (preserve details, code blocks, parameters)
- confidence: 0.0–1.0 (how certain you are this is a real element)
- tags: 1–5 lowercase keywords (e.g., ["sqlite", "embedding", "performance"])
- quote: the exact source text snippet (≤200 chars)

Output: strict JSON matching the schema. Only include high-confidence elements (confidence ≥ 0.6).
"""

USER_PROMPT_TEMPLATE = """Transcript:

{transcript}

---

Extract all knowledge elements from this transcript. Include decisions, patterns, code snippets, working prompts, facts, and concepts. Discard noise.
"""

# ── Types ──────────────────────────────────────────────────────────────────────


@dataclass
class Element:
    type: str
    title: str
    content: str
    confidence: float
    tags: List[str]
    quote: Optional[str] = None
    line_range: Optional[str] = None

    def to_dict(self):
        return asdict(self)


# ── Core ───────────────────────────────────────────────────────────────────────


def extract(transcript: str, api_key: Optional[str] = None) -> List[Element]:
    """
    Extract elements from a transcript using LLM.

    Parameters
    ----------
    transcript : str
        Full conversation text
    api_key : str | None
        OpenRouter API key (falls back to OPENROUTER_API_KEY)

    Returns
    -------
    List[Element]
    """
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY required")

    client = openai.OpenAI(api_key=api_key, base_url=BASE_URL)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(transcript=transcript[:32000]),
            },
        ],
        temperature=0.2,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    parsed = json.loads(raw)
    raw_elements = parsed.get("elements", [])

    # Convert → Element dataclasses
    elements = []
    for el in raw_elements:
        if el.get("confidence", 0) < 0.6:
            continue  # filter low-confidence
        elements.append(
            Element(
                type=el["type"],
                title=el["title"],
                content=el["content"],
                confidence=el["confidence"],
                tags=el.get("tags", []),
                quote=el.get("quote"),
                line_range=el.get("line_range"),
            )
        )

    return elements


# ── Fallback rule-based extractor ────────────────────────────────────────────


def extract_fallback(transcript: str) -> List[Element]:
    """
    Simple regex-based fallback extractor when LLM unavailable.
    Captures code blocks, decisions, prompts, facts, and bullet-point concepts.
    """
    import re

    elements = []

    # Code blocks
    for match in re.finditer(r"```(\w+)?\n(.*?)\n```", transcript, re.DOTALL):
        lang = match.group(1) or "text"
        code = match.group(2).strip()
        if len(code) > 20:
            elements.append(
                Element(
                    type="code",
                    title=f"{lang} code snippet",
                    content=code,
                    confidence=0.7,
                    tags=[lang, "code", "snippet"],
                    quote=code[:200],
                )
            )

    # Decisions: "We will..." / "I'll..." / "Let's..." / "Decision:"
    decision_pattern = re.compile(
        r"(?:We will|I will|Let\'s|Decision:?|Chose|Selected)\s+(.+?)(?:\.|$)",
        re.IGNORECASE,
    )
    for match in decision_pattern.finditer(transcript):
        elements.append(
            Element(
                type="decision",
                title=match.group(0)[:80].strip(),
                content=match.group(0).strip(),
                confidence=0.65,
                tags=["decision"],
                quote=match.group(0)[:200],
            )
        )

    # Prompts: quoted strings or "Prompt: ..." blocks
    prompt_pattern = re.compile(
        r"(?:Prompt|Instruction|System prompt):\s*\n*```(.+?)```",
        re.DOTALL | re.IGNORECASE,
    )
    for match in prompt_pattern.finditer(transcript):
        prompt_text = match.group(1).strip()
        if len(prompt_text) > 20:
            elements.append(
                Element(
                    type="prompt",
                    title="Working prompt",
                    content=prompt_text,
                    confidence=0.7,
                    tags=["prompt", "llm"],
                    quote=prompt_text[:200],
                )
            )

    # Facts: "X uses Y GB VRAM", "The latency is Z ms" patterns
    fact_pattern = re.compile(
        r"(?:[A-Z][a-zA-Z0-9_]+ (?:uses|requires|consumes|takes|is)\s+.{0,50}?(?:\d+(?:\.\d+)?\s*(?:GB|MB|KB|ms|seconds?|minutes?)))",
        re.IGNORECASE,
    )
    for match in fact_pattern.finditer(transcript):
        elements.append(
            Element(
                type="fact",
                title=match.group(0)[:80],
                content=match.group(0),
                confidence=0.6,
                tags=["fact"],
                quote=match.group(0)[:200],
            )
        )

    # Concepts: Definition-like sentences "X is a Y that ..."
    concept_pattern = re.compile(r"([A-Z][a-zA-Z0-9_]+ (?:is|are|refers to) [^.]+\.)")
    for match in concept_pattern.finditer(transcript):
        sentence = match.group(0)
        if 30 < len(sentence) < 300:
            elements.append(
                Element(
                    type="concept",
                    title=sentence[:80],
                    content=sentence,
                    confidence=0.55,
                    tags=["concept"],
                    quote=sentence[:200],
                )
            )

    return elements


# ── Batch extractor ───────────────────────────────────────────────────────────


def extract_batch(
    transcript_path: Path, use_llm: bool = True, api_key: Optional[str] = None
) -> List[Element]:
    """Read transcript file and extract elements."""
    text = transcript_path.read_text(errors="replace")
    if use_llm:
        try:
            return extract(text, api_key=api_key)
        except Exception:
            pass  # fall back
    return extract_fallback(text)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m aaa_memory.extractor llm <file> | fallback <file>")
        sys.exit(1)

    cmd, path = sys.argv[1], Path(sys.argv[2])
    if cmd == "llm":
        els = extract(path.read_text())
    else:
        els = extract_fallback(path.read_text())

    for el in els:
        print(json.dumps(el.to_dict(), indent=2))
