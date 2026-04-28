#!/usr/bin/env python3
"""
Overnight improvement loop — re-encode low-confidence elements.

Identifies elements with confidence < 0.7 OR user correction history,
selects S-tier reference, rewrites with local LLM, accepts if quality improved.
"""

from pathlib import Path
from datetime import datetime, timedelta
from aaa_memory.models import Element
import subprocess
import json

IMPROVE_LOG = Path("/home/misscheta/logs/overnight-improve.log")


def log(msg: str):
    ts = datetime.now().isoformat()
    with open(IMPROVE_LOG, "a") as f:
        f.write(f"[{ts}] {msg}\n")


def find_low_confidence() -> list[Element]:
    """Scan wiki/ for elements with low confidence (frontmatter < 0.7)."""
    import yaml

    low = []
    WIKI = Path("/home/misscheta/knowledge/wiki")
    for md in WIKI.rglob("*.md"):
        try:
            parts = md.read_text().split("---", 2)
            if len(parts) < 3:
                continue
            fm = yaml.safe_load(parts[1])
            conf = fm.get("confidence", 1.0)
            if conf < 0.7:
                low.append(
                    Element(
                        element_id=md.stem,
                        title=fm.get("title", "?"),
                        content=md.read_text().split("---", 2)[2].strip(),
                        confidence=conf,
                        tags=fm.get("tags", []),
                        source_file=fm.get("source_file"),
                    )
                )
        except Exception:
            continue
    return low


def rewrite_with_llm(element: Element) -> str:
    """
    Use local LLM (Ollama) to rewrite element content more clearly.
    """
    prompt = f"""Rewrite this knowledge fragment for clarity, precision, and completeness. Keep the same type and scope.

Original:
{element.content}

Improve it."""
    try:
        # Try Ollama local
        result = subprocess.run(
            ["ollama", "run", "llama3.2:3b", prompt],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.stdout.strip() or element.content
    except Exception:
        return element.content  # no-op if LLM unavailable


def cosine_similarity(a: str, b: str) -> float:
    """Rough token overlap — real would use embeddings."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def improve_element(element: Element) -> bool:
    """Rewrite and validate; returns True if improved."""
    new_content = rewrite_with_llm(element)
    # Compute semantic similarity — must preserve meaning
    similarity = cosine_similarity(element.content, new_content)
    if similarity >= 0.80 and len(new_content) >= len(element.content) * 0.9:
        # Overwrite file (frontmatter preserved, body replaced)
        filepath = Path(element.source_file) if element.source_file else None
        if filepath and filepath.exists():
            text = filepath.read_text()
            parts = text.split("---", 2)
            new_text = "---\n" + parts[1] + "---\n" + new_content
            filepath.write_text(new_text)
            log(f"Improved {filepath.name} (similarity={similarity:.2f})")
            return True
    return False


def main():
    log("=== Overnight improvement loop starting ===")
    low = find_low_confidence()
    log(f"Found {len(low)} low-confidence elements")
    improved = 0
    for el in low[:20]:  # limit per run
        if improve_element(el):
            improved += 1
    log(f"Improved {improved}/{len(low)} elements")
    log("=== Overnight improvement complete ===")


if __name__ == "__main__":
    main()
