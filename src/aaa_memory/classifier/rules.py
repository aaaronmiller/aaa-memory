"""
Document classifier — rule-based + LLM fallback.

Classifies raw files into: prd, transcript, research_paper, knowledge_extract.
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import json


@dataclass
class ClassificationResult:
    category: (
        str  # 'prd', 'transcript', 'research_paper', 'knowledge_extract', 'unknown'
    )
    confidence: float  # 0.0 – 1.0
    rule_match: Optional[str] = None  # which rule fired
    llm_used: bool = False


# ── Rule patterns ────────────────────────────────────────────────────────────

_PR_D_PATTERNS = [
    r"\b(?:requirements?|specifications?|architecture|design|product)\b",
    r"\b(?:user stories?|acceptance criteria|epic|feature)\b",
    r"\b(?:non-functional|NFRS?|performance|scalability)\b",
    r"(?i)#\s*requirements",
    r"(?i)## \s*architecture",
]

_TRANSCRIPT_PATTERNS = [
    r"(?m)^\s*Human:\s*(.+)",
    r"(?m)^\s*Assistant:\s*(.+)",
    r"(?m)^\s*User:\s*(.+)",
    r"(?m)^\s*Model:\s*(.+)",
    r"Turn \d+",
]

_RESEARCH_PAPER_PATTERNS = [
    r"\b(?:abstract|introduction|methodology|results|conclusion|references)\b",
    r"\b(?:doi|arXiv|journal|proceedings)\b",
    r"\(\d{4}\)",  # citation year
    r"Figure \d+",
    r"Table \d+",
]

_KNOWLEDGE_EXTRACT_PATTERNS = [
    r"\b(?:decision|pattern|concept|insight|finding)\b",
    r"---",  # markdown HR — possible note fragment
    r"\[\[",  # wikilinks
    r"# \[",  # AI匆忙笔记
]

# ── Extension whitelists ─────────────────────────────────────────────────────

PRD_EXTS = {".md", ".txt", ".rst"}
TRANSCRIPT_EXTS = {".jsonl", ".json", ".txt"}
RESEARCH_EXTS = {".pdf", ".tex", ".md"}
KNOWLEDGE_EXTS = {".md", ".txt", ".org", ".adoc"}

# ── Public API ───────────────────────────────────────────────────────────────


def classify_file(
    path: Path, content: Optional[str] = None, llm_fallback=None
) -> ClassificationResult:
    """
    Classify a single file.

    Parameters
    ----------
    path : Path
        File path
    content : str | None
        File contents (if already read); if None, file will be read
    llm_fallback : callable | None
        Function(content) -> category (str) to call if rule-based confidence < 0.7

    Returns
    -------
    ClassificationResult
    """
    if content is None:
        try:
            content = path.read_text(errors="replace")
        except Exception:
            return ClassificationResult("unknown", 0.0)

    stem = path.stem.lower()
    suffix = path.suffix.lower()
    text_blob = (stem + " " + content[:2000]).lower()

    # 1. Extension quick-filter
    if suffix in PRD_EXTS:
        # Still verify with patterns
        pass
    elif suffix in TRANSCRIPT_EXTS:
        for pat in _TRANSCRIPT_PATTERNS:
            if re.search(pat, content, re.MULTILINE):
                return ClassificationResult(
                    "transcript", 0.85, rule_match="ext+pattern"
                )
        return ClassificationResult("transcript", 0.6, rule_match="ext-only")
    elif suffix in RESEARCH_EXTS:
        for pat in _RESEARCH_PAPER_PATTERNS:
            if re.search(pat, content, re.IGNORECASE):
                return ClassificationResult("research_paper", 0.8, rule_match="pattern")
        return ClassificationResult("research_paper", 0.5, rule_match="ext")
    elif suffix not in KNOWLEDGE_EXTS:
        return ClassificationResult("unknown", 0.0)

    # 2. Pattern matching over full text sample
    scores = {"prd": 0, "transcript": 0, "research_paper": 0, "knowledge_extract": 0}

    for pat in _PR_D_PATTERNS:
        if re.search(pat, content, re.IGNORECASE):
            scores["prd"] += 1
            break  # one match enough

    for pat in _TRANSCRIPT_PATTERNS:
        if re.search(pat, content, re.MULTILINE):
            scores["transcript"] += 2  # transcripts have strong markers
            break

    for pat in _RESEARCH_PAPER_PATTERNS:
        if re.search(pat, content, re.IGNORECASE):
            scores["research_paper"] += 1
            break

    for pat in _KNOWLEDGE_EXTRACT_PATTERNS:
        if re.search(pat, content, re.IGNORECASE):
            scores["knowledge_extract"] += 1

    # Pick max
    best = max(scores, key=scores.get)
    best_score = scores[best]

    if best_score > 0:
        confidence = min(0.5 + best_score * 0.15, 0.95)
        return ClassificationResult(best, confidence, rule_match="patterns")
    else:
        # No rule matched — fall back to LLM if available
        if llm_fallback is not None:
            try:
                llm_cat = llm_fallback(content[:8000])
                # Map LLM output to our categories
                mapping = {
                    "PRD": "prd",
                    "product_requirements": "prd",
                    "transcript": "transcript",
                    "conversation": "transcript",
                    "research": "research_paper",
                    "paper": "research_paper",
                    "knowledge": "knowledge_extract",
                    "notes": "knowledge_extract",
                }
                normalized = mapping.get(llm_cat.lower(), "unknown")
                return ClassificationResult(normalized, 0.75, llm_used=True)
            except Exception:
                pass

        return ClassificationResult("unknown", 0.1)


# ── Batch helper ─────────────────────────────────────────────────────────────


def classify_directory(
    root: Path, recursive: bool = True, llm_fallback=None
) -> List[ClassificationResult]:
    """Walk a directory and classify every file."""
    results = []
    pattern = "**/*" if recursive else "*"
    for file in root.glob(pattern):
        if file.is_file():
            result = classify_file(file, llm_fallback=llm_fallback)
            results.append(result)
    return results


# ── CLI (debug) ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m aaa_memory.classifier.rules <path>")
        sys.exit(1)

    path = Path(sys.argv[1])
    res = classify_file(path)
    print(
        json.dumps(
            {
                "category": res.category,
                "confidence": res.confidence,
                "rule_match": res.rule_match,
                "llm_used": res.llm_used,
            },
            indent=2,
        )
    )
