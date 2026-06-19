"""
Key Decision Extractor — summarizes decisions from session transcripts.

For each session, identifies:
- Key decisions made
- Action items created
- Architecture choices
- Rejected alternatives
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Decision:
    title: str
    description: str
    category: str  # "architecture", "process", "tooling", "design", "other"
    confidence: float  # 0.0 - 1.0
    source_turn_id: Optional[str] = None


# Patterns that indicate decisions in transcripts
DECISION_PATTERNS = [
    r"(?:we|I|let's)\s+(?:decided?|chose?|opted?|settled?\s+on|went\s+with)\s+",
    r"(?:decision|conclusion|resolution):\s*",
    r"(?:the\s+(?:best|right|correct)\s+(?:approach|way|solution))\s+(?:is|would\s+be)\s+",
    r"(?:going\s+to\s+use|will\s+(?:use|implement|adopt))\s+",
    r"(?:instead\s+of|rather\s+than)\s+\w+.*?(?:we|I|we'll|I'll)\s+",
    r"(?:rejected|dismissed|ruled\s+out|scrapped)\s+",
    r"(?:final\s+answer|conclusion|summary):",
    r"(?:a|c)ction\s+item:",
    r"(?:TODO|FIXME|HACK|XXX):",
]


def extract_decisions(transcript: str, turn_ids: Optional[List[str]] = None) -> List[Decision]:
    """Extract decisions from a session transcript using pattern matching."""
    decisions = []
    lines = transcript.split("\n")

    for i, line in enumerate(lines):
        for pattern in DECISION_PATTERNS:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                # Extract context around the match
                start = max(0, match.start() - 40)
                end = min(len(line), match.end() + 120)
                context = line[start:end].strip()

                # Categorize
                category = _categorize(line)
                confidence = _estimate_confidence(line, match)

                title = line[match.end():match.end() + 60].strip().rstrip(".,!?")
                if not title:
                    title = context[:60].strip()

                decision = Decision(
                    title=title,
                    description=context,
                    category=category,
                    confidence=confidence,
                    source_turn_id=turn_ids[i] if turn_ids and i < len(turn_ids) else None,
                )
                decisions.append(decision)
                break  # One pattern match per line

    # Deduplicate by similar titles
    unique = []
    seen = set()
    for d in decisions:
        key = d.title.lower().strip()[:40]
        if key not in seen:
            seen.add(key)
            unique.append(d)

    return unique


def _categorize(text: str) -> str:
    """Categorize a decision by keyword analysis."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["architecture", "pattern", "component", "module", "service", "api"]):
        return "architecture"
    if any(w in text_lower for w in ["deploy", "workflow", "process", "pipeline", "ci", "cd"]):
        return "process"
    if any(w in text_lower for w in ["tool", "library", "package", "framework", "sdk", "cli"]):
        return "tooling"
    if any(w in text_lower for w in ["design", "ui", "ux", "layout", "style", "color"]):
        return "design"
    return "other"


def _estimate_confidence(line: str, match: re.Match) -> float:
    """Estimate confidence based on language strength."""
    text_lower = line.lower()
    # Strong language
    if any(w in text_lower for w in ["decided", "final", "will", "must", "going to"]):
        return 0.9
    # Conditional language
    if any(w in text_lower for w in ["maybe", "perhaps", "consider", "might", "could"]):
        return 0.4
    # Moderate
    if any(w in text_lower for w in ["should", "plan", "propose", "suggest"]):
        return 0.6
    return 0.7


def format_decisions_markdown(decisions: List[Decision]) -> str:
    """Format decisions as markdown."""
    lines = ["## Key Decisions\n"]
    for i, d in enumerate(decisions, 1):
        conf = "🟢" if d.confidence >= 0.7 else "🟡" if d.confidence >= 0.4 else "🔴"
        lines.append(f"### {i}. {d.title}")
        lines.append(f"**Category**: {d.category} | **Confidence**: {conf} ({d.confidence:.0%})")
        lines.append(f"\n{d.description}\n")
    return "\n".join(lines)
