"""
Session classifier — infers project_id, session_type, key decisions from parsed turns.
"""

import re
from typing import Dict, Tuple, Optional
from aaa_memory.models import Turn

# Project inference: known path patterns
PROJECT_HINTS = {
    "data-kiln": ["data-kiln", "datakiln"],
    "aaa-memory": ["aaa-memory", "aaa_memory"],
    "clawdi": ["clawdi"],
}

# Session type keywords
TYPE_KEYWORDS = {
    "planning": ["plan", "architecture", "spec", "requirement", "design", "structure"],
    "coding": [
        "implement",
        "refactor",
        "fix",
        "bug",
        "function",
        "class",
        "def ",
        "import ",
    ],
    "debugging": ["error", "exception", "traceback", "bug", "fix", "issue", "failed"],
    "testing": ["test", "spec", "assert", "pytest", "unit test", "e2e"],
    "docs": ["document", "readme", "docstring", "comment"],
    "audit": ["audit", "review", "security", "performance"],
}


def infer_project(cwd: Optional[str], filepath: Optional[str], content: str) -> str:
    """Guess project from cwd, file path, or content keywords."""
    combined = (cwd or "") + (filepath or "") + content[:200]
    combined_lower = combined.lower()
    for project, hints in PROJECT_HINTS.items():
        if any(hint in combined_lower for hint in hints):
            return project
    return "unknown"


def infer_session_type(turns: list[Turn]) -> str:
    """Classify session type from turn content."""
    # Collect first N user turns
    sample = " ".join(t.raw_text.lower() for t in turns[:10] if t.turn_type == "user")
    scores = {}
    for stype, keywords in TYPE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in sample)
        scores[stype] = hits
    if not scores:
        return "general"
    return max(scores, key=scores.get)


def extract_key_decisions(turns: list[Turn], max_decisions: int = 5) -> list[str]:
    """Extract high-level decisions from session using simple heuristics + LLM placeholder."""
    decisions = []
    # Simple pattern: "Decision:", "We will", "I'll...", "Let's..."
    decision_re = re.compile(
        r"(?:Decision:|We will|I decide|Let\'s|Chosen|Selected)\s+(.{10,200})",
        re.IGNORECASE,
    )
    for turn in turns:
        if turn.turn_type != "user":
            continue
        for match in decision_re.finditer(turn.raw_text):
            decisions.append(match.group(0).strip())
            if len(decisions) >= max_decisions:
                return decisions
    return decisions


def classify_session(
    session_id: str,
    turns: list[Turn],
    cwd: Optional[str] = None,
    filepath: Optional[str] = None,
) -> Dict:
    """
    Full classification for a session.

    Returns dict with: project_id, session_type, key_decisions, turn_count
    """
    # Sample content from first few turns
    content_blob = " ".join(t.raw_text for t in turns[:5])

    project_id = infer_project(cwd, filepath, content_blob)
    session_type = infer_session_type(turns)
    key_decisions = extract_key_decisions(turns)

    return {
        "session_id": session_id,
        "project_id": project_id,
        "session_type": session_type,
        "key_decisions": key_decisions,
        "turn_count": len(turns),
        "first_turn": turns[0].created_at if turns else None,
        "last_turn": turns[-1].created_at if turns else None,
    }


if __name__ == "__main__":
    # Demo: classify a sample session file
    import sys
    from aaa_memory.audit.parser import parse_file

    path = Path(sys.argv[1])
    turns = list(parse_file(path))
    result = classify_session(path.stem, turns)
    print(json.dumps(result, indent=2))
