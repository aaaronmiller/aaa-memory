"""
Result formatter — three expansion levels for progressive disclosure.

Levels:
  collapsed: icon + project + model + date + first 80 chars (~300 tokens)
  summary: full summary + topic labels + intent + failure mode
  full: markdown with tool calls, files touched, [[wikilinks]], thread view
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class SearchResult:
    turn_id: str
    agent: str
    project: Optional[str]
    raw_text: str
    created_at: str
    score: float
    source_tiers: list[str]  # which tiers contributed
    metadata: Optional[dict] = None


def format_collapsed(result: SearchResult, max_chars: int = 120) -> dict:
    """
    Discord embed-style collapsed card.
    Returns dict suitable for json.dumps → embed builder.
    """
    text = result.raw_text.strip().replace("\n", " ")
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    dt = datetime.fromisoformat(result.created_at.replace("Z", "+00:00"))
    return {
        "title": f"[[{result.project or 'unknown'}]] {result.agent} • {dt:%b %d %H:%M}",
        "description": text,
        "color": 0x5865F2
        if result.source_tiers[0] == "hot"
        else 0x57F287
        if result.source_tiers[0] == "warm"
        else 0xED4245,
        "fields": [
            {"name": "Tier", "value": ", ".join(result.source_tiers), "inline": True},
            {"name": "Score", "value": f"{result.score:.3f}", "inline": True},
        ],
    }


def format_summary(
    result: SearchResult,
    summary: Optional[str] = None,
    topics: Optional[list[str]] = None,
) -> dict:
    """
    Summary-level card — adds summary and topic labels.
    """
    base = format_collapsed(result, max_chars=300)
    base["description"] = summary or result.raw_text[:400] + "..."
    if topics:
        base["fields"].append(
            {"name": "Topics", "value": ", ".join(topics), "inline": False}
        )
    base["fields"].append(
        {
            "name": "Intent",
            "value": result.metadata.get("intent", "?") if result.metadata else "?",
            "inline": True,
        }
    )
    return base


def format_full(result: SearchResult, include_raw: bool = True) -> str:
    """
    Full markdown thread with spoilers and citations.
    """
    dt = datetime.fromisoformat(result.created_at.replace("Z", "+00:00"))
    header = f"## {result.agent} • {dt:%Y-%m-%d %H:%M:%S} • Score: {result.score:.3f}\n"
    body = result.raw_text
    if include_raw:
        body = f"||{body}||"  # Discord spoiler
    footer = f"\n\n*Source: turn://{result.turn_id}*"
    return header + body + footer


# ── Progressive disclosure pipeline ──────────────────────────────────────────


def render_progressive(
    results: list[SearchResult],
    level: str = "collapsed",
    summaries: dict = None,
    topics: dict = None,
):
    """
    Format a list of results for display.

    Parameters
    ----------
    results : list[SearchResult]
    level : "collapsed" | "summary" | "full"
    summaries : dict[turn_id → summary_str] (required for summary level)
    topics : dict[turn_id → list[str]] (optional for summary level)

    Returns
    -------
    list[dict|str] formatted for UI layer (Discord embeds / web)
    """
    out = []
    for r in results:
        if level == "collapsed":
            out.append(format_collapsed(r))
        elif level == "summary":
            out.append(
                format_summary(
                    r,
                    summary=summaries.get(r.turn_id) if summaries else None,
                    topics=topics.get(r.turn_id) if topics else None,
                )
            )
        elif level == "full":
            out.append(format_full(r))
        else:
            raise ValueError(f"Unknown level: {level}")
    return out


if __name__ == "__main__":
    # Demo
    sample = SearchResult(
        turn_id="test-001",
        agent="claude-code",
        project="aaa-memory",
        raw_text="We should use SQLite because it's zero-ops and file-portable. Embeddings stored in sqlite-vec table.",
        created_at="2026-04-27T18:00:00Z",
        score=0.92,
        source_tiers=["hot"],
        metadata={"intent": "recent"},
    )
    print("Collapsed:", json.dumps(format_collapsed(sample), indent=2))
    print(
        "\nSummary:",
        json.dumps(
            format_summary(
                sample,
                summary="Decision made to use SQLite for storage engine.",
                topics=["database", "sqlite"],
            ),
            indent=2,
        ),
    )
    print("\nFull:\n", format_full(sample))
