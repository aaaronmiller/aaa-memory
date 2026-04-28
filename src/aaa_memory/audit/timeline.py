"""
Timeline assembler — chronological project view linking sessions.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict
from aaa_memory.audit.discover import discover_sessions
from aaa_memory.audit.parser import parse_file
from aaa_memory.audit.classify import classify_session

PROJECTS_DIR = Path("/home/misscheta/knowledge/projects")


def assemble_timeline(project_id: str, days: int = 7) -> Dict:
    """
    Build timeline for a project over recent days.

    1. Discover all sessions
    2. Filter to project_id & date range
    3. Sort chronologically
    4. Link sessions within 48h across agents
    5. Output markdown timeline + JSON index
    """
    all_sessions = discover_sessions()
    # Flatten: List[Dict] with metadata
    sessions = []
    cutoff = datetime.now().timestamp() - (days * 86400)

    for agent, files in all_sessions.items():
        for f in files:
            try:
                turns = list(parse_file(f))
                if not turns:
                    continue
                meta = classify_session(f.stem, turns, filepath=str(f))
                if meta["project_id"] != project_id:
                    continue
                # Date filter
                first_ts = (
                    datetime.fromisoformat(meta["first_turn"]).timestamp()
                    if meta["first_turn"]
                    else 0
                )
                if first_ts < cutoff:
                    continue
                sessions.append(
                    {
                        "agent": agent,
                        "file": str(f),
                        "session_id": meta["session_id"],
                        "start": meta["first_turn"],
                        "end": meta["last_turn"],
                        "type": meta["session_type"],
                        "decisions": meta["key_decisions"],
                        "turns": len(turns),
                        "links": [],  # to be filled
                    }
                )
            except Exception:
                pass

    # Sort
    sessions.sort(key=lambda s: s["start"] or "")

    # Link sessions within 48h window
    for i, s in enumerate(sessions):
        s_time = datetime.fromisoformat(s["start"])
        for j, other in enumerate(sessions):
            if i == j:
                continue
            o_time = datetime.fromisoformat(other["start"])
            if abs((s_time - o_time).total_seconds()) < 172800:  # 48h
                s["links"].append(other["session_id"])

    # Write outputs
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    proj_dir = PROJECTS_DIR / project_id
    proj_dir.mkdir(exist_ok=True)

    # Markdown timeline
    md_lines = [
        f"# Timeline: {project_id}\n",
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n",
        f"Showing last {days} days\n\n",
    ]
    for s in sessions:
        md_lines.append(f"## {s['session_id']} ({s['agent']}, {s['type']})\n")
        md_lines.append(f"- **When:** {s['start']} — {s['end']}\n")
        md_lines.append(f"- **Turns:** {s['turns']}\n")
        if s["decisions"]:
            md_lines.append("- **Key decisions:**\n")
            for d in s["decisions"]:
                md_lines.append(f"  - {d}\n")
        if s["links"]:
            md_lines.append(f"- **Related:** {', '.join(s['links'])}\n")
        md_lines.append("\n")

    (proj_dir / "timeline.md").write_text("".join(md_lines))

    # JSON index for MCP consumption
    index = {
        "project_id": project_id,
        "generated": datetime.now(timezone.utc).isoformat(),
        "sessions": sessions,
    }
    (proj_dir / "timeline.json").write_text(json.dumps(index, indent=2))

    print(f"Timeline written to {proj_dir}")
    return index


if __name__ == "__main__":
    import sys

    pid = sys.argv[1] if len(sys.argv) > 1 else "default"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    assemble_timeline(pid, days)
