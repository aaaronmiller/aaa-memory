"""
OpenCode integration — parse ses_*.json session files.
"""

import json
from pathlib import Path
from typing import Iterator
from aaa_memory.models import Turn


def parse_opencode_sessions(sessions_dir: Path) -> Iterator[Turn]:
    """
    OpenCode stores sessions as JSON files: ses_<timestamp>.json
    Each contains a list of message objects: {role, content, timestamp}
    """
    for ses in sessions_dir.glob("ses_*.json"):
        data = json.loads(ses.read_text())
        for i, msg in enumerate(data.get("messages", [])):
            yield Turn(
                turn_id=f"opencode:{ses.name}:{i}",
                agent="opencode",
                session_id=ses.stem,
                turn_index=i,
                turn_type=msg.get("role", "user"),
                raw_text=msg.get("content", ""),
                created_at=msg.get("ts", ""),
                metadata=json.dumps({"file": str(ses)}),
            )


if __name__ == "__main__":
    import sys

    d = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path.home() / ".opencode" / "sessions"
    )
    count = 0
    for t in parse_opencode_sessions(d):
        print(t.turn_id, t.raw_text[:60])
        count += 1
    print(f"Total turns: {count}")
