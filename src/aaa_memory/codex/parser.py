"""
Codex CLI integration — parse JSONL rollout sessions.

Codex stores conversation history in JSONL files under ~/.codex/rollouts/
"""

import json
from pathlib import Path
from typing import Iterator
from aaa_memory.models import Turn


def parse_codex_rollouts(rollouts_dir: Path) -> Iterator[Turn]:
    for rl in rollouts_dir.glob("*.jsonl"):
        session_id = rl.stem
        with open(rl, "r") as f:
            for i, line in enumerate(f):
                try:
                    msg = json.loads(line)
                    yield Turn(
                        turn_id=f"codex:{rl.name}:{i}",
                        agent="codex",
                        session_id=session_id,
                        turn_index=i,
                        turn_type=msg.get("type", "user"),
                        raw_text=msg.get("text", ""),
                        created_at=msg.get("timestamp", ""),
                        metadata=json.dumps({"rollout": str(rl)}),
                    )
                except Exception:
                    continue


if __name__ == "__main__":
    import sys

    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / ".codex" / "rollouts"
    for t in parse_codex_rollouts(d):
        print(t.turn_id, t.raw_text[:50])
