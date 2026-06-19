"""Codex CLI rollout parser — reads JSONL files."""
import json, glob
from pathlib import Path

CODEX_DIR = Path.home() / ".codex/rollouts"

def parse_rollouts() -> list[dict]:
    sessions = []
    for f in sorted(glob.glob(str(CODEX_DIR / "*.jsonl")))[:10]:
        try:
            turns = []
            for line in Path(f).read_text().strip().split("\n"):
                if line.strip():
                    turns.append(json.loads(line))
            sessions.append({"file": f, "turns": len(turns)})
        except (json.JSONDecodeError, FileNotFoundError):
            continue
    return sessions
