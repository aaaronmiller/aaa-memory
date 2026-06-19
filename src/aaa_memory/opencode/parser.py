"""OpenCode session parser — reads ses_*.json files."""
import json, glob
from pathlib import Path

OPCODE_DIR = Path.home() / ".opencode/sessions"

def parse_sessions() -> list[dict]:
    sessions = []
    for f in sorted(glob.glob(str(OPCODE_DIR / "ses_*.json")))[:10]:
        try:
            data = json.loads(Path(f).read_text())
            sessions.append({"file": f, "session_id": data.get("session_id"), "turns": len(data.get("messages", [])),"project": data.get("project", "")})
        except (json.JSONDecodeError, FileNotFoundError):
            continue
    return sessions
