"""Claude Code lifecycle hooks for session capture.
Registers with ClawMem hook system."""
import json, os, sqlite3, uuid
from pathlib import Path
from datetime import datetime

VAULT = Path(os.getenv("AAA_MEMORY_VAULT", Path.home() / ".cache/aaa-memory/vault.sqlite"))

def store_turn(agent: str, prompt: str, response: str, session_id: Optional[str] = None, project: Optional[str] = None):
    """Store a conversation turn."""
    VAULT.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(VAULT))
    conn.execute("CREATE TABLE IF NOT EXISTS turns (turn_id TEXT PRIMARY KEY, session_id TEXT, agent TEXT, project TEXT, timestamp TEXT, prompt TEXT, response TEXT, raw_text TEXT)")
    tid = str(uuid.uuid4())
    sid = session_id or tid
    raw = f"Human: {prompt}\n\nAssistant: {response}"
    conn.execute("INSERT OR IGNORE INTO turns (turn_id, session_id, agent, project, timestamp, raw_text) VALUES (?, ?, ?, ?, ?, ?)",
                 (tid, sid, agent, project or os.getcwd().split("/")[-1], datetime.utcnow().isoformat(), raw))
    conn.commit()
    conn.close()
    return tid
