"""Claude Code lifecycle hooks for session capture.
Uses the actual vault schema: id, turn_id, agent, session_id, turn_index, turn_type, raw_text, created_at, metadata."""
import json, os, sqlite3, uuid
from pathlib import Path
from datetime import datetime, timezone

VAULT = Path(os.getenv("AAA_MEMORY_VAULT", Path.home() / ".cache/aaa-memory/vault.sqlite"))

def _ensure_schema():
    """Create the turns table matching the actual vault schema."""
    VAULT.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(VAULT))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turn_id TEXT UNIQUE,
            agent TEXT,
            session_id TEXT,
            turn_index INTEGER DEFAULT 0,
            turn_type TEXT DEFAULT 'conversation',
            raw_text TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            metadata TEXT DEFAULT '{}'
        )
    """)
    conn.commit()
    conn.close()

def store_turn(agent: str, prompt: str, response: str, session_id: Optional[str] = None, project: Optional[str] = None) -> str:
    """Store a conversation turn into the vault."""
    _ensure_schema()
    conn = sqlite3.connect(str(VAULT))
    tid = str(uuid.uuid4())
    sid = session_id or tid
    raw = f"Human: {prompt}\n\nAssistant: {response}"
    meta = json.dumps({"project": project or os.getcwd().split("/")[-1]})
    conn.execute(
        "INSERT OR IGNORE INTO turns (turn_id, agent, session_id, raw_text, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (tid, agent, sid, raw, meta, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return tid
