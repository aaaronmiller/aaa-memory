"""Generate and store session summaries as embeddable elements."""
import sqlite3, json, os
from pathlib import Path
from typing import List, Dict
from aaa_memory.audit.extract_decisions import extract_decisions, format_decisions_markdown

VAULT = Path(os.getenv("AAA_MEMORY_VAULT", Path.home() / ".cache/aaa-memory/vault.sqlite"))

def summarize_session(session_id: str) -> Dict:
    """Create a compressed session summary with key decisions."""
    if not VAULT.exists():
        return {}
    conn = sqlite3.connect(str(VAULT))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, turn_id, agent, raw_text, created_at FROM turns WHERE session_id = ? ORDER BY created_at", (session_id,))
        rows = cur.fetchall()
    except sqlite3.OperationalError as e:
        conn.close()
        return {"error": str(e)}
    conn.close()
    if not rows:
        return {"session_id": session_id, "turns": 0}
    
    transcript = "\n".join(r["raw_text"] for r in rows if r["raw_text"])
    decisions = extract_decisions(transcript, [r["turn_id"] for r in rows])
    
    return {
        "session_id": session_id,
        "agent": rows[0]["agent"] if rows else "?",
        "turns": len(rows),
        "first_seen": str(rows[0]["created_at"]) if rows[0]["created_at"] else "?",
        "last_seen": str(rows[-1]["created_at"]) if rows[-1]["created_at"] else "?",
        "decision_count": len(decisions),
        "top_decisions": [d.title for d in decisions[:5]],
        "summary_md": format_decisions_markdown(decisions),
    }

def summarize_all_sessions(limit: int = 20) -> List[Dict]:
    """Summarize all recent sessions."""
    if not VAULT.exists():
        return []
    conn = sqlite3.connect(str(VAULT))
    cur = conn.cursor()
    try:
        cur.execute("SELECT DISTINCT session_id FROM turns ORDER BY MAX(created_at) DESC LIMIT ?", (limit,))
        session_ids = [r[0] for r in cur.fetchall()]
    except sqlite3.OperationalError:
        conn.close()
        return []
    conn.close()
    return [summarize_session(sid) for sid in session_ids]
