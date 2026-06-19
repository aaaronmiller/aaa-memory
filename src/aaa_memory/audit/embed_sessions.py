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
    cur.execute("SELECT turn_id, agent, raw_text FROM turns WHERE session_id = ? ORDER BY timestamp", (session_id,))
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return {}
    
    # Concatenate transcript, strip bash noise
    transcript = "\n".join(r["raw_text"] for r in rows if r["raw_text"])
    
    # Extract decisions
    decisions = extract_decisions(transcript, [r["turn_id"] for r in rows])
    
    summary = {
        "session_id": session_id,
        "agent": rows[0]["agent"] if rows else "?",
        "turns": len(rows),
        "decision_count": len(decisions),
        "top_decisions": [d.title for d in decisions[:5]],
        "summary_md": format_decisions_markdown(decisions),
    }
    return summary
