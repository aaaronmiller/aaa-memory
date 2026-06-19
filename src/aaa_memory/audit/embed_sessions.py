"""Generate and store session summaries as embeddable elements."""
import sqlite3
from typing import List, Dict
from aaa_memory import config
from aaa_memory.audit.extract_decisions import extract_decisions, format_decisions_markdown


def summarize_session(session_id: str) -> Dict:
    """Create a compressed session summary with key decisions."""
    vp = config.get_vault()
    if not vp.exists():
        return {"session_id": session_id, "turns": 0}
    conn = sqlite3.connect(str(vp))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT id, turn_id, agent, raw_text, created_at FROM turns WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError as e:
        conn.close()
        return {"session_id": session_id, "turns": 0, "error": str(e)}
    conn.close()
    if not rows:
        return {"session_id": session_id, "turns": 0}

    transcript = "\n".join(r["raw_text"] for r in rows if r["raw_text"])
    decisions = extract_decisions(
        transcript, [r["turn_id"] for r in rows]
    )

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
    vp = config.get_vault()
    if not vp.exists():
        return []
    conn = sqlite3.connect(str(vp))
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT DISTINCT session_id FROM turns ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        session_ids = [r[0] for r in cur.fetchall()]
    except sqlite3.OperationalError as e:
        conn.close()
        return [{"error": str(e)}]
    conn.close()
    summaries = []
    for sid in session_ids:
        s = summarize_session(sid)
        if s and "error" not in s:
            summaries.append(s)
    return summaries
