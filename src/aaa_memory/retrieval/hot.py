"""Hot tier retrieval — ClawMem (SQLite FTS5 + sqlite-vec hybrid)."""
from aaa_memory import config

import sqlite3
import json
import os
from pathlib import Path
from typing import List, Dict
from aaa_memory.models import Turn

VAULT = Path(
    os.getenv("AAA_MEMORY_VAULT", os.getenv("HOME") + "/.cache/aaa-memory/vault.sqlite")
)
TOP_K = 20


def search(query: str, limit: int = TOP_K) -> List[Dict]:
    """Full-text search with FTS5 auto-creation and LIKE fallback."""
    vp = config.get_vault()
    if not vp.exists():
        return []

    conn = sqlite3.connect(str(vp))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Create FTS5 table and sync if needed
    try:
        cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(turn_id UNINDEXED, agent, raw_text, tokenize='porter unicode61')")
        cur.execute("SELECT COUNT(*) FROM turns")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM turns_fts")
        indexed = cur.fetchone()[0]
        if indexed < total:
            cur.execute("INSERT OR IGNORE INTO turns_fts (turn_id, agent, raw_text) SELECT turn_id, agent, raw_text FROM turns")
            conn.commit()
    except sqlite3.OperationalError:
        pass

    # Try FTS5
    try:
        # Escape FTS5 special chars
        import re
        safe_query = re.sub(r'[\[\](){}*?:]', ' ', query).strip()
        if safe_query:
            # FTS5: split multi-word into OR terms for broader matching
            terms = [t for t in safe_query.split() if len(t) > 1]
            if len(terms) > 1:
                fts_query = " OR ".join(terms)
            else:
                fts_query = safe_query
            cur.execute("SELECT turn_id, agent, raw_text, created_at, rank FROM turns_fts WHERE turns_fts MATCH ? ORDER BY rank LIMIT ?", (fts_query, limit))
            fts_results = [{"turn_id": r["turn_id"], "agent": r["agent"], "raw_text": r["raw_text"], "score": 1.0 / (1.0 + float(r["rank"])) if r["rank"] else 0.0, "tier": "hot"} for r in cur.fetchall()]
            if fts_results:
                return fts_results
    except sqlite3.OperationalError:
        pass

    # LIKE fallback
    try:
        like = f"%{query}%"
        cur.execute("SELECT turn_id, agent, raw_text, created_at FROM turns WHERE raw_text LIKE ? LIMIT ?", (like, limit))
        rows = cur.fetchall()
        results = [{"turn_id": r["turn_id"], "agent": r["agent"], "raw_text": r["raw_text"], "score": 1.0, "tier": "hot"} for r in rows]
        return results
    except sqlite3.OperationalError as e:
        return []
    finally:
        conn.close()
if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "memory"
    res = search(q)
    for r in res[:5]:
        print(f"{r['turn_id']} — score={r['score']:.3f}")
        print(f"  {r['raw_text'][:100]}\n")
