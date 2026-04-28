"""
Hot tier retrieval — ClawMem (SQLite FTS5 + sqlite-vec hybrid).
"""

import sqlite3
import json
import os
from pathlib import Path
from typing import List, Dict
from aaa_memory.models import Turn

VAULT = Path(
    os.getenv("AAA_MEMORY_VAULT", "/home/misscheta/.cache/aaa-memory/vault.sqlite")
)
TOP_K = 20


def search(query: str, limit: int = TOP_K) -> List[Dict]:
    """
    Full-text search over turns via FTS5.

    Returns list of {'turn_id', 'agent', 'raw_text', 'score'} ordered by BM25.
    """
    if not VAULT.exists():
        return []

    conn = sqlite3.connect(str(VAULT))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Verify turns_fts exists
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='turns_fts'"
    )
    if not cur.fetchone():
        conn.close()
        return []

    # FTS5 query with BM25 ranking (lower is better)
    cur.execute(
        """
        SELECT turn_id, raw_text, bm25(turns_fts) AS bm25_score
        FROM turns_fts
        WHERE turns_fts MATCH ?
        ORDER BY bm25_score
        LIMIT ?
    """,
        (query, limit * 2),
    )
    rows = cur.fetchall()

    results = []
    for r in rows:
        # Normalize score: BM25 lower is better → convert to 0-1 where higher=better
        bm25 = r["bm25_score"]
        score = 1.0 / (1.0 + max(0, bm25))  # simple normalization
        results.append(
            {
                "turn_id": r["turn_id"],
                "agent": "unknown",  # could join turns table for actual agent
                "raw_text": r["raw_text"][:300],
                "score": round(score, 4),
            }
        )

    conn.close()
    return results


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "memory"
    res = search(q)
    for r in res[:5]:
        print(f"{r['turn_id']} — score={r['score']:.3f}")
        print(f"  {r['raw_text'][:100]}\n")
