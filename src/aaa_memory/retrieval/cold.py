"""
Cold tier retrieval — compressed SQLite FTS5 for long-term archives.

Turns older than 30 days are moved here from the hot tier.
Uses the same FTS5 schema as hot tier for consistent search.
"""

import sqlite3
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

COLD_VAULT = Path(
    config.COLD_VAULT
)
TOP_K = 10


def _ensure_schema():
    """Create cold storage schema if it doesn't exist."""
    COLD_VAULT.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(COLD_VAULT))
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS turns_archive USING fts5(
            turn_id, agent, raw_text, project, session_id,
            tokenize='porter unicode61'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS archive_meta (
            turn_id TEXT PRIMARY KEY,
            archived_at TEXT DEFAULT (datetime('now')),
            original_vault TEXT,
            source_file TEXT
        )
    """)
    conn.commit()
    conn.close()


def search_archive(query: str, limit: int = TOP_K) -> List[Dict]:
    """Search archived (cold) turns via FTS5.

    Returns list of dicts with keys: turn_id, agent, raw_text, project, score
    """
    _ensure_schema()
    if not COLD_VAULT.exists():
        return []

    conn = sqlite3.connect(str(COLD_VAULT))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        cur.execute(
            "SELECT turn_id, agent, raw_text, project, rank "
            "FROM turns_archive WHERE turns_archive MATCH ? "
            "ORDER BY rank LIMIT ?",
            (query, limit),
        )
        results = []
        for row in cur.fetchall():
            results.append({
                "turn_id": row["turn_id"],
                "agent": row["agent"],
                "raw_text": row["raw_text"],
                "project": row["project"],
                "score": 1.0 / (1.0 + float(row["rank"])) if row["rank"] else 0.0,
                "tier": "cold",
            })
        return results
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def archive_turns(turn_ids: List[str], source_vault: str) -> int:
    """Move turns from hot vault to cold archive.

    Returns number of turns archived.
    """
    _ensure_schema()
    if not Path(source_vault).exists():
        return 0

    src = sqlite3.connect(source_vault)
    src.row_factory = sqlite3.Row
    dst = sqlite3.connect(str(COLD_VAULT))
    count = 0

    for tid in turn_ids:
        cur = src.execute("SELECT turn_id, agent, raw_text, project, session_id FROM turns WHERE turn_id = ?", (tid,))
        row = cur.fetchone()
        if not row:
            continue
        try:
            dst.execute(
                "INSERT INTO turns_archive (turn_id, agent, raw_text, project, session_id) VALUES (?, ?, ?, ?, ?)",
                (row["turn_id"], row["agent"], row["raw_text"], row["project"], row["session_id"]),
            )
            dst.execute(
                "INSERT OR IGNORE INTO archive_meta (turn_id, original_vault) VALUES (?, ?)",
                (tid, source_vault),
            )
            count += 1
        except sqlite3.IntegrityError:
            continue

    dst.commit()
    dst.close()
    src.close()
    return count
