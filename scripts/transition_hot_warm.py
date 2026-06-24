#!/usr/bin/env python3
"""Weekly Hot->Warm transition: marks turns older than 7 days as warm."""
import sqlite3, json, os
from pathlib import Path

VAULT = Path(os.getenv("AAA_MEMORY_VAULT", Path.home() / ".cache/aaa-memory/vault.sqlite"))


def _project_from_metadata(raw: str | None) -> str:
    if not raw:
        return "default"
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError:
        return "default"
    return str(metadata.get("project") or metadata.get("cwd") or "default")

def run(dry_run: bool = False):
    print("=" * 50)
    print("Hot -> Warm Transition")
    print(f"{'[DRY RUN]' if dry_run else '[LIVE]'}")
    print("=" * 50)
    
    if not VAULT.exists():
        print("No vault found")
        return
    
    conn = sqlite3.connect(str(VAULT))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tier_warm (
            turn_id TEXT PRIMARY KEY,
            session_id TEXT,
            agent TEXT,
            project TEXT,
            raw_text TEXT,
            created_at TEXT,
            metadata TEXT,
            transitioned_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # Find turns older than 7 days that have not already been promoted.
    cur.execute("""
        SELECT t.turn_id, t.session_id, t.agent, t.raw_text, t.created_at, t.metadata
        FROM turns t
        LEFT JOIN tier_warm w ON w.turn_id = t.turn_id
        WHERE t.created_at < datetime('now', '-7 days')
          AND w.turn_id IS NULL
    """)
    aging = cur.fetchall()
    print(f"Found {len(aging)} turns older than 7 days")
    
    if not dry_run:
        # Mark as warm tier
        for row in aging:
            conn.execute("""
                INSERT OR IGNORE INTO tier_warm
                    (turn_id, session_id, agent, project, raw_text, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                row["turn_id"],
                row["session_id"],
                row["agent"],
                _project_from_metadata(row["metadata"]),
                row["raw_text"],
                row["created_at"],
                row["metadata"],
            ))
        conn.commit()
        print(f"Transitioned {len(aging)} turns to warm tier")
    else:
        print(f"Would transition {len(aging)} turns")
    
    conn.close()

if __name__ == "__main__":
    import sys
    run(dry_run="--dry-run" in sys.argv)
