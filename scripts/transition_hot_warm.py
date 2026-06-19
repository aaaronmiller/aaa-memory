#!/usr/bin/env python3
"""Weekly Hot→Warm transition: moves turns >7 days old to warm tier (Graphiti)."""
import sqlite3, json, os
from pathlib import Path

VAULT = Path(os.getenv("AAA_MEMORY_VAULT", Path.home() / ".cache/aaa-memory/vault.sqlite"))

def run(dry_run: bool = False):
    print("=" * 50)
    print("Hot → Warm Transition")
    print(f"{'[DRY RUN]' if dry_run else '[LIVE]'}")
    print("=" * 50)
    
    if not VAULT.exists():
        print("No vault found")
        return
    
    conn = sqlite3.connect(str(VAULT))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Find turns older than 7 days
    cur.execute("SELECT turn_id, session_id, agent, project, raw_text FROM turns WHERE timestamp < datetime('now', '-7 days')")
    aging = cur.fetchall()
    print(f"Found {len(aging)} turns older than 7 days")
    
    if not dry_run:
        # Mark as warm tier
        conn.execute("CREATE TABLE IF NOT EXISTS tier_warm (turn_id TEXT PRIMARY KEY, session_id TEXT, agent TEXT, project TEXT, raw_text TEXT, transitioned_at TEXT DEFAULT (datetime('now')))")
        for row in aging:
            conn.execute("INSERT OR IGNORE INTO tier_warm (turn_id, session_id, agent, project, raw_text) VALUES (?, ?, ?, ?, ?)",
                        (row["turn_id"], row["session_id"], row["agent"], row["project"], row["raw_text"]))
        conn.commit()
        print(f"Transitioned {len(aging)} turns to warm tier")
    else:
        print(f"Would transition {len(aging)} turns")
    
    conn.close()

if __name__ == "__main__":
    import sys
    run(dry_run="--dry-run" in sys.argv)
