#!/usr/bin/env python3
"""Monthly Warm→Cold transition: archives turns >90 days to cold storage."""
import sys
from pathlib import Path

VAULT = Path(os.getenv("AAA_MEMORY_VAULT", Path.home() / ".cache/aaa-memory/vault.sqlite"))

def run(dry_run: bool = False):
    print("=" * 50)
    print("Warm → Cold Transition")
    print(f"{'[DRY RUN]' if dry_run else '[LIVE]'}")
    print("=" * 50)
    
    import sqlite3
    if not VAULT.exists():
        print("No vault found")
        return
    
    conn = sqlite3.connect(str(VAULT))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Check if warm tier exists
    try:
        cur.execute("SELECT turn_id, raw_text FROM tier_warm WHERE transitioned_at < datetime('now', '-90 days')")
        aging = cur.fetchall()
    except sqlite3.OperationalError:
        print("No warm tier data found")
        conn.close()
        return
    
    print(f"Found {len(aging)} turns older than 90 days in warm tier")
    
    if not dry_run and aging:
        from aaa_memory.retrieval.cold import archive_turns
        turn_ids = [r["turn_id"] for r in aging]
        count = archive_turns(turn_ids, str(VAULT))
        # Remove from warm tier
        for tid in turn_ids:
            conn.execute("DELETE FROM tier_warm WHERE turn_id = ?", (tid,))
        conn.commit()
        print(f"Archived {count} turns to cold storage")
    elif dry_run:
        print(f"Would archive {len(aging)} turns to cold storage")
    
    conn.close()

if __name__ == "__main__":
    run(dry_run="--dry-run" in sys.argv)
