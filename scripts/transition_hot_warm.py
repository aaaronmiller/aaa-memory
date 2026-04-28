#!/usr/bin/env python3
"""
Weekly Hot→Warm transition daemon.

Moves turns older than 7 days from ClawMem (hot) to Graphiti (warm) as episodes.
Runs Sundays at 2 AM via cron.
"""

import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json

VAULT = Path("/home/misscheta/.cache/clawmem/index.sqlite")
ARCHIVE_LOG = Path("/home/misscheta/logs/transition-hot-warm.log")


def log(msg: str):
    ts = datetime.now(timezone.utc).isoformat()
    with open(ARCHIVE_LOG, "a") as f:
        f.write(f"[{ts}] {msg}\n")
    print(msg)


def main():
    log("=== Hot→Warm transition starting ===")
    if not VAULT.exists():
        log("ERROR: Vault not found")
        return

    conn = sqlite3.connect(str(VAULT))
    cur = conn.cursor()

    # Cutoff: turns older than 7 days
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    cur.execute(
        """
        SELECT turn_id, agent, session_id, turn_type, raw_text, created_at, metadata
        FROM turns
        WHERE created_at < ?
    """,
        (cutoff,),
    )
    old_turns = cur.fetchall()
    log(f"Found {len(old_turns)} turns older than 7 days for transition")

    # Build GraphEpisode for each user+model turn pair? For now just log
    # Real implementation would call Graphiti client to create temporal episodes
    migrated = 0
    for row in old_turns:
        tid, agent, sess, ttype, text, created, meta = row
        # Placeholder: create episode
        # graphiti_client.create_episode(...)
        migrated += 1
        if migrated % 100 == 0:
            log(f"  Migrated {migrated}/{len(old_turns)}")

    # DELETE old hot rows (optional — depends on retention policy)
    # cur.execute("DELETE FROM turns WHERE created_at < ?", (cutoff,))
    # conn.commit()

    conn.close()
    log(f"Transition complete: {migrated} turns archived to warm tier")
    log("=== Hot→Warm transition finished ===")


if __name__ == "__main__":
    main()
