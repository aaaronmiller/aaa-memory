#!/usr/bin/env python3
"""
Clawdi Daily Update Service — Runs at 2 AM daily.

Scans ~/knowledge/raw/ for new turn files, parses them, and appends
to the shared ClawMem SQLite vault at ~/.cache/clawmem/index.sqlite.

Also parses per-agent session files and normalizes to Turns.
"""

import sqlite3
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import time

VAULT_PATH = Path(
    os.getenv("AAA_MEMORY_VAULT", "/home/misscheta/.cache/aaa-memory/vault.sqlite")
)
RAW_BASE = Path("/home/misscheta/knowledge/raw")
LOG_PATH = Path("/home/misscheta/logs/memory-update.log")

# Ensure directories
VAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log(msg):
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] {msg}\n"
    with open(LOG_PATH, "a") as f:
        f.write(line)
    print(line, end="")


def connect():
    conn = sqlite3.connect(str(VAULT_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def ensure_schema(conn):
    """Create required tables if they don't exist."""
    cur = conn.cursor()
    # Minimal turns table — matches ClawMem schema
    cur.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turn_id TEXT UNIQUE NOT NULL,
            agent TEXT NOT NULL,
            session_id TEXT,
            turn_index INTEGER,
            turn_type TEXT NOT NULL,  -- 'user' | 'model' | 'system'
            raw_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata TEXT  -- JSON blob
        )
    """)
    # Create FTS5 virtual table for full-text search
    # Note: content=raw_text means FTS stores content internally; simpler: just list columns
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
            turn_id,
            raw_text,
            tokenize='unicode61'
        )
    """)
    # Triggers to keep FTS in sync
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
            INSERT INTO turns_fts(rowid, turn_id, raw_text)
            VALUES (new.rowid, new.turn_id, new.raw_text);
        END
    """)
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
            DELETE FROM turns_fts WHERE rowid = old.rowid;
        END
    """)
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS turns_au AFTER UPDATE ON turns BEGIN
            INSERT INTO turns_fts(turns_fts, rowid, turn_id, raw_text)
            VALUES('rebuild', new.rowid, new.turn_id, new.raw_text);
        END
    """)
    conn.commit()


def parse_jsonl_file(filepath: Path, agent: str, session_id: str = None):
    """Yield turn dicts from a JSONL file."""
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                turn = json.loads(line)
                yield turn
            except json.JSONDecodeError:
                continue


def normalize_turn(
    raw_turn: dict, agent: str, source_file: str, index: int, session_id: str = None
) -> dict:
    """Convert raw turn dict to normalized Turn schema."""
    messages = raw_turn.get("messages", [])
    if not messages:
        return None

    turns_out = []
    for i, msg in enumerate(messages):
        turn_id = f"{agent}:{source_file}:{index}:{i}"
        turn_type = msg.get("role", "user")
        if turn_type == "model":
            turn_type = "model"
        elif turn_type == "user":
            turn_type = "user"
        else:
            turn_type = "system"

        metadata = {
            "source_file": str(source_file),
            "platform": raw_turn.get("platform", agent),
            "model": msg.get("model"),
            "tags": raw_turn.get("tags", []),
            "capture_ts": raw_turn.get("timestamp"),
        }

        turns_out.append(
            {
                "turn_id": turn_id,
                "agent": agent,
                "session_id": session_id or f"session-{int(time.time())}",
                "turn_index": i,
                "turn_type": turn_type,
                "raw_text": msg.get("content", ""),
                "created_at": raw_turn.get(
                    "timestamp", datetime.now(timezone.utc).isoformat()
                ),
                "metadata": json.dumps(metadata),
            }
        )
    return turns_out


def scan_and_ingest(conn):
    """Walk raw/ directories and ingest new turn files."""
    cur = conn.cursor()
    ingested = 0
    skipped = 0

    # Walk all subdirs of raw/
    for agent_dir in RAW_BASE.iterdir():
        if not agent_dir.is_dir():
            continue
        agent = agent_dir.name  # e.g., 'web/chatgpt' -> take last segment
        for session_file in agent_dir.glob("**/*.jsonl"):
            # Check if already ingested via metadata tracking
            # For now, just process all files (idempotent on turn_id UNIQUE constraint)
            try:
                for raw_turn in parse_jsonl_file(
                    session_file, agent, str(session_file)
                ):
                    # Derive session_id from filename stem
                    sess_id = raw_turn.get("session_id", session_file.stem)
                    normalized = normalize_turn(
                        raw_turn, agent, str(session_file), 0, session_id=sess_id
                    )
                    if not normalized:
                        skipped += 1
                        continue
                    for turn in normalized:
                        try:
                            cur.execute(
                                """
                                INSERT OR IGNORE INTO turns
                                (turn_id, agent, session_id, turn_index, turn_type, raw_text, created_at, metadata)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    turn["turn_id"],
                                    turn["agent"],
                                    turn["session_id"],
                                    turn["turn_index"],
                                    turn["turn_type"],
                                    turn["raw_text"],
                                    turn["created_at"],
                                    turn["metadata"],
                                ),
                            )
                            if cur.rowcount > 0:
                                ingested += 1
                        except sqlite3.Error:
                            skipped += 1
            except Exception as e:
                log(f"ERROR processing {session_file}: {e}")
                skipped += 1

    conn.commit()
    log(f"Ingest complete: {ingested} turns added, {skipped} skipped/duplicates")
    return ingested


def main():
    log("=== Daily update service starting ===")
    try:
        conn = connect()
        ensure_schema(conn)
        log("Schema verified")
        count = scan_and_ingest(conn)
        log(f"Total new turns: {count}")
        conn.close()
    except Exception as e:
        log(f"FATAL: {e}")
        raise
    log("=== Daily update service complete ===")


if __name__ == "__main__":
    main()
