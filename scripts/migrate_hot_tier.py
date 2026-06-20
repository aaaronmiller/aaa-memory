#!/usr/bin/env python3
"""Step 1: Migrate wiki-memory hot tier (memory.json) into aaa-memory vault.

Adds a hot_memories table to the vault and imports existing records.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

VAULT = Path.home() / ".cache" / "aaa-memory" / "vault.sqlite"
MEMORY_JSON = Path.home() / ".local" / "share" / "ai-wiki" / ".meta" / "memory.json"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _new_id():
    return f"mem-{uuid.uuid4().hex[:16]}"


def create_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hot_memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            tags TEXT DEFAULT '[]',
            project TEXT DEFAULT 'default',
            source TEXT DEFAULT 'unknown',
            pinned INTEGER DEFAULT 0,
            created TEXT NOT NULL,
            accessed TEXT NOT NULL,
            access_count INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_hot_project ON hot_memories(project)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_hot_created ON hot_memories(created)
    """)
    conn.commit()
    print("✓ hot_memories table created")


def migrate_json(conn, json_path):
    if not json_path.exists():
        print(f"  {json_path} not found, skipping")
        return 0

    records = json.loads(json_path.read_text())
    if not isinstance(records, list):
        records = []

    now = _now()
    inserted = 0
    for r in records:
        if not isinstance(r, dict) or not r.get("content"):
            continue
        rid = r.get("id") or _new_id()
        tags = r.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        conn.execute("""
            INSERT OR IGNORE INTO hot_memories
            (id, content, tags, project, source, pinned, created, accessed, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rid,
            r["content"].strip(),
            json.dumps(tags),
            r.get("project", "default"),
            r.get("source", "unknown"),
            1 if r.get("pinned") else 0,
            r.get("created") or r.get("timestamp") or now,
            r.get("accessed") or r.get("created") or now,
            int(r.get("access_count", 0)),
        ))
        inserted += 1

    conn.commit()
    return inserted


def seed_from_wiki_memory(conn):
    """Seed additional memories from wiki-memory's known facts."""
    seeds = [
        ("wiki-memory store lives in ~/.local/share/ai-wiki", ["wiki-memory", "storage"], "wiki-memory", "system"),
        ("codex capture works live", ["codex", "capture"], "wiki-memory", "system"),
        ("aaa-memory vault is at ~/.cache/aaa-memory/vault.sqlite", ["aaa-memory", "vault"], "aaa-memory", "system"),
        ("Hermes config at ~/.hermes/config.yaml, gateway PID managed by systemd", ["hermes", "config"], "hermes", "system"),
        ("model-scan scores refreshed 2026-06-19, 540 models in AA cache", ["model-scan", "scores"], "model-scan", "system"),
        ("model-scan auto-update runs weekly via systemd timer", ["model-scan", "cron"], "model-scan", "system"),
        ("deepseek-v4-flash-free is the current best free model (AI 40.3)", ["model", "deepseek", "free"], "model-scan", "system"),
        ("minimax-m3-free and qwen3.6-plus-free free promos ended", ["model", "minimax", "qwen"], "model-scan", "system"),
        ("OpenRouter: 27 free models confirmed", ["openrouter", "free"], "model-scan", "system"),
        ("Zen: only 3 free models working (deepseek, mimo, nemotron)", ["zen", "free"], "model-scan", "system"),
    ]

    now = _now()
    inserted = 0
    for content, tags, project, source in seeds:
        conn.execute("""
            INSERT OR IGNORE INTO hot_memories
            (id, content, tags, project, source, pinned, created, accessed, access_count)
            VALUES (?, ?, ?, ?, ?, 0, ?, ?, 0)
        """, (_new_id(), content, json.dumps(tags), project, source, now, now))
        inserted += 1

    conn.commit()
    return inserted


def verify(conn):
    count = conn.execute("SELECT COUNT(*) FROM hot_memories").fetchone()[0]
    print(f"\n✓ Verification: {count} hot memories in vault")
    rows = conn.execute("SELECT id, project, source, content FROM hot_memories LIMIT 10").fetchall()
    for r in rows:
        print(f"  [{r[2]}:{r[1]}] {r[3][:70]}")


if __name__ == "__main__":
    print("=== Merging hot tiers ===\n")

    conn = sqlite3.connect(str(VAULT))
    create_table(conn)

    n = migrate_json(conn, MEMORY_JSON)
    print(f"  Migrated {n} records from memory.json")

    n2 = seed_from_wiki_memory(conn)
    print(f"  Seeded {n2} system memories")

    verify(conn)
    conn.close()
