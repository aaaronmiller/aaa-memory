#!/usr/bin/env python3
"""Tests for aaa-memory + wiki-memory integration."""

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

AAA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AAA_ROOT / "src"))


def test_vault_memory_store():
    """Test hot memory store backed by vault."""
    from aaa_memory.hot.mem_store import VaultMemoryStore

    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        vault = Path(f.name)

    # Create tables
    conn = sqlite3.connect(str(vault))
    conn.execute("""
        CREATE TABLE hot_memories (
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
    conn.commit()
    conn.close()

    store = VaultMemoryStore(vault_path=vault)

    # Test add
    rec = store.add("test memory", tags=["test"], project="test")
    assert rec is not None
    assert rec["content"] == "test memory"

    # Test dedup
    rec2 = store.add("test memory", project="test")
    assert rec2["id"] == rec["id"]

    # Test recall
    results = store.recall("test")
    assert len(results) >= 1

    # Test stats
    stats = store.stats()
    assert stats["total"] >= 1

    # Test forget
    n = store.forget("test memory")
    assert n >= 1

    vault.unlink()
    print("✓ vault_memory_store")


def test_dream_agent():
    """Test dream agent runs without errors."""
    from aaa_memory.warm.dream import run_dream_cycle, DreamReport

    report = run_dream_cycle(idle_seconds=10, verbose=False)
    assert isinstance(report, DreamReport)
    assert report.duration >= 0
    print("✓ dream_agent")


def test_cold_tier():
    """Test cold tier search (local fallback)."""
    from aaa_memory.cold import search_local

    # Should not crash even without data
    results = search_local("test query", limit=3)
    assert isinstance(results, list)
    print("✓ cold_tier")


def test_unified_mem():
    """Test unified mem.py CLI."""
    import subprocess

    result = subprocess.run(
        [sys.executable, str(AAA_ROOT / "scripts" / "mem.py"), "stats"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    stats = json.loads(result.stdout)
    assert "total" in stats
    print("✓ unified_mem")


def test_vault_exists():
    """Test vault file exists and has expected tables."""
    vault = Path.home() / ".cache" / "aaa-memory" / "vault.sqlite"
    assert vault.exists(), f"Vault not found at {vault}"

    conn = sqlite3.connect(str(vault))
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    assert "turns" in tables, "turns table missing"
    assert "hot_memories" in tables, "hot_memories table missing"
    assert "wiki_pages" in tables, "wiki_pages table missing"

    conn.close()
    print("✓ vault_exists")


if __name__ == "__main__":
    test_vault_exists()
    test_vault_memory_store()
    test_dream_agent()
    test_cold_tier()
    test_unified_mem()
    print("\n✅ All tests passed")
