import sqlite3
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _load_transition():
    spec = importlib.util.spec_from_file_location(
        "transition_hot_warm",
        ROOT / "scripts" / "transition_hot_warm.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _vault(path):
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turn_id TEXT UNIQUE NOT NULL,
            agent TEXT NOT NULL,
            session_id TEXT,
            turn_index INTEGER,
            turn_type TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata TEXT
        );
    """)
    conn.execute(
        "INSERT INTO turns (turn_id, agent, session_id, turn_index, turn_type, raw_text, created_at, metadata) VALUES (?, ?, ?, ?, ?, ?, datetime('now', '-8 days'), ?)",
        ("old-turn", "claude", "s1", 1, "user", "old memory text", '{"project": "aaa-memory"}'),
    )
    conn.commit()
    conn.close()


def test_hot_to_warm_uses_current_turns_schema(tmp_path, monkeypatch):
    transition = _load_transition()

    vault = tmp_path / "vault.sqlite"
    _vault(vault)
    monkeypatch.setattr(transition, "VAULT", vault)

    transition.run(dry_run=False)

    conn = sqlite3.connect(vault)
    row = conn.execute("SELECT turn_id, project FROM tier_warm").fetchone()
    conn.close()
    assert row == ("old-turn", "aaa-memory")


def test_hot_to_warm_dry_run_does_not_insert(tmp_path, monkeypatch):
    transition = _load_transition()

    vault = tmp_path / "vault.sqlite"
    _vault(vault)
    monkeypatch.setattr(transition, "VAULT", vault)

    transition.run(dry_run=True)

    conn = sqlite3.connect(vault)
    count = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='tier_warm'"
    ).fetchone()[0]
    conn.close()
    assert count == 1
