import json
import sqlite3
from pathlib import Path


def _make_vault(path: Path) -> None:
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
        CREATE VIRTUAL TABLE turns_fts USING fts5(turn_id, raw_text, tokenize='unicode61');
        CREATE TRIGGER turns_ai AFTER INSERT ON turns BEGIN
            INSERT INTO turns_fts(rowid, turn_id, raw_text)
            VALUES (new.rowid, new.turn_id, new.raw_text);
        END;
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
        );
        CREATE VIRTUAL TABLE wiki_pages USING fts5(title, content, category, path, tokenize='porter unicode61');
    """)
    conn.execute(
        "INSERT INTO turns (turn_id, agent, session_id, turn_index, turn_type, raw_text, created_at, metadata) VALUES (?, ?, ?, ?, ?, ?, datetime('now'), ?)",
        ("t1", "claude", "s1", 1, "user", "The hot warm cold memory progression uses clawmem and memvid.", "{}"),
    )
    conn.execute(
        "INSERT INTO hot_memories (id, content, source, pinned, created, accessed) VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))",
        ("m1", "Pinned fact about Cass prompt history fallback.", "test", 1),
    )
    conn.execute(
        "INSERT INTO wiki_pages (title, content, category, path) VALUES (?, ?, ?, ?)",
        ("Memory tiers", "Warm wiki pages are searched alongside ClawMem.", "memory", "/wiki/memory.md"),
    )
    conn.commit()
    conn.close()


def test_pipeline_searches_real_hot_and_wiki_schema(tmp_path, monkeypatch):
    from aaa_memory.retrieval import pipeline

    vault = tmp_path / "vault.sqlite"
    _make_vault(vault)
    monkeypatch.setattr(pipeline, "VAULT", vault)
    monkeypatch.setattr(pipeline, "_clawmem_available", lambda: False)

    results = pipeline.search("hot warm clawmem memvid", limit=5)

    sources = {source for r in results for source in r["sources"]}
    assert "hot" in sources
    assert "wiki" in sources


def test_pipeline_uses_clawmem_retrieve_endpoint(monkeypatch):
    from aaa_memory.retrieval import pipeline

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "results": [{
                    "docid": "abc123",
                    "title": "Memory note",
                    "path": "notes/memory.md",
                    "snippet": "Relevant warm ClawMem snippet.",
                    "score": 0.42,
                }]
            }).encode()

    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["body"] = json.loads(req.data.decode())
        return Response()

    monkeypatch.setattr(pipeline.urllib.request, "urlopen", fake_urlopen)

    results = pipeline._clawmem_search("memory tiers", limit=3)

    assert captured["url"].endswith("/retrieve")
    assert captured["method"] == "POST"
    assert captured["body"]["query"] == "memory tiers"
    assert results[0]["source"] == "warm"
    assert results[0]["raw_text"] == "Relevant warm ClawMem snippet."
