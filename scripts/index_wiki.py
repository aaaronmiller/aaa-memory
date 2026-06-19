#!/usr/bin/env python3
"""Index wiki files into vault (FTS5 + metadata)."""
import sqlite3, json, os
from pathlib import Path

VAULT = Path(os.getenv("AAA_MEMORY_VAULT", Path.home() / ".cache/aaa-memory/vault.sqlite"))
WIKI_DIR = Path.home() / "knowledge/wiki"

def index_all():
    VAULT.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(VAULT))
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS wiki_pages USING fts5(title, content, category, path, tokenize='porter unicode61')")
    conn.execute("CREATE TABLE IF NOT EXISTS wiki_meta (path TEXT PRIMARY KEY, indexed_at TEXT DEFAULT (datetime('now')), word_count INTEGER)")
    
    count = 0
    for md_file in sorted(WIKI_DIR.rglob("*.md")):
        rel = md_file.relative_to(WIKI_DIR)
        title = md_file.stem
        category = md_file.parent.name
        content = md_file.read_text(encoding="utf-8", errors="replace")
        wc = len(content.split())
        try:
            conn.execute("INSERT OR REPLACE INTO wiki_pages (title, content, category, path) VALUES (?, ?, ?, ?)",
                        (title, content, category, str(rel)))
            conn.execute("INSERT OR REPLACE INTO wiki_meta (path, word_count) VALUES (?, ?)", (str(rel), wc))
            count += 1
        except sqlite3.OperationalError:
            continue
    conn.commit()
    conn.close()
    print(f"Indexed {count} wiki pages")

if __name__ == "__main__":
    index_all()
