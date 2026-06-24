#!/usr/bin/env python3
"""
Index wiki markdown files into ClawMem vault.

Reads all .md files from ~/knowledge/wiki/, extracts frontmatter and body,
generates embeddings (using aaa-memory.embedding), and inserts into
vault's documents + content_vectors tables with FTS5 update.
"""

import os
import json
from pathlib import Path
from datetime import datetime, timezone
import sqlite3

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aaa_memory.embedding import get_embedder, embed_to_base64
from aaa_memory.models import (
    Element,
)  # reuse for parsing frontmatter? Actually we'll parse manually
import yaml

VAULT = Path(
    os.getenv("AAA_MEMORY_VAULT", "/home/misscheta/.cache/aaa-memory/vault.sqlite")
)
WIKI_BASE = Path("/home/misscheta/knowledge/wiki")


def connect():
    conn = sqlite3.connect(str(VAULT))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def ensure_schema(conn):
    """Create documents table and content_vectors if not exists (ClawMem schema subset)."""
    cur = conn.cursor()
    # Simplified documents table — matches ClawMem's core fields
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection TEXT NOT NULL,
            path TEXT NOT NULL,
            title TEXT NOT NULL,
            hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            modified_at TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            content_type TEXT NOT NULL DEFAULT 'note',
            confidence REAL DEFAULT 0.5,
            topic_key TEXT,
            obs_quality_score REAL DEFAULT 0.5,
            embed_state TEXT DEFAULT 'pending'
        )
    """)
    # Unique constraint on (collection, path)
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_coll_path ON documents(collection, path)
    """)
    # Content storage (hash → doc)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS content (
            hash TEXT PRIMARY KEY,
            doc TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    # Vectors table (sqlite-vec style — but we'll use generic FLOAT[BLOB] for now)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS content_vectors (
            hash TEXT NOT NULL,
            seq INTEGER NOT NULL DEFAULT 0,
            pos INTEGER NOT NULL DEFAULT 0,
            model TEXT NOT NULL,
            embedded_at TEXT NOT NULL,
            embedding BLOB,
            PRIMARY KEY (hash, seq)
        )
    """)
    # FTS5 for documents
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            filepath, title, body,
            tokenize='unicode61'
        )
    """)
    conn.commit()


def parse_wiki_page(md_path: Path):
    """Parse wiki markdown file into metadata dict and body text."""
    text = md_path.read_text()
    parts = text.split("---", 2)
    frontmatter = {}
    body = ""
    if len(parts) >= 3:
        frontmatter = yaml.safe_load(parts[1]) or {}
        body = parts[2].strip()
    else:
        body = text
    return frontmatter, body


def compute_hash(content: str) -> str:
    """Simple SHA1-like hash (not crypto)."""
    import hashlib

    return hashlib.sha1(content.encode()).hexdigest()


def embed_text(text: str) -> bytes:
    """Generate embedding for text, return raw float32 bytes."""
    embedder = get_embedder("auto")
    emb = embedder.embed(text[:8192])  # limit length
    return emb.vector.tobytes()


def index_wiki(conn):
    cur = conn.cursor()
    indexed = 0
    skipped = 0

    for md in WIKI_BASE.rglob("*.md"):
        # Skip auto-generated index pages
        if md.name == "index.md":
            continue
        try:
            fm, body = parse_wiki_page(md)
            # If frontmatter missing title, use filename stem
            title = fm.get("title", md.stem)
            # Some auto pages have no confidence — assume 0.5
            confidence = fm.get("confidence", 0.5)
            # collection is parent dir name (decisions, code, etc.)
            collection = md.parent.name
            path = str(md.relative_to(WIKI_BASE))
            content_hash = compute_hash(body)
            created_at = fm.get("extraction_ts", datetime.now(timezone.utc).isoformat())
            modified_at = datetime.now(timezone.utc).isoformat()

            # Insert content if not exists
            cur.execute(
                """
                INSERT OR IGNORE INTO content (hash, doc, created_at)
                VALUES (?, ?, ?)
            """,
                (content_hash, body, created_at),
            )
            # Insert document
            cur.execute(
                """
                INSERT OR IGNORE INTO documents
                (collection, path, title, hash, created_at, modified_at, active, content_type, confidence, topic_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    collection,
                    path,
                    title,
                    content_hash,
                    created_at,
                    modified_at,
                    1,
                    "wiki",
                    confidence,
                    fm.get("type"),
                ),
            )
            if cur.rowcount == 0:
                skipped += 1
                continue

            doc_id = cur.lastrowid
            if doc_id is None:
                cur.execute("SELECT id FROM documents WHERE hash = ?", (content_hash,))
                row = cur.fetchone()
                doc_id = row[0] if row else None

            # Store embedding
            try:
                embedding_blob = embed_text(body)
                cur.execute(
                    """
                    INSERT OR REPLACE INTO content_vectors (hash, seq, pos, model, embedded_at, embedding)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        content_hash,
                        0,
                        0,
                        "gemma-fallback",
                        datetime.now(timezone.utc).isoformat(),
                        embedding_blob,
                    ),
                )
            except Exception as e:
                print(f"  ⚠️  Embedding failed for {path}: {e}")
                # Continue without embedding — it's ok

            # Update FTS
            try:
                cur.execute(
                    """
                    INSERT INTO documents_fts(rowid, filepath, title, body)
                    VALUES (?, ?, ?, ?)
                """,
                    (doc_id, path, title, body),
                )
            except sqlite3.OperationalError as e:
                if "no such table" in str(e):
                    # FTS not available — skip
                    pass
                else:
                    raise

            indexed += 1
            print(f"Indexed: {path}")
        except Exception as e:
            print(f"Error indexing {md}: {e}")

    conn.commit()
    print(f"\n✅ Indexed {indexed} wiki pages ({skipped} already existed)")


def main():
    print("=== Wiki → Vault Indexer ===")
    if not VAULT.exists():
        print(f"ERROR: Vault not found at {VAULT}")
        return 1
    conn = connect()
    ensure_schema(conn)
    index_wiki(conn)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
