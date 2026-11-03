"""Append-only, idempotent evidence journal for agent session capture.

This module deliberately has no LLM, network, or optional-package dependency.
The journal stores redacted source text; original private files remain under
their own source system's retention and access policy.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(authorization\s*:\s*bearer\s+)[^\s\"']+"),
    re.compile(r"(?i)\b((?:--api-key|--auth-token|--client-secret|--password)\s*(?:=|\s+))[^\s\"']+"),
    re.compile(r"(?i)\b((?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[=:]\s*)[^\s,\"'}]+"),
    re.compile(r"\b(?:ghp_|gho_|ghu_|ghs_|ghr_|github_pat_)[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
)
_PRIVATE_BLOCK = re.compile(r"<private>.*?</private>", re.I | re.S)
_CONTEXT_BLOCK = re.compile(r"<!--\s*AAA_MEMORY_CONTEXT_START\s*-->.*?<!--\s*AAA_MEMORY_CONTEXT_END\s*-->", re.I | re.S)
_LIMIT_BYTES = 4 * 1024 * 1024


def redact(text: str) -> str:
    """Remove excluded memory spans and common credential formats before storage."""
    text = _PRIVATE_BLOCK.sub("[PRIVATE CONTENT EXCLUDED]", text)
    text = _CONTEXT_BLOCK.sub("[INJECTED CONTEXT EXCLUDED]", text)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(
            lambda m: (m.group(1) if m.lastindex else "") + "[REDACTED]", text
        )
    return text


def default_path() -> Path:
    return Path(os.environ.get("AAA_EVIDENCE_DB", "~/.cache/aaa-memory/evidence.sqlite")).expanduser()


class EvidenceConflict(ValueError):
    """A source event key was reused with different content."""


class EvidenceJournal:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path).expanduser() if path is not None else default_path()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS evidence_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                external_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                project TEXT NOT NULL,
                kind TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                ingested_at TEXT NOT NULL,
                source_path TEXT,
                raw_sha256 TEXT NOT NULL,
                stored_sha256 TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                UNIQUE(source, external_id)
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS ix_evidence_session ON evidence_events(session_id, id)")
            db.execute("CREATE INDEX IF NOT EXISTS ix_evidence_project ON evidence_events(project, id)")
            db.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts
                USING fts5(content, content='evidence_events', content_rowid='id')""")
            db.executescript("""CREATE TRIGGER IF NOT EXISTS evidence_ai AFTER INSERT ON evidence_events BEGIN
                INSERT INTO evidence_fts(rowid, content) VALUES (new.id, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS evidence_ad AFTER DELETE ON evidence_events BEGIN
                INSERT INTO evidence_fts(evidence_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
            END;""")

    def append(
        self, *,
        source: str,
        external_id: str,
        session_id: str,
        content: str,
        project: str = "unclassified",
        kind: str = "observation",
        observed_at: str | None = None,
        source_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Commit one event atomically; replays return the existing identifier.

        Matching source/external_id with differing payload raises EvidenceConflict.
        No summarization or potentially fallible agent work executes in this path.
        """
        if not all(isinstance(x, str) and x.strip() for x in (source, external_id, session_id)):
            raise ValueError("source, external_id and session_id are required")
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        if len(content.encode("utf-8")) > _LIMIT_BYTES:
            raise ValueError("event exceeds 4 MiB limit")
        metadata = metadata or {}
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be an object")
        # Never trust foreign event metadata to be free of credentials.
        metadata_json = redact(json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str))
        clean = redact(content)
        raw_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        stored_digest = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        when = observed_at or now
        datetime.fromisoformat(when.replace("Z", "+00:00"))
        self.initialize()
        with self._connect() as db:
            row = db.execute(
                "SELECT id, raw_sha256 FROM evidence_events WHERE source=? AND external_id=?",
                (source, external_id),
            ).fetchone()
            if row:
                if row["raw_sha256"] != raw_digest:
                    raise EvidenceConflict(f"source event changed: {source}/{external_id}")
                return {"id": row["id"], "duplicate": True, "stored_sha256": stored_digest}
            try:
                cursor = db.execute(
                    """INSERT INTO evidence_events
                    (source, external_id, session_id, project, kind, observed_at,
                     ingested_at, source_path, raw_sha256, stored_sha256, content, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (source, external_id, session_id, project, kind, when,
                     now, source_path, raw_digest, stored_digest, clean, metadata_json),
                )
            except sqlite3.IntegrityError:
                # Two concurrent writers raced; validate the existing version.
                row = db.execute(
                    "SELECT id, raw_sha256 FROM evidence_events WHERE source=? AND external_id=?",
                    (source, external_id),
                ).fetchone()
                if row and row["raw_sha256"] == raw_digest:
                    return {"id": row["id"], "duplicate": True, "stored_sha256": stored_digest}
                raise EvidenceConflict(f"source event changed: {source}/{external_id}")
            return {"id": cursor.lastrowid, "duplicate": False, "stored_sha256": stored_digest}

    def get(self, evidence_id: int) -> dict[str, Any] | None:
        self.initialize()
        with self._connect() as db:
            row = db.execute("SELECT * FROM evidence_events WHERE id=?", (evidence_id,)).fetchone()
            return dict(row) if row else None

    def timeline(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        self.initialize()
        with self._connect() as db:
            rows = db.execute(
                """SELECT id, source, external_id, kind, observed_at, project, content
                FROM evidence_events WHERE session_id=?
                ORDER BY observed_at, id LIMIT ?""",
                (session_id, max(1, min(limit, 500))),
            ).fetchall()
            return [dict(row) for row in rows]

    def search(self, query: str, limit: int = 10, project: str | None = None) -> list[dict[str, Any]]:
        """FTS5 search returns an index; call get(id) for complete evidence."""
        self.initialize()
        if not query.strip():
            return []
        # Quoted terms avoid FTS operators in arbitrary user input.
        terms = re.findall(r"\w+", query, flags=re.UNICODE)
        if not terms:
            return []
        expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms[:16])
        sql = """SELECT e.id, e.source, e.external_id, e.session_id,
                e.project, e.kind, e.observed_at, substr(e.content, 1, 160) AS excerpt,
                bm25(evidence_fts) AS score
                FROM evidence_fts JOIN evidence_events e ON e.id=evidence_fts.rowid
                WHERE evidence_fts MATCH ?"""
        params: list[Any] = [expression]
        if project is not None:
            sql += " AND e.project=?"
            params.append(project)
        sql += " ORDER BY score, e.id DESC LIMIT ?"
        params.append(max(1, min(limit, 100)))
        with self._connect() as db:
            return [dict(row) for row in db.execute(sql, params).fetchall()]

    def verify(self) -> dict[str, Any]:
        """Check stored-content integrity and FTS row count; raw sources are external."""
        self.initialize()
        corrupt: list[int] = []
        with self._connect() as db:
            rows = db.execute("SELECT id, stored_sha256, content FROM evidence_events").fetchall()
            for row in rows:
                if hashlib.sha256(row["content"].encode("utf-8")).hexdigest() != row["stored_sha256"]:
                    corrupt.append(row["id"])
            indexed = db.execute("SELECT COUNT(*) FROM evidence_fts").fetchone()[0]
            return {"events": len(rows), "indexed": indexed, "corrupt_ids": corrupt,
                    "ok": not corrupt and indexed == len(rows)}
