"""
Hot memory store backed by aaa-memory vault.

Drop-in replacement for wiki-memory's MemoryStore that reads/writes the
hot_memories table in the aaa-memory vault instead of a standalone JSON file.
Keeps the same interface so mem.py and the hooks work unchanged.
"""

import json
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

VAULT = Path.home() / ".cache" / "aaa-memory" / "vault.sqlite"

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "to", "of", "in", "on", "for", "with", "as", "at", "by", "it", "this",
    "that", "i", "you", "we", "they", "do", "does", "did", "how", "what",
    "when", "where", "why", "can", "should", "would", "could", "my", "me",
}

DEFAULT_LIMIT = 8


def _now():
    return datetime.now(timezone.utc).isoformat()


def _new_id():
    return f"mem-{uuid.uuid4().hex[:16]}"


def _tokens(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _normalise_tags(tags):
    if tags is None:
        return []
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except (json.JSONDecodeError, TypeError):
            tags = tags.split(",")
    return [t.strip() for t in tags if str(t).strip()]


class VaultMemoryStore:
    """Hot memory store backed by the aaa-memory vault."""

    def __init__(self, vault_path=None):
        self.vault = Path(vault_path or VAULT)

    def _conn(self):
        conn = sqlite3.connect(str(self.vault))
        conn.row_factory = sqlite3.Row
        return conn

    def _row_to_dict(self, row) -> dict:
        d = dict(row)
        d["tags"] = _normalise_tags(d.get("tags"))
        d["pinned"] = bool(d.get("pinned"))
        return d

    def add(self, content, tags=None, project="default",
            source="unknown", pinned=False):
        content = (content or "").strip()
        if not content:
            return None
        conn = self._conn()
        # Dedup
        existing = conn.execute(
            "SELECT id FROM hot_memories WHERE LOWER(content)=? AND project=?",
            (content.lower(), project)
        ).fetchone()
        if existing:
            conn.close()
            return {"id": existing["id"], "content": content, "tags": _normalise_tags(tags),
                    "project": project, "source": source, "pinned": pinned}

        rid = _new_id()
        now = _now()
        conn.execute("""
            INSERT INTO hot_memories (id, content, tags, project, source, pinned, created, accessed, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (rid, content, json.dumps(_normalise_tags(tags)), project, source,
              1 if pinned else 0, now, now))
        conn.commit()
        conn.close()
        return {"id": rid, "content": content, "tags": _normalise_tags(tags),
                "project": project, "source": source, "pinned": pinned,
                "created": now, "accessed": now, "access_count": 0}

    def all(self, project=None):
        conn = self._conn()
        if project:
            rows = conn.execute(
                "SELECT * FROM hot_memories WHERE project=? ORDER BY created DESC",
                (project,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM hot_memories ORDER BY created DESC").fetchall()
        conn.close()
        return [self._row_to_dict(r) for r in rows]

    def recall(self, query, limit=DEFAULT_LIMIT, project=None):
        records = self.all(project)
        if not records:
            return []
        q_tokens = _tokens(query)
        now = time.time()
        scored = []
        for r in records:
            score = self._score(r, q_tokens, now)
            if score > 0:
                scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [r for _, r in scored[:limit]]
        self._touch(top)
        return top

    def _score(self, rec, q_tokens, now):
        text = rec["content"] + " " + " ".join(rec.get("tags", []))
        r_tokens = _tokens(text)
        if not r_tokens:
            return 0.0
        overlap = q_tokens & r_tokens
        if not overlap and not rec.get("pinned"):
            return 0.0
        relevance = len(overlap) / max(1, len(q_tokens)) if q_tokens else 0.0
        try:
            age_days = (now - datetime.fromisoformat(rec["created"]).timestamp()) / 86400
        except (ValueError, OSError):
            age_days = 30.0
        recency = max(0.0, 1.0 - age_days / 30.0)
        pin_boost = 1.0 if rec.get("pinned") else 0.0
        return 2.0 * relevance + 0.5 * recency + 1.5 * pin_boost

    def recent(self, limit=DEFAULT_LIMIT, project=None):
        records = self.all(project)
        records.sort(key=lambda r: (r.get("pinned", False), r.get("created", "")), reverse=True)
        return records[:limit]

    def forget(self, needle):
        conn = self._conn()
        rows = conn.execute("SELECT id, content FROM hot_memories").fetchall()
        removed = 0
        for r in rows:
            if r["id"] == needle or needle.lower() in r["content"].lower():
                conn.execute("DELETE FROM hot_memories WHERE id=?", (r["id"],))
                removed += 1
        conn.commit()
        conn.close()
        return removed

    def _touch(self, recs):
        if not recs:
            return
        conn = self._conn()
        ids = [r["id"] for r in recs]
        now = _now()
        for rid in ids:
            conn.execute("""
                UPDATE hot_memories SET accessed=?, access_count=access_count+1 WHERE id=?
            """, (now, rid))
        conn.commit()
        conn.close()

    def stats(self):
        conn = self._conn()
        total = conn.execute("SELECT COUNT(*) FROM hot_memories").fetchone()[0]
        pinned = conn.execute("SELECT COUNT(*) FROM hot_memories WHERE pinned=1").fetchone()[0]
        rows = conn.execute("SELECT project, COUNT(*) as c FROM hot_memories GROUP BY project").fetchall()
        conn.close()
        return {
            "total": total,
            "pinned": pinned,
            "projects": {r["project"]: r["c"] for r in rows},
            "vault": str(self.vault),
        }
