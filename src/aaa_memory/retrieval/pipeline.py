"""
Unified retrieval pipeline — intent-aware, tier-routed search.

Tiers:
  Hot  → aaa-memory vault (SQLite FTS5, fast writes)
  Warm → ClawMem (indexed docs, FTS + vector search)
  Cold → ClawMem archive / local fallback

Orchestrates:
1. Intent classification (rule-based)
2. Tier selection
3. Search across tiers
4. Score fusion
5. Result formatting
"""

import json
import sqlite3
import urllib.request
import urllib.error
from typing import List, Dict, Optional
from pathlib import Path

VAULT = Path.home() / ".cache" / "aaa-memory" / "vault.sqlite"
CLAWMEM_URL = "http://localhost:7438"


def _clawmem_available() -> bool:
    try:
        req = urllib.request.Request(f"{CLAWMEM_URL}/health", timeout=2)
        urllib.request.urlopen(req, timeout=2)
        return True
    except Exception:
        return False


def _clawmem_search(query: str, limit: int = 5) -> List[Dict]:
    """Search ClawMem via REST API."""
    try:
        body = json.dumps({"query": query, "limit": limit, "mode": "fts"}).encode()
        req = urllib.request.Request(
            f"{CLAWMEM_URL}/search", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            results = data.get("results", [])
            # Normalize to pipeline format
            return [{
                "turn_id": r.get("docid", ""),
                "agent": "clawmem",
                "raw_text": r.get("snippet", r.get("title", "")),
                "content": r.get("snippet", ""),
                "title": r.get("title", ""),
                "score": r.get("score", 0),
                "source": "warm",
            } for r in results]
    except Exception:
        return []


def _hot_search(query: str, limit: int = 5) -> List[Dict]:
    """Search aaa-memory vault (turns + hot_memories)."""
    if not VAULT.exists():
        return []

    results = []
    try:
        conn = sqlite3.connect(str(VAULT))
        conn.row_factory = sqlite3.Row

        # Search turns via FTS5
        try:
            rows = conn.execute("""
                SELECT turn_id, agent, raw_text, created_at
                FROM turns
                WHERE turns MATCH ?
                ORDER BY rank LIMIT ?
            """, (query, limit)).fetchall()
            for r in rows:
                results.append({
                    "turn_id": r["turn_id"],
                    "agent": r["agent"],
                    "raw_text": r["raw_text"],
                    "score": 0.5,
                    "source": "hot",
                })
        except sqlite3.OperationalError:
            pass

        # Search hot memories (keyword match)
        rows = conn.execute("SELECT * FROM hot_memories").fetchall()
        q_lower = query.lower()
        for r in rows:
            content = r["content"].lower()
            if any(w in content for w in q_lower.split() if len(w) > 2):
                results.append({
                    "turn_id": r["id"],
                    "agent": r["source"],
                    "raw_text": r["content"],
                    "score": 0.6 if r["pinned"] else 0.4,
                    "source": "hot",
                })

        conn.close()
    except Exception:
        pass

    return results[:limit]


def _wiki_search(query: str, limit: int = 5) -> List[Dict]:
    """Search wiki pages in vault."""
    if not VAULT.exists():
        return []

    try:
        conn = sqlite3.connect(str(VAULT))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT title, content, category, path
            FROM wiki_pages
            WHERE wiki_pages MATCH ?
            ORDER BY rank LIMIT ?
        """, (query, limit)).fetchall()
        conn.close()
        return [{
            "turn_id": r["path"],
            "agent": "wiki",
            "raw_text": r["content"][:200],
            "title": r["title"],
            "score": 0.7,
            "source": "wiki",
        } for r in rows]
    except Exception:
        return []


def search(query: str, limit: int = 5, **kwargs) -> List[Dict]:
    """
    Search across all memory tiers.

    Returns list of dicts with: turn_id, agent, raw_text, score, source
    """
    all_results = []

    # Hot tier (always available)
    all_results.extend(_hot_search(query, limit))

    # Wiki pages
    all_results.extend(_wiki_search(query, limit))

    # Warm tier (ClawMem if available)
    if _clawmem_available():
        all_results.extend(_clawmem_search(query, limit))

    # Deduplicate by turn_id
    seen = set()
    unique = []
    for r in all_results:
        tid = r.get("turn_id", "")
        if tid not in seen:
            seen.add(tid)
            unique.append(r)

    # Sort by score
    unique.sort(key=lambda x: x.get("score", 0), reverse=True)

    return unique[:limit]


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "memory"
    results = search(q)
    print(json.dumps(results, indent=2, default=str))
