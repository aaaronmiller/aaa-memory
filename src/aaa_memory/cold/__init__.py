"""
Cold tier — ClawMem integration with FTS5 fallback.

When ClawMem is running, uses its REST API for FTS search.
When ClawMem is unavailable, falls back to local vault FTS5.
"""

import json
import sqlite3
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict

CLAWMEM_URL = "http://localhost:7438"
VAULT = Path.home() / ".cache" / "aaa-memory" / "vault.sqlite"


def clawmem_available() -> bool:
    try:
        req = urllib.request.Request(f"{CLAWMEM_URL}/health", timeout=2.0)
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return False


def search_clawmem(query: str, limit: int = 5) -> List[Dict]:
    """Search via ClawMem REST API (FTS mode)."""
    try:
        body = json.dumps({"query": query, "limit": limit, "mode": "fts"}).encode()
        req = urllib.request.Request(
            f"{CLAWMEM_URL}/search", data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode())
            return data.get("results", [])
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return []


def search_local(query: str, limit: int = 5) -> List[Dict]:
    """Fallback: FTS5 search on vault wiki_pages."""
    if not VAULT.exists():
        return []
    try:
        conn = sqlite3.connect(str(VAULT))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT title, content, category, path
            FROM wiki_pages
            WHERE wiki_pages MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def search(query: str, limit: int = 5) -> List[Dict]:
    """Search cold tier — ClawMem FTS if available, local FTS5 fallback."""
    if clawmem_available():
        results = search_clawmem(query, limit)
        if results:
            return results
    return search_local(query, limit)
