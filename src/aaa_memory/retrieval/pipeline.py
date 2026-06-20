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
4. RRF score fusion
5. Token budget enforcement
"""

import json
import sqlite3
import urllib.request
import urllib.error
import re
from typing import List, Dict, Optional
from pathlib import Path

VAULT = Path.home() / ".cache" / "aaa-memory" / "vault.sqlite"
CLAWMEM_URL = "http://localhost:7438"


def _clawmem_available() -> bool:
    try:
        req = urllib.request.Request(f"{CLAWMEM_URL}/health")
        urllib.request.urlopen(req, timeout=2)
        return True
    except Exception:
        return False


def _clawmem_search(query: str, limit: int = 5) -> List[Dict]:
    """Search ClawMem via REST API."""
    try:
        req = urllib.request.Request(
            f"{CLAWMEM_URL}/documents?pattern={query}&limit={limit}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            results = data.get("documents", [])
            return [{
                "turn_id": r.get("docid", ""),
                "agent": "clawmem",
                "raw_text": r.get("body", "")[:300],
                "title": r.get("title", ""),
                "score": 0.5,
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


def classify_intent(query: str) -> str:
    """Simple intent classification (rule-based).

    Returns one of: recent, factual, relational, archival, ambiguous
    """
    q = query.lower()

    # Recent: time-sensitive queries
    if any(w in q for w in ["today", "yesterday", "just", "recent", "last", "latest", "now", "current"]):
        return "recent"

    # Archival: historical queries
    if any(w in q for w in ["old", "archive", "history", "previously", "before", "2024", "2025"]):
        return "archival"

    # Relational: asking about connections
    if any(w in q for w in ["relate", "connect", "link", "between", "compare", "difference"]):
        return "relational"

    # Factual: specific information
    if any(w in q for w in ["what is", "how to", "why", "when", "where", "who", "which"]):
        return "factual"

    return "ambiguous"


def rrf_fusion(result_lists: List[List[Dict]], top_k: int = 10, k: int = 60) -> List[Dict]:
    """Reciprocal Rank Fusion across multiple result lists.

    RRF score = Σ(1 / (k + rank_i)) for each list where the item appears.
    k is a smoothing constant (default 60).
    """
    scores = {}
    sources = {}

    for results in result_lists:
        for rank, r in enumerate(results):
            tid = r.get("turn_id", "")
            if not tid:
                continue
            rrf_score = 1.0 / (k + rank + 1)
            scores[tid] = scores.get(tid, 0.0) + rrf_score
            if tid not in sources:
                sources[tid] = {
                    "turn_id": tid,
                    "agent": r.get("agent", "?"),
                    "raw_text": r.get("raw_text", ""),
                    "title": r.get("title", ""),
                    "sources": [],
                }
            sources[tid]["sources"].append(r.get("source", "?"))

    # Sort by RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # Build final results
    final = []
    for tid in sorted_ids[:top_k]:
        item = sources[tid]
        item["score"] = round(scores[tid], 4)
        final.append(item)

    return final


def enforce_token_budget(results: List[Dict], token_estimates: Dict[str, int], budget: int = 2000) -> List[Dict]:
    """Enforce token budget on results.

    Greedily includes results until budget is exhausted.
    """
    included = []
    total_tokens = 0

    for r in results:
        tid = r.get("turn_id", "")
        tokens = token_estimates.get(tid, len(r.get("raw_text", "")) // 4)
        if total_tokens + tokens <= budget:
            included.append(r)
            total_tokens += tokens
        else:
            break

    return included


def search(query: str, limit: int = 5, intent: Optional[str] = None, **kwargs) -> List[Dict]:
    """
    Search across all memory tiers.

    Returns list of dicts with: turn_id, agent, raw_text, score, source
    """
    if intent is None:
        intent = classify_intent(query)

    all_results = []

    # Hot tier (always available)
    hot = _hot_search(query, limit * 2)
    if hot:
        all_results.append(hot)

    # Wiki pages
    wiki = _wiki_search(query, limit)
    if wiki:
        all_results.append(wiki)

    # Warm tier (ClawMem if available)
    if _clawmem_available():
        warm = _clawmem_search(query, limit)
        if warm:
            all_results.append(warm)

    if not all_results:
        return []

    # Fuse results via RRF
    fused = rrf_fusion(all_results, top_k=limit * 2)

    # Token budget enforcement
    token_estimates = {r["turn_id"]: len(r.get("raw_text", "")) // 4 for r in fused}
    final = enforce_token_budget(fused, token_estimates, budget=2000)

    return final[:limit]


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "memory"
    results = search(q)
    print(json.dumps(results, indent=2, default=str))
