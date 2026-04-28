"""
Score fusion — Reciprocal Rank Fusion (RRF) across tier result lists.

Each result list: [{'turn_id': str, 'score': float, 'source': 'hot'|'warm'|'cold'}, ...]
Returns fused ranking.
"""

from typing import List, Dict
from collections import defaultdict

RANK_CONSTANT = 60  # standard RRF k


def rrf_fusion(ranked_lists: List[List[Dict]], top_k: int = 50) -> List[Dict]:
    """
    Fuse multiple ranked result lists via Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists : list of lists
        Each sublist is ordered by that tier's ranking (best first)
    top_k : int
        Return this many fused results

    Returns
    -------
    list[dict] with fields: turn_id, score, sources (which tiers contributed)
    """
    scores: Dict[str, float] = defaultdict(float)
    sources: Dict[str, set] = defaultdict(set)

    for rank_list in ranked_lists:
        for rank, doc in enumerate(rank_list):
            tid = doc["turn_id"]
            src = doc.get("source", "unknown")
            rrf = 1.0 / (RANK_CONSTANT + rank)
            scores[tid] += rrf
            sources[tid].add(src)

    # Sort by fused score descending
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    results = []
    for tid, score in fused[:top_k]:
        results.append(
            {"turn_id": tid, "score": round(score, 4), "sources": sorted(sources[tid])}
        )

    return results


# ── Cross-encoder rerank placeholder (T058) ────────────────────────────────────


def rerank_top_k(
    query: str, candidates: List[Dict], model: str = "qwen3-reranker-0.6B"
) -> List[Dict]:
    """
    Re-rank top-K candidates with a cross-encoder.

    Placeholder — actual reranker would load local model or call API.
    """
    # For now: no-op — return in original order but with rerank_flag
    for c in candidates:
        c["reranked"] = True
    return candidates


# ── Token budget (T059) ────────────────────────────────────────────────────────


def enforce_token_budget(
    results: List[Dict], token_estimates: Dict[str, int], budget: int = 2000
) -> List[Dict]:
    """
    Greedy selection until token budget exhausted.

    token_estimates: turn_id → token count
    """
    selected = []
    used = 0
    for r in results:
        tid = r["turn_id"]
        tokens = token_estimates.get(tid, 300)  # default avg
        if used + tokens > budget:
            break
        selected.append(r)
        used += tokens
    return selected


# ── Echo-loop prevention (T060) ────────────────────────────────────────────────

SENTINEL_MARKERS = [
    "[MEMORY-INTERNAL]",
    "[FROM-MEMORY]",
    "[IGNORE]",
]


def strip_echo_cycles(text: str) -> str:
    """Remove content between sentinel markers to prevent infinite self-reference."""
    for marker in SENTINEL_MARKERS:
        # Remove everything from marker to end-of-marker pattern
        pattern = re.compile(re.escape(marker) + r".*?" + re.escape(marker), re.DOTALL)
        text = pattern.sub("", text)
    return text


def detect_echo(
    source_turn_id: str, candidate_turn_id: str, source_text: str, candidate_text: str
) -> bool:
    """
    Detect if candidate result is an echo of the source query.
    Simple n-gram overlap heuristic.
    """
    # If >90% token overlap, likely echo
    src_tokens = set(source_text.lower().split())
    cand_tokens = set(candidate_text.lower().split())
    if not src_tokens or not cand_tokens:
        return False
    overlap = len(src_tokens & cand_tokens) / len(src_tokens)
    return overlap > 0.9 and source_turn_id != candidate_turn_id


if __name__ == "__main__":
    # Quick RRF sanity check
    list1 = [{"turn_id": "a", "score": 0.9}, {"turn_id": "b", "score": 0.8}]
    list2 = [{"turn_id": "b", "score": 0.95}, {"turn_id": "c", "score": 0.7}]
    fused = rrf_fusion([list1, list2])
    print("Fused:", fused)
