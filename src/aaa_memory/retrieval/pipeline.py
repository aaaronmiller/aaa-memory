"""
Unified retrieval pipeline — intent-aware, tier-routed search.

Orchestrates:
1. Intent classification (Nemotron 3 Super or rule fallback)
2. Tier selection (hot / warm / cold / all)
3. Parallel search across tiers
4. Score fusion (RRF)
5. Token budget enforcement
6. Echo prevention
"""

import os
from typing import List, Dict, Optional
from aaa_memory.router import classify_intent, Intent
from aaa_memory.retrieval.hot import search as hot_search
from aaa_memory.retrieval.warm import search_relationship as warm_search
from aaa_memory.retrieval.cold import search_archive as cold_search
from aaa_memory.retrieval.fusion import rrf_fusion, enforce_token_budget, detect_echo

from aaa_memory.config import TOP_K

def search(query: str, limit: int = TOP_K, intent: Optional[Intent] = None, source_turn_id: Optional[str] = None) -> List[Dict]:
    """
    End-to-end retrieval pipeline.

    Parameters
    ----------
    query : str
        Natural language query
    limit : int
        Max results to return after all processing
    intent : Intent | None
        Pre-classified intent (skip classification if provided)
    source_turn_id : str | None
        If known, used for echo-loop detection

    Returns
    -------
    list[dict] with keys: turn_id, agent, raw_text, score, sources[], metadata
    """
    # 1. Classify intent if not provided
    if intent is None:
        intent = classify_intent(query)

    # 2. Route to tiers
    tier_results = []

    if intent in (Intent.RECENT, Intent.FACTUAL, Intent.AMBIGUOUS):
        # Hot tier: FTS5 + vector (vector TBD)
        hot = hot_search(query, limit=limit*2)
        # Add source tag
        for r in hot:
            r['source'] = 'hot'
        tier_results.append(hot)

    if intent in (Intent.RELATIONSHIP, Intent.AMBIGUOUS):
        # Warm tier: Graphiti (placeholder)
        warm = warm_search(query, limit=limit*2)
        for r in warm:
            r['source'] = 'warm'
        tier_results.append(warm)

    if intent in (Intent.ARCHIVAL, Intent.AMBIGUOUS):
        # Cold tier: MemVid archive
        cold = cold_search(query, limit=limit*2)
        for r in cold:
            r['source'] = 'cold'
        tier_results.append(cold)

    # 3. Fuse results via RRF
    if len(tier_results) == 0:
        return []
    elif len(tier_results) == 1:
        fused = tier_results[0]
    else:
        fused = rrf_fusion(tier_results, top_k=limit*2)

    # 4. Echo-loop detection
    if source_turn_id:
        filtered = []
        for r in fused:
            # Need to retrieve full turn to check — placeholder; skip for now
            filtered.append(r)
        fused = filtered

    # 5. Token budget — estimate tokens (rough: chars/4)
    token_estimates = {r['turn_id']: len(r.get('raw_text', '')) // 4 for r in fused}
    final = enforce_token_budget(fused, token_estimates, budget=2000)

    return final[:limit]

if __name__ == '__main__':
    import sys, json
    q = sys.argv[1] if len(sys.argv) > 1 else "memory"
    results = search(q)
    print(json.dumps(results, indent=2))
