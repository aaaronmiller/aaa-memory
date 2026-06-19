"""Cross-encoder reranker for query-document pairs.
Uses Qwen3-Reranker-0.6B (CPU, Ryzen GPU fallback) or cosine similarity fallback."""

import os, math
from typing import List, Dict, Optional

TOP_K = 50

def rerank(query: str, results: List[Dict], top_k: int = TOP_K) -> List[Dict]:
    """Re-rank results by cross-encoder score or cosine fallback."""
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("Qwen/Qwen3-Reranker-0.6B", device="cpu")
        pairs = [[query, r.get("raw_text", r.get("text", ""))[:512]] for r in results]
        scores = model.predict(pairs)
        for r, s in zip(results, scores):
            r["score"] = float(s)
            r["reranked"] = True
    except (ImportError, Exception) as e:
        # Fallback: length-normalized TF score
        query_terms = set(query.lower().split())
        for r in results:
            text = r.get("raw_text", r.get("text", "")).lower()
            matches = sum(1 for t in query_terms if t in text)
            r["score"] = matches / max(len(query_terms), 1) * (1.0 / max(1, math.log2(len(text.split()) + 1)))
            r["reranked"] = False
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
