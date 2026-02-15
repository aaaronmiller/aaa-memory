# Topic 07: Retrieval Pipeline

## Summary
Implement the two-stage retrieval pipeline with broad recall, hybrid search (vector + keyword), precision reranking via Qwen3-VL-Reranker-8B, and cross-modal retrieval support.

---

## Overview
> Sources: `opus-prd2-v3.md` (lines 361–395), `docs/UNIFIED_PRD.md` (lines 233–237, Implementation Roadmap Stage 4), `chatgpt5.2-prd.md` (lines 25, 44), `gemini-prd.md` (lines 353–364)

The retrieval pipeline is a two-stage process: broad recall followed by precision reranking, with support for cross-modal queries.

---

## Stage 1: Broad Recall
> Source: `opus-prd2-v3.md` (lines 365–370)

```yaml
recall:
  model: "primary"              # Qwen3-VL-Embedding-8B
  top_k: 100
  similarity_threshold: 0.5
  multimodal_query_support: true
```

- Embed the query using Qwen3-VL-Embedding-8B
- Search across FAISS/HNSW index for top 100 candidates
- Filter by minimum similarity threshold (0.5)
- Support multimodal queries (text, image, or mixed)

## Hybrid Search
> Source: `opus-prd2-v3.md` (lines 372–377)

```yaml
hybrid:
  enabled: true
  vector_weight: 0.7
  keyword_weight: 0.3
  keyword_method: "bm25"
```

- Combine vector similarity (70% weight) with BM25 keyword matching (30% weight)
- BM25 provides exact-match capability that vector search may miss

## Stage 2: Precision Reranking
> Source: `opus-prd2-v3.md` (lines 379–385)

```yaml
reranking:
  enabled: true
  model: "reranker"             # Qwen3-VL-Reranker-8B
  top_k_input: 100
  top_k_output: 10
  multimodal_rerank: true
```

- Take top 100 candidates from recall stage
- Pass each (query, document) pair through Qwen3-VL-Reranker-8B
- Cross-attention mechanism for deep inter-modal interaction
- Output top 10 most relevant results
- Relevance score via yes/no token generation probability

## Cross-Modal Retrieval
> Source: `opus-prd2-v3.md` (lines 387–391)

```yaml
cross_modal:
  enabled: true
  query_modalities: ["text", "image", "mixed"]
  result_modalities: ["text", "image", "code", "mixed"]
```

- Query with text → retrieve images, code, or mixed content
- Query with image → retrieve related text or code
- Query with mixed (text + image) → retrieve any modality

## Performance Targets
> Source: `opus-prd2-v3.md` (lines 393–395)

```yaml
targets:
  max_latency_ms: 550
  min_relevance_score: 0.6
```

## Cascading "Zoom" Pattern (MemVid-specific)
> Source: `gemini-prd.md` (lines 353–364)

For MemVid cold storage retrieval using quad-encoded vectors:
1. **Scout (Paragraph Layer):** Find general concepts (broad context)
2. **Snipe (Sentence Layer):** Check specific facts in the region (precise)
3. **Stitch (Boundary Layer):** Retrieve boundary vectors to follow connections

---

## Query Classification
> Source: `chatgpt5.2-prd.md` (lines 25, 28-29)

Before retrieval, classify the user's query to determine:
- Which domain(s) to search (prompts, codebase, research)
- Which chunking methodology's results to prioritize
- Whether to use multimodal retrieval

---

## Implementation Tasks

1. Create `src/retrieval/pipeline.py` — Main two-stage retrieval orchestrator
2. Create `src/retrieval/recall.py` — Stage 1 broad recall with FAISS/HNSW
3. Create `src/retrieval/hybrid_search.py` — BM25 + vector hybrid scoring
4. Create `src/retrieval/reranker.py` — Stage 2 Qwen3-VL-Reranker integration
5. Create `src/retrieval/cross_modal.py` — Cross-modal query handling
6. Create `src/retrieval/query_classifier.py` — Query intent classification and domain routing
7. Create `src/retrieval/zoom_retriever.py` — Cascading zoom pattern for MemVid
8. Create `src/retrieval/analytics.py` — Retrieval event logging to `retrieval_events` table
9. Implement latency monitoring and optimization (target <550ms)

---

## Conflicts & Ambiguities

1. **Query classification model:** `chatgpt5.2-prd.md` asks "what kind of model class are we gonna need to properly categorize the user's request" but doesn't specify one. The proxy shim (`gemini-prd.md` line 214) mentions "Classification (Intent Detection)" but doesn't name a model. Need to decide: lightweight local model, or use the embedding model itself for classification.

2. **Retrieval across memory tiers:** The pipeline config in `opus-prd2-v3.md` describes retrieval from the MemVid/FAISS store. But the system also has Hot (ByteRover) and Warm (Graphiti) memory. The proxy shim queries Hot+Warm for context injection. Need to clarify whether the retrieval pipeline also searches across all tiers or only Cold.

3. **Latency target:** 550ms for standard queries, but `opus-prd2-v3.md` verification queries allow 600ms for image queries and 700ms for mixed queries. The 550ms target may be text-only.
