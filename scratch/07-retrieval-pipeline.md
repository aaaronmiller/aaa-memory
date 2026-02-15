# Topic: Retrieval Pipeline

## Summary
Two-stage retrieval system with broad recall, hybrid search, precision reranking, and cross-modal retrieval capabilities.

---

## Pipeline Overview

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md), [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md)

```
Query → Stage 1: Recall (top 100) → Hybrid Search → Stage 2: Rerank (top 10) → Results
```

---

## Stage 1: Broad Recall

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md)

```yaml
recall:
  model: "primary"              # Qwen3-VL-Embedding-8B
  top_k: 100
  similarity_threshold: 0.5
  multimodal_query_support: true
```

- Embed the query using Qwen3-VL-Embedding-8B
- Retrieve top 100 candidates by vector similarity
- Minimum similarity threshold: 0.5
- Supports multimodal queries (text, image, mixed)

---

## Hybrid Search

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md)

```yaml
hybrid:
  enabled: true
  vector_weight: 0.7
  keyword_weight: 0.3
  keyword_method: "bm25"
```

- Combines vector similarity (70%) with BM25 keyword matching (30%)
- Improves recall for exact-match queries that pure vector search might miss

---

## Stage 2: Precision Reranking

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md), [opus-prd1-v3.md](../opus-prd1-v3.md)

```yaml
reranking:
  enabled: true
  model: "reranker"             # Qwen3-VL-Reranker-8B
  top_k_input: 100
  top_k_output: 10
  multimodal_rerank: true
```

- Takes 100 candidates from recall stage
- Uses Qwen3-VL-Reranker-8B (Single-Tower, Cross-Attention)
- Outputs top 10 most relevant results
- Supports multimodal reranking (query and documents can be mixed-modal)
- Relevance score via yes/no token generation probability

---

## Cross-Modal Retrieval

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md), [opus-prd1-v3.md](../opus-prd1-v3.md)

```yaml
cross_modal:
  enabled: true
  query_modalities: ["text", "image", "mixed"]
  result_modalities: ["text", "image", "code", "mixed"]
```

Enables queries like:
- Text query → retrieve images/code/video
- Image query → retrieve related text/code
- Mixed query (text + image) → retrieve any modality

> **Source:** [opus-prd3-v3.md](../opus-prd3-v3.md)

Example cross-modal queries:
- "Find the diagram referenced by the code comment describing the vectorized kernel"
- "Find video clips illustrating the algorithm described in Section 3.2"
- "Find the code implementing the architecture in this screenshot"

---

## Performance Targets

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md)

| Metric | Target |
|--------|--------|
| Max latency | 550ms (text), 600ms (image), 700ms (mixed) |
| Min relevance score | 0.6 |

---

## MemVid "Zoom" Pattern Retrieval

> **Source:** [gemini-prd.md](../gemini-prd.md)

For MemVid (Cold Memory) retrieval, use cascading lookup across quad-encoded layers:

1. **Scout (Paragraph Layer):** Find general concepts — broad context
2. **Snipe (Sentence Layer):** Check specific facts in the region
3. **Stitch (Boundary Layer):** Retrieve boundary vectors to see cross-section connections

---

## Query Classification & Routing

> **Source:** [chatgpt5.2-prd.md](../chatgpt5.2-prd.md)

The retrieval system needs to:
1. Classify the user's query intent
2. Determine which domain(s) to search (prompts, codebase, research)
3. Select appropriate retrieval strategy based on query type
4. Route to the correct MemVid file(s) and chunking method indices

### Model for Query Classification
> **Source:** [chatgpt5.2-prd.md](../chatgpt5.2-prd.md)

A model is needed to categorize user requests and determine which RAG ingestion methodology to use for retrieval. The user suggests this could potentially be done recursively/iteratively.

---

## Context Injection (Proxy Integration)

> **Source:** [gemini-prd.md](../gemini-prd.md)

The Proxy enriches queries by:
1. Running intent classification
2. Querying Graphiti (Warm) + ByteRover (Hot)
3. Prepending relevant context as a "System Note"

---

## Implementation Requirements

1. Implement vector similarity search (FAISS/HNSW)
2. Implement BM25 keyword search
3. Build hybrid search score combiner (0.7 vector + 0.3 keyword)
4. Integrate Qwen3-VL-Reranker-8B for precision reranking
5. Build cross-modal query support
6. Implement query classification/routing logic
7. Build the "Zoom" pattern for MemVid retrieval
8. Implement latency monitoring and optimization
9. Build retrieval analytics tracking (retrieval_events table)

---

## Conflicts / Ambiguities

- **⚠️ Query routing model:** chatgpt5.2-prd.md asks what model class is needed for query classification but doesn't specify one. No other document provides a concrete answer. This needs to be determined during implementation.
- **⚠️ Latency targets vary:** 550ms for text queries, 600ms for image, 700ms for mixed — these are from verification_queries in opus-prd2-v3.md. The general target is 550ms. Mixed-modal queries may need relaxed targets.
- **⚠️ Sub-second vs 550ms:** chatgpt5.2-prd.md mentions "sub-second latency" as a goal; opus-prd2-v3.md specifies 550ms. These are compatible but 550ms is the stricter target.
