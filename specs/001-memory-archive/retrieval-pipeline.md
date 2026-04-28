# Hybrid Retrieval Pipeline

## Pipeline Architecture

```
User Query (natural language or SQL)
         │
         ▼
┌─────────────────────────────────────────┐
│            QUERY PARSER                 │
│                                         │
│ Detects:                                │
│ • SQL → direct execution (read-only)    │
│ • Semantic → vector search              │
│ • Graph → edge traversal                │
│ • Hybrid → all modes + fusion           │
│                                         │
│ Also extracts:                          │
│ • Metadata filters (project, date, etc) │
│ • Intent signal (recent/relational/     │
│   archival/factual/ambiguous)           │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┴────────────┬──────────────┬──────────────┐
    ▼                         ▼              ▼              ▼
┌─────────┐          ┌──────────────┐ ┌──────────┐  ┌────────────┐
│ LEXICAL  │          │   SEMANTIC   │ │  GRAPH   │  │  METADATA  │
│ (FTS5)   │          │  (sqlite-vec)│ │ (edges)  │  │  (filters) │
│          │          │              │ │          │  │            │
│ • exact  │          │ • meaning    │ │ • entity │  │ • project  │
│   token  │          │   similarity │ │   links  │  │ • date     │
│   match  │          │ • cosine     │ │ • causal │  │ • model    │
│ • BM25   │          │   distance   │ │   chains │  │ • agent    │
│   score  │          │ • RRF score  │ │ • MPFP   │  │ • type     │
│          │          │              │ │   score  │  │            │
└────┬─────┘          └──────┬───────┘ └────┬─────┘  └──────┬─────┘
     │                       │              │               │
     └───────────┬───────────┘              │               │
                 ▼                          │               │
    ┌────────────────────────┐              │               │
    │  RECIPROCAL RANK FUSION│              │               │
    │                        │              │               │
    │ RRF(r) = Σ 1/(k+rank)  │              │               │
    │ k=60 (default)         │              │               │
    └───────────┬────────────┘              │               │
                │                           │               │
                ▼                           │               │
    ┌────────────────────────┐              │               │
    │ METADATA FILTER APPLY  │◄─────────────┘               │
    │                        │                              │
    │ Apply project, date,   │◄─────────────────────────────┘
    │ model, type filters    │
    │ to fused results       │
    └───────────┬────────────┘
                │
                ▼
    ┌────────────────────────┐
    │   RERANKER             │
    │                        │
    │ Cross-encoder model    │
    │ (qwen3-reranker-0.6B   │
    │  on Ryzen, CPU cosine  │
    │  on Surface)           │
    │                        │
    │ Re-scores top-50 →     │
    │ top-10 by precision    │
    └───────────┬────────────┘
                │
                ▼
    ┌────────────────────────┐
    │ TOKEN BUDGET ENFORCER  │
    │                        │
    │ Greedy select until    │
    │ budget exhausted       │
    │ (default 2000 tokens)  │
    │ Truncate at sentence   │
    │ boundary if partial    │
    └───────────┬────────────┘
                │
                ▼
    ┌────────────────────────┐
    │ PROGRESSIVE DISCLOSURE │
    │                        │
    │ Level 1: Collapsed     │
    │   [icon][project][model]│
    │   [date] [first 80ch]  │
    │                        │
    │ Level 2: Summary       │
    │   + full summary_short │
    │   + topic labels       │
    │   + intent category    │
    │   + failure mode       │
    │                        │
    │ Level 3: Full          │
    │   + raw_text            │
    │   + tool calls         │
    │   + files touched      │
    │   + commands run       │
    │   + related turns graph│
    └────────────────────────┘
```

## Query Examples by Mode

### Lexical (BM25)
```sql
-- "What was that exact command I ran for deploying data-kiln"
SELECT turn_id, project_id, model_name, timestamp_utc, summary_short,
       bm25(turns_fts) AS rank_score
FROM turns_fts
WHERE turns_fts MATCH 'deploy data-kiln command'
ORDER BY rank_score DESC
LIMIT 10;
```

### Semantic (Vector)
```sql
-- "Conversations related to agent slash command configuration"
SELECT t.turn_id, t.project_id, t.model_name, t.timestamp_utc,
       t.summary_short,
       vec_distance_cosine(e.vector, :query_embedding) AS semantic_score
FROM embeddings e
JOIN turns t ON e.entity_id = t.turn_id
WHERE e.entity_type = 'turn'
ORDER BY semantic_score ASC
LIMIT 10;
```

### Graph Traversal
```sql
-- "Show me the conversation where I first described the error handling
--  pattern, and everything that referenced it since"
WITH RECURSIVE related AS (
    -- Start from the anchor turn
    SELECT edge_id, source_node, target_node, edge_type, weight, 0 AS depth
    FROM knowledge_edges
    WHERE source_node = 'turn:abc-123'  -- The anchor turn
    UNION ALL
    -- Follow edges outward
    SELECT ke.edge_id, ke.source_node, ke.target_node, ke.edge_type, ke.weight, r.depth + 1
    FROM knowledge_edges ke
    JOIN related r ON ke.source_node = r.target_node OR ke.target_node = r.source_node
    WHERE r.depth < 3  -- Max 3 hops
)
SELECT t.turn_id, t.project_id, t.summary_short, r.edge_type, r.depth, r.weight
FROM related r
JOIN turns t ON t.turn_id = REPLACE(r.target_node, 'turn:', '')
ORDER BY r.depth, r.weight DESC;
```

### Metadata Filter
```sql
-- "Everything I've ever done in Python"
SELECT turn_id, project_id, model_name, timestamp_utc, summary_short
FROM turns
WHERE json_each.value LIKE '%python%'  -- topic_labels JSON contains 'python'
   OR commands_run LIKE '%python%'
   OR files_touched LIKE '%.py%'
ORDER BY timestamp_utc DESC;
```

### Hybrid (All Modes)
```python
async def hybrid_search(query: str, filters: dict, k: int = 10) -> list[Result]:
    """Execute all four retrieval modes, fuse, filter, rerank."""

    # 1. Lexical
    lexical = await fts5_search(query, limit=k * 3)

    # 2. Semantic
    query_embedding = await embedder.encode(query)
    semantic = await vector_search(query_embedding, limit=k * 3)

    # 3. Graph
    graph = await graph_search(query, max_hops=3, limit=k)

    # 4. Metadata filter results from all three
    filtered = apply_metadata_filters(lexical + semantic + graph, filters)

    # 5. Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion(filtered, k=60)

    # 6. Rerank with cross-encoder
    reranked = await cross_encoder_rerank(fused[:50], query, top_k=k)

    # 7. Token budget enforcement
    results = enforce_token_budget(reranked, budget=2000)

    return results
```

## Intent-to-Tier Mapping

| Intent Signal | Primary Tier(s) | Query Mode | Rationale |
|---------------|-----------------|------------|-----------|
| "just now", "this session", "the error I got" | Hot (ClawMem) | Lexical + Metadata | Active session context, recent timestamps |
| "why did we", "how does X relate to Y" | Warm (Graphiti) | Graph + Semantic | Relationship queries need edge traversal |
| "months ago", "that old project" | Cold (MemVid) | Lexical + Metadata | Historical lookup, timestamp-based |
| "what is the API for", "show me the config" | Hot + Warm | Lexical + Semantic | Factual lookup, exact match + semantic |
| Broad/ambiguous | All tiers | Hybrid (all modes) | Maximize recall when intent unclear |

## Why Hybrid, Not Just Semantic

Your use-cases demand all four retrieval modes because they ask fundamentally different questions:

- **"Everything I've ever done in Python"** → This is a metadata filter on `topic_labels` + lexical on file extensions. Semantic search is useless here — you want exact matches, not similar meanings.

- **"What was that exact command I ran for deploying data-kiln"** → Lexical search on `commands_run`. The command is a specific string. Embedding "deploy data-kiln" won't find "rsync -avz /build user@server:/deploy".

- **"Conversations related to agent slash command configuration"** → Semantic search on `normalized_text`. The user doesn't remember exact words. They remember the concept. Embeddings bridge the vocabulary gap.

- **"Show me the conversation where I first described the error handling pattern, and everything that referenced it since"** → Graph traversal following `derived_from` edges from the anchor turn. Neither lexical nor semantic can follow provenance chains.

## Anticipatory Prefetch

During async hooks (PostToolUse, Stop), a background process:

1. Analyzes the last 6 conversation turns
2. Predicts 3-5 likely follow-up topics using:
   - Entity co-occurrence from the current session
   - Historical query patterns (retrieval_queries table)
   - Graph proximity to recently accessed nodes
3. Pre-queries relevant memory tiers
4. Stores results in session-local cache (`~/.aaa-memory/state/prefetch_cache/`)
5. On next UserPromptSubmit: check prefetch cache first (hits bypass tier queries entirely)

**Target**: 75% cache hit rate on topically coherent sessions (derived from VoiceAgentRAG pattern, Salesforce March 2026).
