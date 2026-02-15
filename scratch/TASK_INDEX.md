# Task Index: RAG v3.0 Implementation Topics

## Overview

This index maps 13 topic-focused scratch files decomposed from the RAG v3.0 project documentation. Each file consolidates all requirements for a single topic from across 8 source documents.

### Source Documents
| Document | Focus |
|----------|-------|
| `chatgpt5.2-prd.md` | Original requirements, cost analysis, agentic deployment |
| `gemini-prd.md` | Autodidactic Omni-Loop, memory hierarchy, sleep-time compute |
| `opus-prd1-v3.md` | RAG v3.0 architecture, embedding models, metadata schema |
| `opus-prd2-v3.md` | YAML configuration for all components |
| `opus-prd3-v3.md` | Foundational theory, multimodal revolution, chunking theory |
| `docs/UNIFIED_PRD.md` | Consolidated specification |
| `docs/SCHEMA_REFERENCE.md` | Database schema, TypeScript interfaces, config schema |
| `docs/AGGREGATION_PLAN.md` | Overlap/conflict analysis between documents |

---

## Topic Index

| # | File | Topic | Complexity | Dependencies |
|---|------|-------|-----------|-------------|
| 01 | [01-embedding-model-stack.md](./01-embedding-model-stack.md) | Embedding Model Stack | Large | None |
| 02 | [02-chunking-strategies.md](./02-chunking-strategies.md) | Chunking Strategies | Large | 01 |
| 03 | [03-metadata-schema.md](./03-metadata-schema.md) | Metadata Schema - 12 Dimensions | Large | None |
| 04 | [04-database-schema.md](./04-database-schema.md) | SQLite Database Schema | Medium | 03 |
| 05 | [05-memvid-storage.md](./05-memvid-storage.md) | MemVid Video-Encoded Storage | Large | 01, 04 |
| 06 | [06-memory-hierarchy.md](./06-memory-hierarchy.md) | Three-Tiered Memory Hierarchy | Medium | 05 |
| 07 | [07-retrieval-pipeline.md](./07-retrieval-pipeline.md) | Retrieval Pipeline | Large | 01, 04, 05 |
| 08 | [08-orchestration-concurrency.md](./08-orchestration-concurrency.md) | Orchestration & Concurrency | Medium | 02, 07 |
| 09 | [09-proxy-shim.md](./09-proxy-shim.md) | Proxy / Shim - The Gatekeeper | Medium | 06, 07 |
| 10 | [10-sleep-time-compute.md](./10-sleep-time-compute.md) | Sleep-Time Compute & Self-Improvement | Large | 06, 09 |
| 11 | [11-quality-assurance.md](./11-quality-assurance.md) | Quality Assurance & Error Handling | Small | 03, 04 |
| 12 | [12-domain-configuration.md](./12-domain-configuration.md) | Domain Configuration & Content Routing | Small | 02, 05 |
| 13 | [13-strategic-integrations.md](./13-strategic-integrations.md) | Strategic Integrations - Advanced | Large | 06, 10 |

---

## Suggested Implementation Order

### Stage 1: Foundation (No dependencies)
1. **03 - Metadata Schema** — Define all TypeScript interfaces and Pydantic models for the 12-dimension schema. This is the data contract everything else depends on.
2. **01 - Embedding Model Stack** — Set up Qwen3-VL-Embedding-8B, reranker, and boundary detection model. Core capability needed by all pipelines.
3. **04 - Database Schema** — Create SQLite tables, indexes, and data access layer. Depends on metadata schema being defined.

### Stage 2: Chunking Pipeline
4. **02 - Chunking Strategies** — Implement all 7 chunking methods. Depends on embedding models for semantic chunking.
5. **12 - Domain Configuration** — Configure content routing (file type → domain → chunking methods). Depends on chunking strategies.
6. **11 - Quality Assurance** — Implement chunk validation, coherence scoring, error handling. Can run in parallel with chunking.

### Stage 3: Storage & Retrieval
7. **05 - MemVid Storage** — Implement H.265 video encoding, QR frames, quad-encoding, FAISS indices. Depends on embedding models and database schema.
8. **07 - Retrieval Pipeline** — Two-stage retrieval with hybrid search and reranking. Depends on MemVid and embedding models.
9. **08 - Orchestration** — Wire up the agentic swarm, concurrency, MCP servers. Depends on chunking and retrieval being implemented.

### Stage 4: Autonomous System
10. **06 - Memory Hierarchy** — Implement ByteRover (Hot), Graphiti (Warm), transition daemons. Depends on MemVid for Cold tier.
11. **09 - Proxy / Shim** — Build the Claude wrapper with context injection. Depends on memory hierarchy and retrieval.
12. **10 - Sleep-Time Compute** — Implement autonomous refinement loops. Depends on proxy and memory hierarchy.

### Stage 5: Advanced Features
13. **13 - Strategic Integrations** — Hypergraph, Active Inference, Formal Verification, Model Merging, Contrastive Alignment. Only after core system is stable.

---

## Dependency Graph

```
Stage 1:  [03 Metadata] ──→ [04 Database]
          [01 Embedding] ─┐
                          │
Stage 2:  [02 Chunking] ←─┘──→ [12 Domains]
          [11 Quality] ←── [03] + [04]

Stage 3:  [05 MemVid] ←── [01] + [04]
          [07 Retrieval] ←── [01] + [05]
          [08 Orchestration] ←── [02] + [07]

Stage 4:  [06 Memory Hierarchy] ←── [05]
          [09 Proxy] ←── [06] + [07]
          [10 Sleep-Time] ←── [06] + [09]

Stage 5:  [13 Strategic] ←── [06] + [10]
```

---

## Cross-Document Conflicts Summary

| Conflict | Documents | Resolution |
|----------|-----------|------------|
| Embedding dimensions | chatgpt5.2-prd vs opus-prd2 | Use opus-prd2 values: 4096 native, MRL options [256,512,1024,2048,4096] |
| Chunk sizes | chatgpt5.2-prd vs opus-prd2 vs AGGREGATION_PLAN | Use opus-prd2 YAML config as authoritative |
| Number of chunking methods | Various (4, 6, or 7) | 7 methods is the complete list |
| Retention periods (hours vs days) | gemini-prd | "h" is a typo; use days |
| Agentic vs algorithmic chunking | chatgpt5.2-prd vs opus-prd2 | Semantic chunking is algorithmic (0.6B model), not full LLM agent |
| QR frames vs text frames | gemini-prd Appendix I vs Appendix II | QR approach is production intent; text rendering is simplified example |
| Gemini embedding alternative | chatgpt5.2-prd only | Not addressed elsewhere; treat as optional future consideration |
| Corpus size | chatgpt5.2-prd (35MB) vs gemini-prd (75K pages) | Different corpora or time horizons; design for the larger estimate |

---

## Tech Stack Summary

| Component | Technology |
|-----------|-----------|
| Primary Embedding | Qwen3-VL-Embedding-8B |
| Reranker | Qwen3-VL-Reranker-8B |
| Boundary Detection | Qwen3-Embedding-0.6B |
| Text Fallback | Qwen3-Embedding-8B |
| Metadata Store | SQLite |
| Vector Search | FAISS (HNSW) |
| Graph Database | FalkorDB (Bolt Protocol) |
| Video Encoding | FFmpeg (H.265/HEVC) |
| AST Parsing | tree-sitter |
| Languages | Python (primary), TypeScript (interfaces) |
| Orchestration | Headless Claude Code + MCP |
| Local LLM | Ollama |
| Remote Models | OpenRouter (free tier for sleep-time) |
| Target Hardware | M3 Max MacBook Pro |
