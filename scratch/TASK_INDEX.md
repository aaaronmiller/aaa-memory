# Task Index — RAG v3.0 Architecture Implementation

## Overview

This index maps the project's documentation into 14 focused implementation topics. Each scratch file consolidates all requirements, specifications, and details for a single topic from across all source documents.

### Source Documents
| Document | Description |
|----------|-------------|
| `chatgpt5.2-prd.md` | Original RAG requirements, cost analysis, agentic deployment |
| `gemini-prd.md` | Autodidactic Omni-Loop: memory hierarchy, sleep-time compute, strategic integrations |
| `opus-prd1-v3.md` | RAG v3.0 spec: Qwen3-VL models, metadata schema, TypeScript interfaces |
| `opus-prd2-v3.md` | YAML configuration: embedding, chunking, orchestration, retrieval, domains |
| `opus-prd3-v3.md` | Foundational theory: multi-pass chunking, embedding model selection |
| `docs/UNIFIED_PRD.md` | Consolidated PRD with executive summary and implementation roadmap |
| `docs/SCHEMA_REFERENCE.md` | Complete SQLite schema, TypeScript interfaces, configuration schema |
| `docs/AGGREGATION_PLAN.md` | Overlap/conflict analysis across all source documents |

---

## Topic Index

| # | Topic | Scratch File | Summary | Complexity | Dependencies |
|---|-------|-------------|---------|------------|--------------|
| 01 | Embedding Model Stack | [`01-embedding-model-stack.md`](01-embedding-model-stack.md) | Configure Qwen3-VL-Embedding-8B, Reranker, boundary detection model, and text-only fallback | Medium | None |
| 02 | Chunking Strategies | [`02-chunking-strategies.md`](02-chunking-strategies.md) | Implement 7 chunking methods (fixed-size, sentence, semantic, recursive, AST, multimodal boundary, screenshot-code fusion) | Large | 01 (semantic chunking needs boundary detection model) |
| 03 | Metadata Schema | [`03-metadata-schema.md`](03-metadata-schema.md) | Define 12-dimensional chunk metadata (Pydantic models + TypeScript interfaces) | Medium | None |
| 04 | Database Schema | [`04-database-schema.md`](04-database-schema.md) | SQLite tables for chunks, embeddings, relationships, entities, retrieval events, MemVid indices | Medium | 03 |
| 05 | MemVid Storage | [`05-memvid-storage.md`](05-memvid-storage.md) | H.265 video-encoded storage with QR frames, quad-encoding, and FAISS indices | Large | 01, 04 |
| 06 | Memory Hierarchy | [`06-memory-hierarchy.md`](06-memory-hierarchy.md) | Three-tiered Hot/Warm/Cold memory (ByteRover, Graphiti, MemVid) with transition protocols | Large | 04, 05 |
| 07 | Retrieval Pipeline | [`07-retrieval-pipeline.md`](07-retrieval-pipeline.md) | Two-stage retrieval: broad recall + reranking, hybrid search, cross-modal queries | Medium | 01, 04, 05 |
| 08 | Orchestration & Concurrency | [`08-orchestration-concurrency.md`](08-orchestration-concurrency.md) | Ingestion pipeline with concurrent workers, batching, MCP server integrations | Large | 02, 03, 14 |
| 09 | Proxy Shim | [`09-proxy-shim.md`](09-proxy-shim.md) | Claude Code wrapper: intercept, enrich, execute, capture, sanitize, ingest | Medium | 06, 07 |
| 10 | Sleep-Time Compute | [`10-sleep-time-compute.md`](10-sleep-time-compute.md) | Autonomous daemon with 3 refinement loops, Tribunal, Mutator, Taste Oracle | Large | 06, 09 |
| 11 | Quality Assurance | [`11-quality-assurance.md`](11-quality-assurance.md) | Chunk validation, coherence scoring, outlier detection, error handling | Small | 03, 04 |
| 12 | Domain Configuration | [`12-domain-configuration.md`](12-domain-configuration.md) | Three content domains (prompts, codebase, research) with per-domain settings | Small | 02, 05 |
| 13 | Strategic Integrations | [`13-strategic-integrations.md`](13-strategic-integrations.md) | Advanced features: hypergraph, curiosity module, formal verification, model merging, value alignment | Large | 06, 10 |
| 14 | Content Detection & Routing | [`14-content-detection-routing.md`](14-content-detection-routing.md) | File classification by content type, modality, and domain; routing to chunking pipelines | Small | 12 |

---

## Suggested Implementation Order

### Phase 1: Foundation (Week 1-2)
> No external dependencies. Sets up the core data structures and models.

1. **03 — Metadata Schema** (Medium) — Define all data types first
2. **04 — Database Schema** (Medium) — Create SQLite tables matching the metadata schema
3. **01 — Embedding Model Stack** (Medium) — Set up model wrappers and initialization
4. **12 — Domain Configuration** (Small) — Define the three content domains

### Phase 2: Ingestion Pipeline (Week 3-4)
> Depends on Phase 1. Builds the content processing pipeline.

5. **14 — Content Detection & Routing** (Small) — File classification and routing logic
6. **02 — Chunking Strategies** (Large) — All 7 chunking methods
7. **11 — Quality Assurance** (Small) — Validation and error handling
8. **08 — Orchestration & Concurrency** (Large) — Wire everything together with workers

### Phase 3: Storage & Retrieval (Week 5-6)
> Depends on Phase 2. Implements the storage backends and retrieval pipeline.

9. **05 — MemVid Storage** (Large) — Video-encoded storage with quad-encoding
10. **07 — Retrieval Pipeline** (Medium) — Two-stage retrieval with reranking

### Phase 4: Memory & Autonomy (Week 7-8)
> Depends on Phase 3. Builds the autonomous memory system.

11. **06 — Memory Hierarchy** (Large) — Three-tiered memory with transitions
12. **09 — Proxy Shim** (Medium) — Claude Code wrapper with context injection
13. **10 — Sleep-Time Compute** (Large) — Autonomous refinement loops

### Phase 5: Advanced Features (Week 9+)
> Optional. Enhances the system with cutting-edge capabilities.

14. **13 — Strategic Integrations** (Large) — Hypergraph, curiosity, verification, merging, alignment

---

## Cross-Document Conflicts & Ambiguities

### Major Conflicts

| Topic | Conflict | Documents | Resolution |
|-------|----------|-----------|------------|
| Chunking sizes | Fixed-size: 512 tokens (opus-prd2) vs 1.5-3K tokens (AGGREGATION_PLAN) | `opus-prd2-v3.md`, `docs/AGGREGATION_PLAN.md` | Use YAML config (512 tokens) as authoritative |
| Chunking method count | 6 methods (roadmap) vs 7 methods (spec) | `docs/UNIFIED_PRD.md` | 7 methods — sentence-based may be folded into semantic |
| Memory retention | "7-90h" (hours) vs "7-90d" (days) for Graphiti | `gemini-prd.md`, `docs/UNIFIED_PRD.md` | Use days; 30-day staleness threshold is the actionable number |
| Vector store | ChromaDB/FAISS (UNIFIED_PRD) vs FAISS only (gemini-prd) | `docs/UNIFIED_PRD.md`, `gemini-prd.md` | Start with FAISS; ChromaDB as optional alternative |

### Ambiguities Requiring Decisions

| Topic | Question | Suggested Default |
|-------|----------|-------------------|
| 02 | Can semantic + recursive hierarchical share a single agent pass? | Separate passes for quality |
| 05 | QR code frames vs text-rendered frames for MemVid? | QR for data integrity, text for debugging |
| 07 | Which model for query classification? | Use embedding model similarity, not a separate classifier |
| 08 | Which LLM class per specialist worker? | Sonnet-class for semantic/hierarchical, programmatic for fixed/sentence |
| 09 | Which local LLM for output sanitization? | Small model (Phi-3 or similar) via Ollama |
| 11 | Algorithm for coherence scoring? | Embedding similarity between chunk halves |
| 14 | How to classify ambiguous .md files? | Directory path heuristics + content structure analysis |

---

## Complexity Summary

| Complexity | Count | Topics |
|-----------|-------|--------|
| Small | 4 | 11, 12, 14 |
| Medium | 5 | 01, 03, 04, 07, 09 |
| Large | 5 | 02, 05, 06, 08, 10, 13 |
