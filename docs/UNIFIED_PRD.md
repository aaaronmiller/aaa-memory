# Unified Product Requirements Document (PRD)
## Cutting-Edge RAG Architecture v3.0: Multimodal Agentic Swarms with Knowledge Graph Metadata

**Version:** 3.0.0  
**Date:** January 9, 2026  
**Author:** Ice-ninja / Sliither  

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Core Philosophy & Constraints](#core-philosophy--constraints)
3. [System Components & Roles](#system-components--roles)
4. [Architecture Overview](#architecture-overview)
5. [Embedding Model Stack](#embedding-model-stack)
6. [Chunking Strategies](#chunking-strategies)
7. [Metadata Schema](#metadata-schema)
8. [Implementation Roadmap](#implementation-roadmap)
9. [Cost Analysis](#cost-analysis)
10. [Code Reference](#code-reference)
11. [Source Documents](#source-documents)

---

## Executive Summary

This document represents the definitive architecture for Ice-ninja's next-generation RAG system, incorporating the **Qwen3-VL-Embedding-8B** multimodal model released January 7-8, 2026. This architecture represents the absolute state-of-the-art as of January 9, 2026, featuring unified text-image-video embeddings, a sophisticated knowledge graph metadata schema rivaling enterprise systems, and agentic orchestration via headless Claude Code.

### What Makes This Cutting-Edge:

- **Qwen3-VL-Embedding-8B**: MMEB-V2 rank #1 (77.8 score), multimodal unified vector space
- **Qwen3-VL-Reranker-8B**: Cross-attention precision reranking for query-document pairs
- **Knowledge Graph Metadata**: 12-dimension schema with provenance chains, semantic linkage, quality scoring
- **Cross-Modal Retrieval**: Query with text, retrieve images/code/video (or any combination)
- **MemVid Multimodal**: Video-encoded storage supporting mixed-modality chunks
- **Autodidactic Omni-Loop**: Full-Cycle Autonomous Memory & Self-Improvement System

---

## Core Philosophy & Constraints

- **Zero Placeholders:** All data must be actionable, structured, and persistent.
- **Local-First:** Primary storage and compute management occur on the user's M3 Max MacBook Pro.
- **Cost Efficiency:** Utilizes Free/OpenRouter tiers for "Heavy Hitter" models (DeepSeek, Qwen, Mistral) during sleep cycles.
- **Bidirectional Learning:** The system learns from the user (active coding) and teaches itself (sleep simulation).
- **Multimodal Revolution:** The architecture moves beyond text monoculture to embrace the world's actual complexity.

---

## System Components & Roles

| Component | Role | Responsibility |
|----------|------|----------------|
| **The Proxy (Shim)** | **The Gatekeeper** | Wraps `claude` command. Intercepts user prompts and model outputs. Injects context from memory. Sanitizes data via Pydantic schemas before storage. |
| **ByteRover** | **Hot Memory** | Filesystem-based active context (0-24h). Stores live "Working Memory," active Git branches, and "in-flight" ideas. Optimized for speed (grep/find). |
| **Graphiti** | **Warm Memory** | Temporal Knowledge Graph (7-90h). Stores structured relationships (Node: `AuthPattern` -> Edge: `MITIGATES` -> Node: `CSRF`). Handles "Skill" storage and "Lineage". |
| **MemVid** | **Cold Memory** | Deep Archive (90d+). Uses H.265 compressed video (QR frames) to store massive datasets. Quad-Encoded vectors (Word/Sentence/Paragraph/Boundary) for high-fidelity retrieval. |
| **The Tribunal** | **The Critic** | A dynamic graph of adversarial personas (Security Zealot, Pedant, Visionary) that critique generated artifacts during sleep cycles. |
| **The Mutator** | **The Evolution** | Uses Genetic Algorithms to rewrite "Skill Files" (prompts) based on Tribunal feedback. |
| **The Taste Oracle** | **The Quality Gate** | Vector-based novelty detector. Compares outputs against a "Gold Standard" baseline in MemVid to reject derivative or hallucinated work. |

---

## Architecture Overview

The Autodidactic Omni-Loop is a local-first, zero-cost architectural framework designed to transform a standard coding assistant (Claude Code) into a self-improving, memory-persistent Artificial General Intelligence (AGI) aimed at autonomous software engineering. The system eliminates "amnesia" by implementing a three-tiered memory hierarchy (Hot/Warm/Cold) and leverages "Sleep-Time Compute" to iteratively refine its own skills, code patterns, and architectural understanding without human intervention.

### Data Lifecycle & Transition Layers

The critical failure point of most agents is the "handover" between memory states. This system enforces strict graduation protocols.

**Transition A: The "Digest" (Hot → Warm)**
- **Trigger:** Nightly "Sleep Cycle" Daemon (or system idle > 15m).
- **Input:** Raw interaction logs from **ByteRover** (cleaned via Proxy).
- **Process (The Dreamer):**
  - **Structuring:** Converts raw logs into strict Graphiti Nodes.
  - **Filtering:** Discards "chatter". Keeps only "Solved Problems" and "Architectural Decisions."
- **Output:** New Nodes added to **Graphiti**. Raw logs purged from ByteRover.

**Transition B: The "Freeze" (Warm → Cold)**
- **Trigger:** Weekly "Archivist" Job (Sunday).
- **Input:** Stale Graphiti nodes (>30 days inactive) + Curated "Gold Standard" datasets.
- **Process (The Renderer):**
  - **Deconstruction:** Serializes nodes into JSON.
  - **Rendering:** Generates QR Code images (PNGs) of the JSON data.
  - **Quad-Encoding:** Generates 4 vector layers (Token, Fact, Context, Boundary).
  - **Stitching:** Compiles images into an H.265 `.mp4` video file.
- **Output:** A portable **MemVid** archive file.

---

## Embedding Model Stack (January 2026 SOTA)

### Primary: Qwen3-VL-Embedding-8B

Released **January 7-8, 2026** (arXiv:2601.04720). This is a **vision-language multimodal** embedding model.

| Specification | Qwen3-VL-Embedding-8B | Qwen3-VL-Embedding-2B |
|---------------|----------------------|----------------------|
| Parameters | 8.14B | 2.13B |
| Layers | 36 | 28 |
| Context Length | 32,768 tokens | 32,768 tokens |
| Embedding Dimensions | 4096 | 2048 |
| MMEB-V2 Score | **77.8** (Rank #1) | 73.2 |

**Supported Input Modalities:**
- Pure text
- Pure image
- Pure video
- Text + image (mixed)
- Text + video (mixed)
- Image + video (mixed)
- Text + image + video (mixed)
- Screenshots (treated as images with OCR awareness)

### Reranker: Qwen3-VL-Reranker-8B

Two-stage retrieval pipeline requires precision reranking after initial recall.

| Specification | Qwen3-VL-Reranker-8B | Qwen3-VL-Reranker-2B |
|---------------|---------------------|---------------------|
| Parameters | 8.14B | 2.13B |
| Architecture | Single-Tower | Single-Tower |
| Input | (Query, Document) pairs | (Query, Document) pairs |
| Output | Relevance score | Relevance score |

---

## Chunking Strategies

Chunking has evolved into a _four-layer epistemic scaffolding system_:
1. **Fixed-length chunking**
2. **Sentence/semantic-unit chunking**
3. **Semantic coherence chunking (agentic)**
4. **Recursive hierarchical chunking (agentic)**

### Method 1: Fixed-Size Chunking (Data/Config)
- **Window tokens:** 512
- **Overlap tokens:** 50

### Method 2: Sentence-Based Chunking (Pure Text)
- **Window size:** 3 sentences
- **Min chunk tokens:** 128
- **Max chunk tokens:** 2048

### Method 3: Semantic Chunking (Pure Text)
- **Similarity threshold:** 0.75
- **Boundary detection model:** Qwen3-Embedding-0.6B

### Method 4: Recursive Hierarchical Chunking (Pure Text)
- **Chunk size tokens:** 1024
- **Overlap tokens:** 100
- **Separators:** ["\n\n", "\n", ". ", " "]

### Method 5: AST Structural Chunking (Pure Code)
- **Languages supported:** Python, TypeScript, JavaScript, Go, Rust, Java
- **Prepend parent context:** true
- **Preserve docstrings:** true

### Method 6: Multimodal Boundary Detection (NEW - Mixed Content)
- **Visual context window:** 1 paragraph before/after
- **Caption detection:** true

### Method 7: Screenshot-Code Fusion (NEW)
- **Matching strategies:** filename_similarity, ocr_text_matching, reference_comment_detection

---

## Metadata Schema

This schema is designed for cutting-edge RAG with full provenance tracking, semantic graph linkage, quality scoring, and cross-modal relationship mapping.

### Schema Overview: 12 Dimensions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CHUNK METADATA SCHEMA v3.0                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. IDENTITY          │  2. PROVENANCE        │  3. CONTENT               │
│  - chunk_id (UUID)    │  - source_uri         │  - content_type           │
│  - content_hash       │  - git_commit_sha     │  - modalities[]           │
│  - version            │  - author             │  - language               │
│  - parent_chunk_id    │  - created_at         │  - mime_type              │
│                       │  - modified_at        │  - byte_size              │
├─────────────────────────────────────────────────────────────────────────────┤
│  4. STRUCTURE         │  5. HIERARCHY         │  6. SEMANTIC              │
│  - chunk_method       │  - depth_level        │  - topic_cluster_id       │
│  - token_count        │  - section_path[]     │  - entities[]             │
│  - char_count         │  - heading_text       │  - keywords[]             │
│  - overlap_prev       │  - parent_heading     │  - summary                │
│  - overlap_next       │  - sibling_ids[]      │  - intent_class           │
├─────────────────────────────────────────────────────────────────────────────┤
│  7. CODE_SPECIFIC     │  8. MULTIMODAL        │  9. EMBEDDING             │
│  - ast_node_type      │  - referenced_images[]│  - model_id               │
│  - parent_scope       │  - referenced_code[]  │  - dimensions             │
│  - signature          │  - cross_modal_links[]│  - mrl_truncated          │
│  - imports[]          │  - visual_elements[]  │  - quantization           │
│  - complexity_score   │  - ocr_text           │  - embedding_hash         │
│  - docstring          │  - diagram_type       │  - embedded_at            │
├─────────────────────────────────────────────────────────────────────────────┤
│  10. GRAPH            │  11. QUALITY          │  12. RETRIEVAL            │
│  - incoming_refs[]    │  - confidence_score   │  - access_count           │
│  - outgoing_refs[]    │  - validation_status  │  - retrieval_success_rate │
│  - semantic_neighbors[]│ - error_flags[]      │  - user_feedback_score    │
│  - coreference_chain  │  - review_status      │  - freshness_decay        │
│  - dependency_graph   │  - chunking_quality   │  - last_accessed_at       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Roadmap

### Stage 1: Foundation (Week 1-2)
- [ ] Set up Qwen3-VL-Embedding-8B environment (flash_attention_2)
- [ ] Implement metadata schema and SQLite tables
- [ ] Build multimodal content detector
- [ ] Configure MemVid with H.265

### Stage 2: Chunking Pipeline (Week 3-4)
- [ ] Implement all 6 chunking methods
- [ ] Build multimodal boundary detection
- [ ] Add screenshot-code fusion logic
- [ ] Create cross-modal link discovery

### Stage 3: Agentic Swarm (Week 5-6)
- [ ] Build headless orchestrator with modality routing
- [ ] Implement specialized agents
- [ ] Add entity extraction and graph building
- [ ] Create quality validation pipeline

### Stage 4: Retrieval System (Week 7-8)
- [ ] Implement two-stage retrieval pipeline
- [ ] Add Qwen3-VL-Reranker integration
- [ ] Build retrieval analytics
- [ ] Performance optimization (target <550ms)

---

## Cost Analysis

| Component | Model | Cost |
|-----------|-------|------|
| Embedding | Qwen3-VL-Embedding-8B | ~$0.03/1M tokens* |
| Reranking | Qwen3-VL-Reranker-8B | ~$0.05/1M tokens* |
| Ingestion (35MB) | One-time | ~$0.10 |
| Queries (10K/day, annual) | - | ~$5.00 |

*Estimated - not yet on OpenRouter, requires self-hosting or wait for API availability.

---

## Code Reference

The system architecture incorporates several key technical components:

### Key Directories and Files
- **src/models/** - Contains Qwen3-VL model implementations
- **src/chunking/** - Multi-strategy chunking algorithms
- **src/memvid/** - Video-encoded storage system
- **src/graphiti/** - Knowledge graph implementation
- **src/orchestration/** - Agentic swarm coordination logic
- **src/retrieval/** - Two-stage retrieval pipeline

### Tech Stack
- **Primary Model**: Qwen3-VL-Embedding-8B
- **Reranker**: Qwen3-VL-Reranker-8B
- **Storage**: MemVid (H.265 compressed video)
- **Graph Database**: FalkorDB (via Bolt Protocol)
- **Vector Store**: ChromaDB/FAISS
- **Orchestration**: Headless Claude Code with MCP tools
- **Languages**: Python (primary), TypeScript (frontend)

---

## Source Documents

This unified PRD consolidates information from the following source documents:

1. **chatgpt5.2-prd.md** - Original RAG architecture requirements and multimodal chunking strategies
2. **gemini-prd.md** - Autodidactic Omni-Loop system with ByteRover, Graphiti, and MemVid
3. **opus-prd1-v3.md** - Cutting-edge RAG architecture v3.0 with Qwen3-VL-Embedding
4. **opus-prd2-v3.md** - Configuration specification for RAG v3.0
5. **opus-prd3-v3.md** - Foundational theory of multi-pass chunking and embedding model selection

All source documents contained within this repository have been incorporated into this unified specification.