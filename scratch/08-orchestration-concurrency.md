# Topic 08: Orchestration & Concurrency

## Summary
Implement the agentic orchestration layer that coordinates the ingestion pipeline, manages concurrent workers for different content types, handles batching, and integrates with MCP servers for tool access.

---

## Overview
> Sources: `opus-prd2-v3.md` (lines 297–333), `chatgpt5.2-prd.md` (lines 17–29, 46–50), `docs/UNIFIED_PRD.md` (lines 228–231, Implementation Roadmap Stage 3)

The orchestration layer runs headless via Claude Code CLI, coordinating multiple specialized agents/workers for content detection, routing, chunking, embedding, and graph building.

---

## Headless Operation
> Source: `opus-prd2-v3.md` (lines 300–303)

```yaml
orchestration:
  headless:
    enabled: true
    logic_file: "orchestration_logic_v3.md"
    checkpoint_interval_minutes: 5
```

- Runs as headless Claude Code CLI process
- Natural language orchestration logic (provider-agnostic)
- Checkpoints every 5 minutes for crash recovery
- Deployed via monitoring layer (e.g., autoclot or similar)

> Source: `chatgpt5.2-prd.md` (lines 46–50)

- Orchestration logic should be natural language (provider-agnostic)
- Agents deployed as `.claude.md`-based configurations with skills, plugins, and MCP tools
- Uses Anthropic agentic SDK pattern

---

## Concurrency Configuration
> Source: `opus-prd2-v3.md` (lines 305–313)

```yaml
concurrency:
  max_files: 50
  modality_detector_workers: 2
  content_router_workers: 2
  code_specialist_workers: 8
  text_specialist_workers: 8
  multimodal_specialist_workers: 4
  graph_builder_workers: 2
  integration_workers: 2
```

### Worker Types:
1. **Modality Detector (2 workers):** Classify incoming files by content type and modality
2. **Content Router (2 workers):** Route classified content to appropriate chunking pipeline
3. **Code Specialist (8 workers):** AST structural chunking for code files
4. **Text Specialist (8 workers):** Semantic, recursive hierarchical, sentence, and fixed-size chunking
5. **Multimodal Specialist (4 workers):** Multimodal boundary detection and screenshot-code fusion
6. **Graph Builder (2 workers):** Entity extraction, relationship building, semantic neighbor computation
7. **Integration (2 workers):** Final assembly — write chunks, embeddings, and metadata to storage

---

## Batching Configuration
> Source: `opus-prd2-v3.md` (lines 315–319)

```yaml
batching:
  embedding_batch_size: 16        # Smaller for multimodal (GPU memory)
  integration_buffer_size: 50     # Buffer before flushing to DB
  integration_flush_timeout_ms: 5000  # Max wait before flush
```

---

## MCP Server Integrations
> Source: `opus-prd2-v3.md` (lines 321–333)

| MCP Server | Description | Config |
|-----------|-------------|--------|
| `filesystem-mcp` | Sandboxed file system access | — |
| `git-mcp` | Git history for provenance | — |
| `embedding-mcp` | Unified multimodal embedding API | `default_model: Qwen/Qwen3-VL-Embedding-8B` |
| `memvid-mcp` | Video-encoded vector storage | — |
| `entity-mcp` | Named entity extraction | — |

---

## Asynchronous Processing
> Source: `chatgpt5.2-prd.md` (lines 17–23)

- Files should be processed asynchronously (multiple files in parallel)
- Fixed-size and sentence-based chunking: programmatic (no LLM needed)
- Semantic and recursive hierarchical chunking: require LLM intelligence (agentic)
- Question raised: Can semantic + recursive hierarchical share a single agent pass?
  - Answer not definitively provided; likely separate passes for quality

### Model Selection for Agentic Chunking
> Source: `chatgpt5.2-prd.md` (lines 20–22)

- Sonnet/Gemini Pro class: For complex semantic and hierarchical chunking
- Haiku/Gemini Flash class: For simpler agentic tasks
- Free models (MIMO V2 via OpenRouter): For cost-sensitive tasks
- Decision depends on required intelligence level for each chunking method

---

## Pipeline Flow

```
Input Files
    ↓
[Modality Detector] (2 workers)
    ↓
[Content Router] (2 workers)
    ↓ (fan-out by content type)
┌─────────────────────────────────────┐
│ [Code Specialist]     (8 workers)   │
│ [Text Specialist]     (8 workers)   │
│ [Multimodal Specialist] (4 workers) │
└─────────────────────────────────────┘
    ↓ (chunks + metadata)
[Graph Builder] (2 workers)
    ↓ (enriched with relationships)
[Integration] (2 workers)
    ↓
Storage (SQLite + MemVid + FAISS)
```

---

## Implementation Tasks

1. Create `src/orchestration/pipeline.py` — Main ingestion pipeline orchestrator
2. Create `src/orchestration/worker_pool.py` — Worker pool management with configurable concurrency
3. Create `src/orchestration/modality_detector.py` — File content type and modality classification
4. Create `src/orchestration/content_router.py` — Route content to appropriate chunking specialist
5. Create `src/orchestration/specialists/code.py` — Code chunking specialist worker
6. Create `src/orchestration/specialists/text.py` — Text chunking specialist worker
7. Create `src/orchestration/specialists/multimodal.py` — Multimodal chunking specialist worker
8. Create `src/orchestration/graph_builder.py` — Entity extraction and relationship building
9. Create `src/orchestration/integrator.py` — Buffered write to storage with flush logic
10. Create `orchestration_logic_v3.md` — Natural language orchestration instructions
11. Implement checkpoint/recovery mechanism

---

## Conflicts & Ambiguities

1. **Agent model selection:** `chatgpt5.2-prd.md` discusses using different LLM classes (Sonnet vs Haiku vs free) for different chunking tasks but doesn't make a final decision. The orchestration config in `opus-prd2-v3.md` doesn't specify which LLM each specialist uses. Need to define model assignments per worker type.

2. **Headless CLI vs daemon:** The orchestration runs as headless Claude Code CLI (`opus-prd2-v3.md`), but the sleep-time system runs as a `launchd` daemon (`gemini-prd.md` line 231). These are separate processes — the orchestration handles ingestion, the daemon handles sleep-time refinement.

3. **Max files = 50:** The `max_files: 50` limit in concurrency config is unclear — is this 50 files being processed simultaneously, or 50 files per batch? Given the worker counts (total ~28 workers), 50 simultaneous files seems reasonable.
