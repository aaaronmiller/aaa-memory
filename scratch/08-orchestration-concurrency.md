# Topic: Orchestration & Concurrency

## Summary
Headless agentic orchestration via Claude Code, concurrency settings, batching strategies, MCP server configurations, and the agentic swarm architecture.

---

## Headless Operation

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md), [chatgpt5.2-prd.md](../chatgpt5.2-prd.md)

```yaml
orchestration:
  headless:
    enabled: true
    logic_file: "orchestration_logic_v3.md"
    checkpoint_interval_minutes: 5
```

- Orchestration logic is defined in natural language (markdown file)
- Provider-agnostic — designed for Anthropic agentic SDK
- Deployed via Claude Code headless CLI with monitoring layer (e.g., autoclot)
- Uses `.claude/claude.md` based skills, plugins, and MCP tools

---

## Concurrency Settings

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md)

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

### Worker Roles

| Worker Type | Count | Purpose |
|------------|-------|---------|
| Modality Detector | 2 | Detect content modalities in incoming files |
| Content Router | 2 | Route content to appropriate chunking pipeline |
| Code Specialist | 8 | AST parsing, code chunking, dependency extraction |
| Text Specialist | 8 | Semantic/recursive/sentence chunking for text |
| Multimodal Specialist | 4 | Multimodal boundary detection, screenshot-code fusion |
| Graph Builder | 2 | Entity extraction, relationship building, semantic neighbors |
| Integration | 2 | Final assembly, quality validation, storage |

---

## Batching Configuration

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md)

```yaml
batching:
  embedding_batch_size: 16          # Smaller for multimodal (memory constraints)
  integration_buffer_size: 50
  integration_flush_timeout_ms: 5000
```

---

## MCP Server Configuration

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md)

| MCP Server | Description | Config |
|-----------|-------------|--------|
| filesystem-mcp | Sandboxed file system access | — |
| git-mcp | Git history for provenance | — |
| embedding-mcp | Unified multimodal embedding API | default_model: Qwen3-VL-Embedding-8B |
| memvid-mcp | Video-encoded vector storage | — |
| entity-mcp | Named entity extraction | — |

---

## Agentic Swarm Architecture

> **Source:** [chatgpt5.2-prd.md](../chatgpt5.2-prd.md), [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md)

### Ingestion Pipeline Agents

The orchestrator coordinates specialized agents for the ingestion pipeline:

1. **Modality Detector Agent** — Classifies incoming content type and modalities
2. **Content Router Agent** — Routes to appropriate chunking strategy
3. **Chunking Agents** (per method):
   - Fixed-size chunker (programmatic, no LLM)
   - Sentence-based chunker (programmatic, no LLM)
   - Semantic chunker (uses Qwen3-Embedding-0.6B for boundary detection)
   - Recursive hierarchical chunker (may need Sonnet/Pro-class LLM)
   - AST structural chunker (tree-sitter based, programmatic)
   - Multimodal boundary chunker (needs VL model)
   - Screenshot-code fusion chunker (needs VL model + OCR)
4. **Entity Extraction Agent** — NER and relationship extraction
5. **Graph Builder Agent** — Knowledge graph construction
6. **Quality Validation Agent** — Chunk quality scoring
7. **Integration Agent** — Final assembly and storage

### Asynchronous Processing
> **Source:** [chatgpt5.2-prd.md](../chatgpt5.2-prd.md)

- Files should be processed asynchronously (multi-file corpus)
- Different chunking methods can run in parallel on different files
- Embedding can be batched across chunks

---

## Orchestration Logic File

> **Source:** [chatgpt5.2-prd.md](../chatgpt5.2-prd.md)

The orchestration logic should be:
- Natural language (provider-agnostic)
- Markdown-based configuration
- Compatible with Anthropic agentic SDK
- Deployable via Claude Code headless CLI

---

## Key Directories

> **Source:** [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md)

```
src/models/          — Qwen3-VL model implementations
src/chunking/        — Multi-strategy chunking algorithms
src/memvid/          — Video-encoded storage system
src/graphiti/        — Knowledge graph implementation
src/orchestration/   — Agentic swarm coordination logic
src/retrieval/       — Two-stage retrieval pipeline
```

---

## Implementation Requirements

1. Create orchestration logic markdown file
2. Implement worker pool with configurable concurrency
3. Build content routing logic (content type → chunking method)
4. Implement async file processing pipeline
5. Set up MCP server integrations
6. Build checkpoint/resume system (5-minute intervals)
7. Implement embedding batching with configurable batch size
8. Create monitoring/logging for worker status

---

## Conflicts / Ambiguities

- **⚠️ Agent vs programmatic:** chatgpt5.2-prd.md envisions LLM agents for semantic/hierarchical chunking, but opus-prd2-v3.md treats these as algorithmic processes with configurable parameters. The implementation should use algorithmic approaches with LLM fallback for edge cases.
- **⚠️ Orchestration tool:** chatgpt5.2-prd.md mentions "autoclot or something" as monitoring layer. This is vague — the specific monitoring tool needs to be determined.
- **⚠️ Worker counts:** The concurrency settings (8 code + 8 text + 4 multimodal = 20 specialist workers) assume significant compute resources. May need to be tuned for M3 Max MacBook Pro.
