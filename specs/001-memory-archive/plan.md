# Implementation Plan: Personal AI Interaction Archive

**Feature**: 001-memory-archive  
**Status**: Draft  
**Branch**: 001-memory-archive

---

## Technical Context

### Architecture Overview

```
                    INGESTION LAYER
                    (Everything → single store)
                         │
                         ▼
              ELEMENT EXTRACTION PIPELINE
              (Raw → structured elements)
                         │
                         ▼
              CURATED KNOWLEDGE LAYER
              (Karpathy Wiki — pointer-based)
                         │
                         ▼
              MEMORY ORCHESTRATION LAYER
              (aaa-memory — routing, tiering)
                         │
              +----------+----------+
              │          │          │
           ClawMem   Graphiti   MemVid V2
           (Hot)     (Warm)     (Cold)
              │          │          │
              +----------+----------+
                         │
              [Score Fusion + Reranking]
                         │
              AGENT INTEGRATION LAYER
          (OpenClaw ContextEngine,
           Hermes MemoryProvider,
           Claude Code hooks, etc.)
```

### Storage Engine Decision

**Primary: SQLite + sqlite-vec** (unified across all tiers)

**Rationale**:
- Zero ops — no PostgreSQL daemon to manage
- File-portable — `~/.cache/clawmem/index.sqlite` can be backed up, synced, copied
- ClawMem already uses it natively
- sqlite-vec supports pre-computed embeddings, no re-encoding needed
- WAL mode handles concurrent multi-session reads
- FTS5 provides adequate BM25 for personal-scale (~500k turns max)

**When to consider PostgreSQL**:
- When concurrent write throughput becomes a bottleneck (> 10 sessions writing simultaneously)
- When tsvector-level full-text search is insufficient (unlikely for personal scale)
- For the web viewer if deployed as a shared service (Phase 7+)

**Schema**: See `schema.sql` below.

### Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Core package | Python 3.12+ | LLM ecosystem, Pydantic for schemas |
| Ingestion | LLM (Nemotron 3 Super, free) + rule-based classifier | Cost-free intent classification |
| Hot tier | ClawMem (TypeScript/Bun, SQLite + sqlite-vec) | Native multi-agent support, hooks + MCP |
| Warm tier | Graphiti (Python, Kuzu embedded) | Temporal knowledge graph, entity extraction |
| Cold tier | MemVid V2 (.mv2 files, HNSW index) | Compressed long-term archive |
| Embeddings | Qwen3-Embedding-8B (Ryzen), EmbeddingGemma-300M (Surface), Jina v3 (cloud fallback) | SOTA open embedder |
| Intent classifier | Nemotron 3 Super (120B/12B MoE, free on OpenRouter) | Free tier, 1M context |
| Reranker | Cross-encoder (qwen3-reranker-0.6B on Ryzen, CPU cosine on Surface) | Precision boost |
| Agent integration | OpenClaw ContextEngine, Hermes MemoryProvider ABC, Claude Code hooks | Native interfaces |
| Web viewer | Bun + Hono + Svelte (Phase 7) | Lightweight, progressive disclosure |

---

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| Local-first, zero cloud dependency | ✅ | Free-tier APIs only as fallback |
| Markdown as source of truth | ✅ | All wiki files are plain markdown with YAML frontmatter |
| Git-trackable | ✅ | `~/knowledge/` is a git repo |
| Human-readable | ✅ | No black-box storage — everything is inspectable |
| No vendor lock-in | ✅ | SQLite files, markdown, .mv2 — all portable |
| Never pay for software | ✅ | All components open-source or free-tier |

---

## Phase 0: Research

### R-001: Storage Engine Unification

**Decision**: SQLite + sqlite-vec as primary engine.

**Rationale**: ClawMem already uses this. It's file-portable, zero-ops, and handles our scale. PostgreSQL adds operational complexity without proportional benefit for single-user deployment.

**Alternatives considered**:
- PostgreSQL + pgvector: Better concurrent writes, tsvector full-text. But requires daemon, not file-portable, adds ops burden.
- DuckDB: Great for analytics, not for real-time retrieval.

### R-002: Embedding Model Selection

**Decision**: Multi-provider with priority chain:
1. Qwen3-Embedding-8B (Ryzen desktop, ~5GB VRAM) — SOTA for technical content
2. EmbeddingGemma-300M Q8 (Surface, ~0.4GB VRAM) — lightweight, local
3. Jina v3 API (cloud fallback, free tier) — when local unavailable

**Rationale**: Technical content benefits from 8B embedder, but Surface VRAM constraint forces lightweight fallback. Storing embeddings with the element avoids re-encoding.

### R-003: MemVid V2 Availability

**Decision**: Treat as optional. Primary cold tier: compressed markdown archives with BM25 (sqlite-vec FTS5). If MemVid V2 is available and working, use it. If not, the .mv2 format can be adopted later without schema changes — the adapter pattern isolates the backend.

**Rationale**: MemVid V2 references exist but no installable package found. The MemvidAdapter in aaa-memory spec already handles both cases (creates .mv2 if available, falls back to SQLite FTS5).

---

## Phase 1: Design

### Data Model

See `schema.sql` — the complete PostgreSQL/SQLite compatible schema.

### Key Design Decisions

1. **Raw is immutable** — `raw_text` never modified after ingestion. All enhancements go into derived fields.
2. **Derived is rebuildable** — summaries, embeddings, topic labels, clusters can be deleted and regenerated from raw.
3. **Single encode at extraction** — embeddings stored with element, reused by ClawMem, Graphiti, MemVid V2.
4. **Sub-indexes are the shard** — master index points to sub-indexes, not individual articles.
5. **Provenance chains** — every element tracks its origin through the pipeline.

### Agent Integration Mapping

| Agent | Integration Method | Hook/Event | Data Captured |
|-------|-------------------|------------|---------------|
| Claude Code | Lifecycle hooks + MCP | UserPromptSubmit, SessionStart, Stop, PreCompact | Full turns, tool calls, file ops |
| OpenClaw | ContextEngine plugin | before_prompt_build, afterTurn, compact | Context assembly, compaction snapshots |
| Hermes | MemoryProvider ABC + hermes-lcm plugin | search(), store(), health_check() | Session summaries, decisions, verbatim turns |
| Qwen Code | Context files + MCP | /memory refresh, PROJECT_SUMMARY.md | Context changes, session summaries |
| OpenCode | MCP + JSON parser | ses_*.json files | Full session history |
| Codex CLI | JSONL parser | rollout sessions | User prompts, model summaries |
| Web UIs | Tampermonkey | fetch/XHR interception | User prompts, model responses |

**Recommended Hermes Plugin**: hermes-lcm (Lossless Context Management) — preserves every message verbatim in SQLite before compaction fires. Ensures our ingestion pipeline captures complete session data, not truncated summaries. Install alongside Hermes v0.7.0+. Not part of aaa-memory itself — complementary context compression layer.

### API Contracts

Not REST APIs — the system is accessed via:
1. **Lifecycle hooks** (Claude Code, OpenClaw)
2. **MCP tools** (all agents, 31+ tools from ClawMem)
3. **MemoryProvider ABC** (Hermes)
4. **Direct Python API** (scripts, cron jobs)

No HTTP endpoints needed until Phase 7 (web viewer).

---

## Research Summary

| Decision | Choice | Why |
|----------|--------|-----|
| Storage engine | SQLite + sqlite-vec | Zero ops, file-portable, ClawMem native |
| Embedding model | Qwen3-Embedding-8B → EmbeddingGemma-300M → Jina v3 | SOTA → lightweight → cloud fallback |
| Intent classifier | Nemotron 3 Super (free OpenRouter) | 120B/12B MoE, free, 1M context |
| Graph engine | ClawMem's built-in multi-graph (no separate Graphiti for Phase 1) | Avoids running two graph engines. Add Graphiti in Phase 5 if temporal queries need it. |

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Install ClawMem: `bun install`, `clawmem init`, `clawmem setup hooks`, `clawmem setup mcp`
- [ ] Create `~/knowledge/` directory structure with sub-indexes
- [ ] Install joshpocock/karpathy-obsidian-vault schema as `~/knowledge/CLAUDE.md`
- [ ] Initialize git repo in `~/knowledge/`
- [ ] Write Tampermonkey scripts for web UIs (ChatGPT, Gemini, Claude web)
- [ ] Write daily update service skeleton (cron at 2 AM scanner)

### Phase 2: Ingestion Pipeline (Week 1-2)
- [ ] Build document classifier (rule-based, covers ~70% of files)
- [ ] Build element extractor (LLM-based, Nemotron 3 Super free tier)
- [ ] Build metadata injector (frontmatter + provenance)
- [ ] Build embedding module (EmbeddingGemma-300M on Surface)
- [ ] Test on 10 raw transcripts — review extraction quality

### Phase 3: 800-File Vault Migration (Week 2-3)
- [ ] Run classifier on all 800 files → classification_report.json
- [ ] Route by tier into `~/knowledge/raw/` subdirectories
- [ ] Batch element extraction on all Tier 5 transcripts (overnight, 20 per batch)
- [ ] Human review of extractions (approve/reject/edit)
- [ ] Index all wiki/ files into ClawMem
- [ ] Ingest PRDs into Graphiti as episodes
- [ ] Archive raw transcripts to cold storage
- [ ] **Refinement checkpoint at 15%**: validate extraction quality, adjust prompt
- [ ] **Refinement checkpoint at 30%**: cross-reference elements, merge duplicates
- [ ] **Refinement checkpoint at 50%**: schema evolution, re-classify noise
- [ ] **Refinement checkpoint at 75%**: graph density, orphan detection
- [ ] **Refinement checkpoint at 100%**: full audit, lint wiki, final report

### Phase 4: Session Audit (Week 3-4)
- [ ] Discovery scanner: find sessions across all 6 agent storage locations
- [ ] Session classifier: infer project_id, session_type, key_decisions
- [ ] Project timeline assembler: chronological view with related session links
- [ ] CLI: `aaa-memory sessions`, `aaa-memory timeline`, `aaa-memory audit --update`
- [ ] MCP tools for agent-accessible session queries
- [ ] Embed session summaries (chaff stripped, decisions preserved)
- [ ] Create knowledge_edges between related sessions (same project, 48h window)

### Phase 5: Memory Router (Week 4-5)
- [ ] Intent classifier: Nemotron 3 Super + rule-based fallback
- [ ] Three retrieval strategies (Score Fusion, Cascade, Intent-Routed)
- [ ] Score fusion with configurable weights
- [ ] Token budget enforcement
- [ ] Echo-loop prevention

### Phase 6: Agent Integration (Week 5-6)
- [ ] OpenClaw: Register as ContextEngine plugin
- [ ] Hermes: Register as MemoryProvider plugin (+ recommend hermes-lcm install)
- [ ] Qwen Code: Context file injection + MCP
- [ ] OpenCode/Codex: Parsers for session files
- [ ] Cross-agent memory sharing test

### Phase 7: Tier Transitions + Sleep-Time Compute (Week 6-7)
- [ ] Weekly Hot→Warm daemon
- [ ] Monthly Warm→Cold daemon
- [ ] Overnight improvement loop
- [ ] Lint → Fix pipeline
