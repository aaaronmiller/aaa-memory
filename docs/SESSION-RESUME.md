# Session Resume — Memory Architecture Project
**Date**: 2026-04-08  
**Session duration**: ~3 hours (2:00 PM - 5:00 PM PDT)  
**Save timestamp**: $(date '+%Y-%m-%d %H:%M:%S %Z')  
**Status**: URGENT — system crash recovery. ALL context preserved.

---

## 1. WHAT WE'RE BUILDING (Elevator Pitch)

A unified memory architecture for all AI agents (Claude Code, Qwen Code, OpenClaw, Hermes, OpenCode, Codex CLI) that:
- Ingests every interaction (prompts, responses, tool calls, decisions)
- Extracts structured knowledge elements from raw transcripts
- Stores in a Karpathy-style wiki with [[wikilink]] pointers
- Provides hybrid retrieval (BM25 + vector + graph + metadata filters)
- Manages tiered storage (Hot → Warm → Cold → Frozen)
- Audits sessions across all agents and assembles project timelines
- Self-improves via sleep-time compute with adaptive refinement checkpoints

---

## 2. KEY DECISIONS MADE (Unchangeable Without User Input)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage engine | **SQLite + sqlite-vec** | Zero ops, file-portable, ClawMem native, WAL mode for concurrency |
| Embedding model | **Qwen3-Embedding-8B** (Ryzen), **EmbeddingGemma-300M Q8** (Surface), **Jina v3** (cloud fallback) | Single encode at extraction time, reuse everywhere |
| Intent classifier | **Nemotron 3 Super** (free on OpenRouter) + rule-based fallback | 120B/12B MoE, free, 1M context |
| Graph engine | **ClawMem's built-in multi-graph** for Phase 1 | Avoid running two graph engines. Add Graphiti in Phase 5 if needed |
| Cold storage | **MemVid V2** (.mv2 files) | 90% size reduction, scheduled compute, negligible storage |
| Frozen storage | **H.265-compressed embeddings** (NEW) | 10-50x compression via temporal deltas on project-grouped embeddings |
| Karpathy wiki | **Sub-indexed sharding** | Master index → sub-indexes (~500-700 items each) → never exceed soft limit |
| Agent injection | **Native interfaces** (OpenClaw ContextEngine, Hermes MemoryProvider, Claude Code hooks) | Don't rebuild what exists |
| hermes-lcm | **Recommended plugin** (not integrated) | Complementary context compression layer for Hermes sessions |
| MoE vs dense | **MoE-first for free tier** | Inference economics: MoE serves more concurrent users per GPU on OR |
| Fallback ordering | **Tool-use reliability first**, not general intelligence | Cold-start tool calls are the primary failure mode |
| Refinement checkpoints | **At 15/30/50/75/100% ingestion** | Auto-adjust extraction prompts, detect duplicates, evolve schema |

---

## 3. OPTIMAL HERMES CONFIG (Validated Against PinchBench + Community Reports)

```yaml
model:
  default: arcee-ai/trinity-large-preview:free     # 77.7% PinchBench free (measured)
  tool_calls: minimax/minimax-m2.5:free             # 87.8% PinchBench — best free tool-use
  auxiliary: google/gemma-4-26b-a4b-it:free         # 83.9% PinchBench (MoE variant, NOT dense 31B)
  base_url: http://127.0.0.1:8082/v1
  api_mode: chat_completions

fallback_providers:
  - provider: openrouter
    model: minimax/minimax-m2.5:free               # 87.8% PinchBench
  - provider: openrouter
    model: z-ai/glm-4.5-air:free                   # 85.7% PinchBench, 76.4% BFCL-v3
  - provider: openrouter
    model: stepfun/step-3.5-flash:free             # 85.3% PinchBench
  - provider: openrouter
    model: google/gemma-4-26b-a4b-it:free          # 83.9% PinchBench
  - provider: openrouter
    model: arcee-ai/trinity-large-preview:free     # 77.7% PinchBench free (measured)
  - provider: openrouter
    model: nvidia/nemotron-3-super-120b-a12b:free  # 75.0% PinchBench free (measured)

smart_model_routing:
  enabled: true
  max_simple_chars: 160
  max_simple_words: 28
  cheap_model:
    provider: openrouter
    model: nvidia/nemotron-3-nano-30b-a3b:free     # Community-validated, OpenClaw is #1 user

compression:
  enabled: true
  threshold: 0.50      # Was 0.25 — reduce cache-breaking compression cycles
  target_ratio: 0.20   # Was 0.10 — retain more context per cycle
  protect_last_n: 10   # Was 5 — protect more recent tool results

agent:
  max_turns: 240
  gateway_timeout: 1800
  tool_use_enforcement: auto
  reasoning_effort: medium
```

**What was CHANGED from user's original config:**
- Removed `gemma-4-31b-it:free` from fallback (dense, beaten by MoE variant 76.4% vs 83.9%)
- Removed `gpt-oss-120b:free` from fallback (67.1% PinchBench, WORST at agent loops despite 90% MMLU-Pro)
- Added `glm-4.5-air:free` (85.7% PinchBench, was missing entirely)
- Added `gemma-4-26b-a4b-it:free` (MoE variant at 83.9%)
- Reordered: tool-use-first ordering, NOT "smartest first"
- Compression: threshold 0.25→0.50, target 0.10→0.20, protect_last_n 5→10

**Equivalent OpenClaw config** (`openclaw.json`):
```json
{
  "models": {
    "default": "arcee-ai/trinity-large-preview:free",
    "tools": "minimax/minimax-m2.5:free",
    "fallback": [
      "minimax/minimax-m2.5:free",
      "z-ai/glm-4.5-air:free",
      "stepfun/step-3.5-flash:free",
      "google/gemma-4-26b-a4b-it:free",
      "arcee-ai/trinity-large-preview:free",
      "nvidia/nemotron-3-super-120b-a12b:free"
    ]
  },
  "settings": {
    "smart_routing": true,
    "cheap_model": "nvidia/nemotron-3-nano-30b-a3b:free",
    "compression_threshold": 0.50,
    "compression_target": 0.20,
    "max_turns": 240,
    "gateway_timeout": 1800
  }
}
```

---

## 4. PINCHBENCH SCORES (The Complete Validated Table)

| Model | PinchBench Best | PinchBench Avg | BFCL-v3 | Tau-Bench | Galileo Tool Selection | Free on OR? |
|-------|----------------|----------------|---------|-----------|----------------------|-------------|
| minimax/minimax-m2.5 | 87.8% | 79.4% | Unpublished | Unpublished | Unpublished | YES |
| z-ai/glm-4.5-air | 85.7% | 77.7% | 76.4% | 77.9% Retail | 1.00 (perfect) | YES |
| stepfun/step-3.5-flash | 85.3% | 76.9% | Unpublished | Unpublished | Unpublished | YES |
| google/gemma-4-26b-a4b-it (MoE) | 83.9% | 77.2% | Unpublished | Unpublished | Unpublished | YES |
| arcee-ai/trinity-large-preview:free | 77.7% | 65.1% | None | Unpublished | Unpublished | YES |
| google/gemma-4-31b-it (dense) | 76.4% | 68.4% | None | Unpublished | Unpublished | YES |
| nvidia/nemotron-3-super:free | 75.0% | 69.6% | Unpublished | Unpublished | Unpublished | YES |
| openai/gpt-oss-120b | 67.1% | 50.2% | ~67-68% | 67.8% Retail | Unpublished | YES |

**Free vs Paid measured gap** (only 2 models have both):
- Nemotron 3 Super: paid 88.6% vs free 75.0% = **-13.6 point drop**
- Trinity Large Preview: paid 80.6% vs free 77.7% = **-2.9 point drop**

The gap varies by provider infrastructure quality, not model capability.

**General intelligence (MMLU-Pro)** — different ranking:
- gpt-oss-120b: 90.0% (smartest free model for general reasoning)
- minimax-m2.5: 85.0%
- glm-4.5-air: ~81%
- trinity-large-preview: ~78-82% (est.)

**Key insight**: GPT-OSS-120B is the smartest free model (90% MMLU-Pro) but the WORST at agent loops (67.1% PinchBench). This is why separating default and tool_calls model slots matters.

---

## 5. MEMORY ARCHITECTURE — Full Stack

### Directory Structure

```
~/knowledge/                          # Karpathy Wiki (git-tracked)
├── CLAUDE.md                         # Schema + agent workflow rules (from joshpocock)
├── raw/                              # Immutable source documents
│   ├── prds/                         # ~50 files, 30-50k each
│   ├── youtube/                      # ~300 files, <10k each
│   ├── papers/                       # ~150 files, up to 50k each
│   └── transcripts/                  # ~300 files, up to 200k each
├── wiki/                             # LLM-compiled, sharded by sub-index
│   ├── index.md                      # Master pointer → sub-indexes ONLY (not individual articles)
│   ├── log.md                        # Chronological activity log
│   ├── projects/                     # ~50 items — one dir per project
│   ├── research/                     # ~450 items — ai-agents/, compression/, memory-systems/
│   ├── concepts/                     # ~1000 items — agent-patterns/, memory-arch/
│   ├── prompts/                      # ~500 items — claude-code/, research/, creative/
│   ├── code/                         # ~300 items — hooks/, systemd/, ai-models/
│   └── decisions/                    # ~200 items — proxy-arch/, memory-design/, infra/
└── references/                       # S-tier references for sleep-time compute
    ├── technical-writing/
    ├── architecture-decisions/
    ├── research-summaries/
    └── prd-templates/

~/.cache/clawmem/index.sqlite         # Shared memory vault (WAL mode, FTS5 + sqlite-vec)
~/.aaa-memory/                        # Runtime state
├── config/config.yaml                # Router weights, thresholds, providers
├── state/extract_progress.json       # Extraction state (resume on crash)
├── state/prefetch_cache/             # Session-local prefetch results
├── hot/sessions/                     # Per-session JSONL files
├── cold/
│   ├── raw-transcripts.mv2           # MemVid V2 archived transcripts
│   └── embedding-archives/           # H.265-compressed embedding videos (Tier 4)
├── logs/                             # Structured JSON logs
└── run/strategy_override.json        # Orchestrator strategy override
```

### Storage Tiers

| Tier | Storage | Retention | Query Latency | Content |
|------|---------|-----------|---------------|---------|
| **Hot** | ClawMem SQLite + sqlite-vec | < 7 days | < 50ms | Active sessions, extracted elements |
| **Warm** | Graphiti Kuzu graph | 7-90 days | < 150ms | Entity relationships, temporal knowledge |
| **Cold** | MemVid V2 .mv2 files | 90+ days | < 500ms | Compressed documents with provenance |
| **Frozen** | H.265 .mp4 + index | 180+ days | < 1s | Compressed embeddings only (temporal deltas) |

### Element Extraction Pipeline

```
Raw transcript (up to 200k)
  → LLM classifies document type (rule-based covers ~70%)
  → For transcripts: LLM extracts elements:
    ├── decisions     → wiki/decisions/ + Graphiti episodes
    ├── patterns      → wiki/concepts/
    ├── code snippets → wiki/code/
    ├── prompts       → wiki/prompts/
    ├── facts         → ClawMem indexed (hot tier)
    ├── concepts      → wiki/concepts/
    └── noise         → discarded
  → Each element gets:
    ├── YAML frontmatter (title, type, tags, confidence, importance)
    ├── Provenance metadata (source, source_type, extracted_at, project, agent, session_id)
    ├── [[wikilinks]] to related concepts
    ├── Embedding (single encode at extraction time — stored in 3 places)
    └→ Stored in: markdown frontmatter (base64), SQLite vec0 table, element metadata JSON
```

### Session Audit System (FR-016, Phase 4)

```
Discovery scanner → finds sessions across all agents:
  ├── Claude Code: ~/.claude/projects/*/ JSONL
  ├── OpenClaw: ~/.openclaw/sessions/*.json
  ├── Qwen Code: ~/.qwen/logs/ text logs
  ├── OpenCode: ~/.local/share/opencode/storage/session/ ses_*.json
  ├── Codex CLI: ~/.codex/sessions/ JSONL rollout
  └── Hermes: ~/.hermes/state.db + memory files

Session classifier → infers:
  ├── project_id (from working dir, file paths, content topics)
  ├── session_type (planning, coding, debugging, testing, docs, audit)
  ├── key_decisions
  ├── token_summary
  ├── start_time, end_time, duration
  └── related_sessions (same project, 48h window)

Output:
  ├── CLI: aaa-memory sessions --project X
  ├── CLI: aaa-memory timeline --project X --last 7d
  ├── CLI: aaa-memory audit --project X --update
  └── MCP: agent-accessible session queries
```

### Adaptive Refinement Checkpoints (FR-017)

| Milestone | Action |
|-----------|--------|
| **15%** (~120 files) | Validate extraction quality. If < 70% approval: adjust prompt, re-extract batch. Update schema if new element types discovered. |
| **30%** | Cross-reference extracted elements → detect duplicate concepts across sources → merge related wiki pages → update [[wikilink]] graph → validate embedding consistency. |
| **50%** | Schema evolution check → identify uncaptured metadata fields → detect new topic clusters needing sub-indexes → re-run classification on previously "noise" elements → generate mid-point quality report. |
| **75%** | Graph density analysis → identify orphan wiki pages (zero inbound links) → detect isolated concept clusters → suggest new [[wikilink]] connections → validate retrieval pipeline against actual query patterns. |
| **100%** | Full audit → lint entire wiki → re-embed all elements if embedding model improved → generate project timelines from session data → produce final quality report. |

### Hybrid Retrieval Pipeline

```
User Query → Query Parser (detects SQL/semantic/graph/hybrid + extracts metadata filters + intent signal)
  → 4 parallel retrieval modes:
    ├── Lexical: FTS5 BM25 (exact token matching)
    ├── Semantic: sqlite-vec cosine similarity (meaning-based)
    ├── Graph: edge traversal with MPFP (relationship queries)
    └── Metadata: JSON filters (project, date, agent, type)
  → Reciprocal Rank Fusion (k=60)
  → Metadata filter application
  → Cross-encoder reranker (top-50 → top-10)
  → Token budget enforcer (greedy select, truncate at sentence boundary, default 2000 tokens)
  → Progressive disclosure (3 levels: Collapsed → Summary → Full)
```

### Intent-to-Tier Mapping

| Intent Signal | Primary Tier(s) | Query Mode |
|---------------|----------------|------------|
| "just now", "this session", "the error I got" | Hot | Lexical + Metadata |
| "why did we", "how does X relate to Y" | Warm | Graph + Semantic |
| "months ago", "that old project" | Cold | Lexical + Metadata |
| "what is the API for", "show me the config" | Hot + Warm | Lexical + Semantic |
| Broad/ambiguous | All tiers | Hybrid (all modes) |

---

## 6. DATABASE SCHEMA SUMMARY

10 tables in `specs/001-memory-archive/schema.sql`:

1. **turns** — Atomic unit: source_system, project_id, session_id, role, raw_text (immutable), normalized_text, summary_short/full, model_name, provider, tool_calls (JSON), files_touched (JSON), commands_run (JSON), git_branch, intent_category, topic_labels (JSON), failure_mode, embedding_vector_id
2. **elements** — Extracted knowledge: element_type (decision/pattern/code/prompt/fact/concept), title, content, confidence, importance, tags, related_links ([[wikilinks]]), source_file, wiki_path, embedding_vector_id
3. **embeddings** — Pre-computed at extraction: entity_type, entity_id, vector (BLOB), model_name, model_dim, chunk_index, chunk_total
4. **wiki_pages** — Karpathy pointers: path, sub_index, page_type, content_hash, sources (JSON), related_pages (JSON), pinned, access_count, last_accessed
5. **knowledge_edges** — Relationships: source_node, target_node, edge_type (relates_to/part_of/solved_by/caused/supersedes/contradicts/derived_from/used_in), weight, evidence_turns (JSON), confidence
6. **extracted_skills** — Reusable patterns: skill_name, canonical_prompt, source_turns, source_elements, usage_count, success_rate, promoted_to_command
7. **slash_command_candidates** — High-frequency operations: command_name, steps (JSON), frequency, confidence
8. **retrieval_queries** — Self-improvement: query_text, query_type, filters (JSON), results_returned, results_clicked (JSON), satisfaction
9. **schema_evolution** — Migration log: schema_version, migration_sql, justification, triggered_by, rolled_back
10. **ingestion_state** — Crash recovery: source_file, last_offset (byte), last_turn_id, status, error_message

Plus: FTS5 virtual table with auto-sync triggers (INSERT/UPDATE/DELETE), indexes optimized for all retrieval modes, and sqlite-vec vec0 table documentation.

---

## 7. RESEARCH FINDINGS FROM THIS SESSION

### Supermemory (supermemoryai, 21.5k stars)
- **What it is**: Ontology-driven fact tracker — not Karpathy wiki, not standard RAG
- **Architecture**: PostgreSQL + Drizzle ORM + Cloudflare Workers/KV
- **Key features**: Auto fact extraction, dynamic user profiles (~50ms latency), hybrid RAG+memory search, temporal metadata for expiration/contradiction handling
- **Benchmarks**: #1 on LongMemEval, LoCoMo, ConvoMem
- **What we steal**: Temporal versioning concept — add `temporal_version`, `superseded_by`, `expires_at`, `contradiction_of` to elements table so facts can evolve
- **What we don't**: Cloud dependency. Supermemory runs on Cloudflare Workers. We stay local-first.
- **NOT Karpathy wiki**: It tracks stateful facts with temporal metadata, not pointer-based documents.

### hermes-lcm (Lossless Context Management)
- **What it is**: Context compression plugin for Hermes Agent, NOT a memory system
- **Architecture**: Immutable SQLite for verbatim messages, Summary DAG (D0 minutes → D1 hours → D2 days), 3-tier fallback (L1 detailed → L2 bullets → L3 truncation)
- **Tools**: lcm_grep, lcm_describe, lcm_expand
- **Status**: MVP, 35 tests, requires PR #5700 merged first, tested against Hermes v0.7.0
- **Verdict**: Complementary to our system. Recommended as Hermes plugin for better session fidelity. Our ingestion pipeline benefits because we get complete session data instead of truncated summaries.

### MemVid V2
- **What it is**: Rust rewrite of Python v1. Stores AI memory as "Smart Frames" in single .mv2 file
- **Format**: Content + embedding + timestamp + relationship metadata per frame
- **Compression**: 90% size reduction — ~50,000 documents in ~200MB
- **Retrieval**: Sub-5ms to sub-17ms for 50k documents, >60% higher accuracy than traditional RAG
- **Features**: WAL mode for crash safety, zero cloud dependencies, model-agnostic
- **Keep as cold tier.** No reason not to use it.

### LLM.265 Paper (HKUST, ACM MM 2025) — "Video Codecs are Secretly Tensor Codecs"
- **The insight**: H.264/H.265/VP9 are not video-specific. They're general-purpose tensor compressors that exploit spatio-temporal redundancy.
- **Mechanism**: I-frame (full reference tensor) + P-frames (delta only) + motion vectors + DCT + quantization
- **For our system**: Session embeddings are highly temporally redundant. Same project across 10 sessions = vectors change incrementally. H.265 treats them as "video frames" → 10-50x compression vs raw FP32.
- **Our novel contribution**: Nobody has applied this to semantic embedding sequences for memory systems.

### N4MC (Neural 4D Mesh Compression, arXiv Feb 2026)
- **What it is**: First neural framework for compressing time-varying 4D mesh sequences
- **Compression**: 89.56x for static meshes, 4-6x better than existing methods for 4D sequences
- **Architecture**: Volumetric TSDF-Def tensors → 3D ConvNeXt autoencoder → latent compression → 3D interpolation transformer
- **NOT applicable to text embeddings**: Intrinsically 3D-geometric. Uses 3D convolutions, Marching Cubes, volume tracking. Can't apply to abstract semantic vectors.

### Gaussian Splatting Compression (3DGS)
- **What it is**: Real-time 3D scene rendering using Gaussian primitives
- **Compression techniques**: Progressive quantization (PCGS, NeurIPS 2025), vector quantization with codebooks (3DGS.zip, Eurographics 2026), distribution regularization + probabilistic pruning (AAAI 2026)
- **NOT applicable directly**: 3DGS works because scenes have geometric locality. Text embeddings have no spatial structure.
- **What we steal**: Progressive quantization for embedding precision. Vector quantization codebooks for embedding compression. Pruning for stale/low-value embeddings.

### Google TurboQuant (ICLR 2026)
- **What it is**: KV cache compression for LLM inference (6x reduction via PolarQuant + QJL)
- **NOT applicable**: Strictly for inference-time working memory, not persistent storage

---

## 8. IMPLEMENTATION PHASES

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **1: Foundation** | Week 1 | ClawMem install, ~/knowledge/ structure, Karpathy wiki schema, Tampermonkey scripts |
| **2: Ingestion Pipeline** | Week 1-2 | Document classifier, element extractor, metadata injector, embedding module |
| **3: 800-File Vault Migration** | Week 2-3 | Classify all files, batch extraction, human review, index into ClawMem, refinement checkpoints at 15/30/50/75/100% |
| **4: Session Audit** | Week 3-4 | Discovery scanner, session classifier, timeline assembler, CLI + MCP |
| **5: Memory Router** | Week 4-5 | Intent classifier, 3 retrieval strategies, score fusion, token budget |
| **6: Agent Integration** | Week 5-6 | OpenClaw ContextEngine, Hermes MemoryProvider, Qwen context injection, OpenCode/Codex parsers, hermes-lcm recommendation |
| **7: Tier Transitions + Sleep-Time** | Week 6-7 | Weekly Hot→Warm, monthly Warm→Cold, H.265 embedding compression, overnight improvement loop, lint → fix pipeline |

---

## 9. OPEN QUESTIONS (Unresolved)

1. **MemVid V2 installable?** — Not verified as pip-installable. Adapter pattern isolates backend. Start with compressed markdown + FTS5, add MemVid when available.
2. **Qwen3.6-plus evaluation** — User had great results during free release. Scores 88.6% PinchBench. Costs money. Needs investigation vs current free models.
3. **MVP scope** — Doing both: ingest 800 vault files AND set up continuous capture. Parallel tracks.

---

## 10. SIDE PROJECTS / TANGENTIAL ITEMS (NOT MVP)

| # | Item | Status |
|---|------|--------|
| 1 | **H.265 Frozen Tier** — Compressed embedding archive via temporal deltas | Research only, Phase 7 addition |
| 2 | **4D/5D/6D Neural Compression** — N4MC-style for memory | Discussed, rejected — not applicable to text embeddings |
| 3 | **Gaussian Splatting for Non-Visual Data** — 3DGS compression on embeddings | Partial steal — quantization + codebooks only, not splatting |
| 4 | **Supermemory Ontology** — Temporal versioning for facts | Adopt concept only — add temporal fields to elements table |
| 5 | **hermes-lcm Plugin** — Lossless context management | Recommended plugin, not integrated |
| 6 | **Tampermonkey Web UI Capture** — ChatGPT/Gemini/Claude web intercept | Phase 1, low priority |
| 7 | **Reddit Viral Article** — Free model stack config | Complete, saved to disk |
| 8 | **Session Audit + Project Timeline** | Integrated into MVP — Phase 4 |
| 9 | **Adaptive Refinement Checkpoints** | Integrated into MVP — Phase 3 |
| 10 | **OpenClaw Supermemory Plugin** | Rejected — cloud-dependent |
| 11 | **Google TurboQuant** | Rejected — irrelevant |
| 12 | **Qwen3.6-plus Evaluation** | Open question |

---

## 11. FILES ON DISK

### In `~/code/aaa-memory/` (the project)
```
aaa-memory/
├── .specify/memory/constitution.md
├── specs/001-memory-archive/
│   ├── spec.md                          # REQUIREMENTS (WHAT)
│   ├── plan.md                          # DESIGN (HOW)
│   ├── schema.sql                       # 10 tables + FTS5 + indexes
│   ├── retrieval-pipeline.md            # 4-mode hybrid retrieval
│   ├── quickstart.md                    # Setup + daily use
│   └── checklists/requirements.md       # Spec quality checklist
├── docs/
│   └── reddit-free-model-stack-article.md  # Viral Reddit post
├── src/aaa_memory/{adapters,transitions,hooks}/  # Empty, for Phase 1+
├── scripts/
├── config/
└── tests/
```

### In `~/code/` (research docs)
```
MEMORY-MASTERPLAN-v3.md    # Full architecture with diagrams
UNIFIED-MEMORY-PLAN.md     # Earlier v1 assessment
GNOME_AUDIO_BUG_MASTER.md  # Audio bug consolidated doc (unrelated)
```

---

## 12. NEXT STEP (When Resuming)

**Phase 0** — Foundation (1-2 hours):
1. Install ClawMem: `cd ~/code && git clone https://github.com/yoloshii/ClawMem.git && cd ClawMem && bun install`
2. `clawmem init --vault ~/knowledge`
3. `clawmem setup hooks` + `clawmem setup mcp`
4. Create `~/knowledge/` directory structure with all sub-indexes
5. Clone joshpocock/karpathy-obsidian-vault → copy `CLAUDE.md` to `~/knowledge/`
6. `git init ~/knowledge`
7. Verify: `clawmem stats`, `clawmem search "test"`

---

## 13. CRITICAL CONTEXT FROM CONVERSATION

### What User Likes
- Karpathy pointer-based memory system ([[wikilinks]], deterministic navigation)
- MoE models for free-tier tool use (speed + provider reliability compounds across agent loops)
- Single encode at extraction time, reuse everywhere (no re-encoding)
- Sub-indexes as the shard (never load full wiki, only relevant sub-index)
- Human-readable first, git-trackable, local-only
- MemVid V2 cold tier (scheduled compute, negligible storage)

### What User Dislikes
- Being taught things they already know (RAG basics, data models)
- Assuming family-level benchmark scores transfer to architectural variants
- Pigionholing on tool-use benchmarks when asking about general intelligence
- Black-box storage (everything must be inspectable)

### User's Document Taxonomy (800 files, ~35MB)
- **PRDs**: 30-50k, ~50 files, structured (goals, architecture, dependencies)
- **YouTube extractions**: <10k, ~300 files, semi-structured (lists, prompts, scripts, mermaid, recipes)
- **Research papers**: up to 50k, ~150 files, structured (abstract, methods, results)
- **Raw LLM transcripts**: up to 200k, ~300 files, unstructured (conversation, tangents, orphaned ideas)

### User's Proxy Setup
- Headroom running on :8787 (token compression proxy to OpenRouter)
- Ultimate Proxy configured on :8082 but NOT running
- RTK (Rust Token Killer) v0.34.0 active via hooks
- Headroom memory/learning DISABLED (no --memory/--learn flags)

---

**END OF SESSION RESUME**
Everything above is saved. System can crash and recover from this file.
