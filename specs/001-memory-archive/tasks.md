# Tasks: Personal AI Interaction Archive (001-memory-archive)

Generated: 2026-04-27 (via spec-kit manual init)  
Plan: specs/001-memory-archive/plan.md  
Spec: specs/001-memory-archive/spec.md

**Tech Stack**: Python 3.12+, SQLite + sqlite-vec, ClawMem (hot), Graphiti (warm), MemVid V2 (cold), Qwen3-Embedding-8B, Nemotron 3 Super

---

## Phase 1: Setup

**Goal**: Initialize project structure, install dependencies, configure environment

- [ ] T001 Create project directory structure
  - `~/knowledge/` with subdirectories: `raw/transcripts/`, `raw/prds/`, `wiki/decisions/`, `wiki/code/`, `wiki/prompts/`, `wiki/concepts/`
  - `~/.cache/clawmem/` for SQLite vault
  - `~/knowledge/.git/` — initialize repository
- [ ] T002 Install ClawMem (Bun package)
  - `bun install -g clawmem` or `bun add clawmem` in project
  - Run `clawmem init` to create `~/.cache/clawmem/index.sqlite`
  - Run `clawmem setup hooks` to install Claude Code lifecycle hooks
  - Run `clawmem setup mcp` to expose MCP tools
- [ ] T003 Install Karpathy wiki schema
  - Clone joshpocock/karpathy-obsidian-vault as `~/knowledge/CLAUDE.md`
  - Copy index files to `~/knowledge/index/`
- [ ] T004 Initialize git in `~/knowledge/`
  - `git init`, add `.gitignore` for `raw/` (large files), commit initial structure
- [ ] T005 Write Tampermonkey scripts for web UIs
  - Create `scripts/tampermonkey-chatgpt.js`, `scripts/tampermonkey-gemini.js`, `scripts/tampermonkey-claude-web.js`
  - Hook into `fetch`/`XMLHttpRequest` to capture user prompts and model responses
  - Write to `~/knowledge/raw/web/<platform>/<timestamp>.jsonl`
- [ ] T006 Write daily update service skeleton
  - Create `scripts/daily-update.sh` (cron at 2 AM)
  - Scanner: walk `~/knowledge/raw/` for new files
  - Parser: extract turns from each format (Claude Code JSONL, OpenClaw JSON, etc.)
  - Writer: append to `turns` table in ClawMem vault
  - Logging: structured JSON to `~/logs/memory-update.log`

** dependencies **: T001 → T003 → T004 → T005/T006 in parallel

---

## Phase 2: Ingestion Pipeline

**Goal**: Classify, extract, embed, and index document batches

### User Story 2.1: Document Classifier

**As a** developer with mixed file types in my vault  
**I want** the system to automatically recognize PRDs, transcripts, research papers, and knowledge extracts  
**so that** each file is routed to the correct extractor pipeline

**Acceptance**:
- 10 test files across categories → classifier accuracy ≥ 90%
- Classification result stored in `metadata.classification.type`

**Implementation Tasks**:
- [ ] T021 Build rule-based classifier (file path heuristics, size, extension, YAML frontmatter tags)
  - Create `src/classifier/rules.py` with patterns for each category
  - Regex for PRD keywords ("architecture", "requirements", "spec")
  - Transcript detection: presence of ` Human: ` / ` Assistant: ` exchanges
  - Research paper detection: abstract section, citations, references
- [ ] T022 Create training dataset from 100 labeled vault files
  - Label `raw/` samples into 4 categories
  - Store labels in `classification_report.json`
- [ ] T023 Evaluate classifier on test set, tune thresholds to reach 90%+ accuracy
  - Generate confusion matrix, adjust rules
- [ ] T024 Integrate LLM fallback (Nemotron 3 Super) for ambiguous cases
  - Prompt: "Classify this document into: PRD, transcript, research_paper, knowledge_extract"
  - Cache LLM results to avoid repeat calls
- [ ] T025 Write classifier unit tests (pytest, 20+ test cases across edge cases)

**Dependencies**: T021 → T022 → T023/T024 in parallel → T025  
**Parallel**: T024 and T023 can run concurrently once rule base stable

---

### User Story 2.2: Element Extractor

**As a** developer with raw LLM transcripts  
**I want** the system to extract structured knowledge elements (decisions, patterns, code, prompts, facts, concepts)  
**so that** unstructured conversations become searchable, linkable knowledge

**Acceptance**:
- Extract from 10 transcripts → ≥ 80% of extracted elements rated useful in human review
- Each element has: `type`, `title`, `content`, `confidence`, `metadata`

**Implementation Tasks**:
- [ ] T026 Build LLM-based extractor with structured output
  - Create `src/extractor/llm_extractor.py`
  - Prompt template: "Extract all knowledge elements from this transcript..."
  - Pydantic schema for Element (type, title, content, tags, confidence)
  - Nemotron 3 Super via OpenRouter (free tier)
- [ ] T027 Build rule-based fallback extractor (regex for code blocks, bullet lists, "Decision:" markers)
  - `src/extractor/rules_extractor.py`
  - Triggered if LLM unavailable or confidence < 0.6
- [ ] T028 Metadata injector: add YAML frontmatter to each extracted element
  - `src/metadata/injector.py`
  - Fields: title, type, tags, source_file, extraction_ts, project, agent, session_id
  - Generate `[[wikilinks]]` from mentioned concepts
- [ ] T029 Embedding module: compute and store embeddings at extraction time
  - `src/embedding/encoder.py`
  - Priority chain: Qwen3-Embedding-8B (Ryzen) → EmbeddingGemma-300M (Surface) → Jina v3 API
  - Store in: (1) markdown frontmatter (base64), (2) SQLite vec table, (3) element metadata JSON
- [ ] T030 Test pipeline on 10 transcripts → manual review extraction quality
  - Run `review_extractions.py` interactive approval UI
  - Target: 80%+ useful elements approval rate
- [ ] T031 Write extraction tests (fixtures with expected elements, CI validation)

**Dependencies**: T026 → T027 → T028 → T029 → T030/T031 parallel  
**Parallel**: T030 (manual review) and T031 (test writing) can start after T029

---

### User Story 2.3: Wiki Compiler

**As a** developer  
**I want** extracted elements to be compiled into structured wiki articles with Karpathy-style pointer indexes  
**so that** knowledge is navigable via `[[wikilinks]]` and hierarchical index pages

**Acceptance**:
- Extracted elements auto-compile into appropriate `wiki/<type>/` markdown files
- Index pages point to articles via `[[wikilink]]` syntax
- All wiki files have proper YAML frontmatter

**Implementation Tasks**:
- [ ] T032 Implement wiki compiler service
  - `src/wiki/compiler.py`
  - Maps element types to wiki directories: decisions→`wiki/decisions/`, code→`wiki/code/`, prompts→`wiki/prompts/`, concepts→`wiki/concepts/`
  - Generates markdown with YAML frontmatter and `[[wikilinks]]` for related concepts
- [ ] T033 Build Karpathy pointer-based index system
  - `src/wiki/indexer.py`
  - Master index (`wiki/index.md`) lists sub-indexes per category
  - Sub-indexes (e.g., `wiki/decisions/index.md`) use pointer tables: `[[decision-001]], [[decision-002]]`
- [ ] T034 Create wiki lint tool
  - `src/wiki/linter.py`
  - Detects: orphans (no inbound links), dead links (pointing to missing pages), stale claims (outdated info), contradictions
  - Output: `wiki_lint_report.md`

**Dependencies**: T032 (after T028 complete) → T033 → T034

---

## Phase 3: 800-File Vault Migration

**Goal**: Process entire Obsidian vault with refinement checkpoints

### User Story 3.1: Batch Classification & Ingestion

**As a** developer with 800-file vault  
**I want** the entire archive processed in batches with quality gates  
**so that** I can monitor progress and refine the pipeline mid-flight

**Acceptance**:
- All 800 files classified, extracted, indexed
- Refinement checkpoints at 15%, 30%, 50%, 75%, 100%
- Quality reports at each checkpoint guide adjustments

**Implementation Tasks**:
- [ ] T035 Run classifier on entire vault → `classification_report.json` with counts per category
  - Script: `scripts/vault_classify.py`
  - Output: `~/knowledge/classification_report.json`
- [ ] T036 Route files into `raw/<type>/` subdirectories based on classification
- [ ] T037 Batch element extraction on all Tier 5 transcripts (largest category)
  - Process 20 files per batch, commit results after each batch
  - Overnight run: estimate ~40 hours for 800 files at 20/batch → parallelize across batches? (TBD)
- [ ] T038 Human review interface: `scripts/review_extractions.py` (interactive CLI)
  - Approve/reject/edit each extracted element
  - Store decisions in `extraction_reviews.jsonl`
- [ ] T039 Index all `wiki/` files into ClawMem (FTS5 + vector)
  - `clawmem index wiki/ --recursive`
- [ ] T040 Ingest PRDs into Graphiti as episodes
  - `src/graphiti/ingest.py` — creates GraphEpisode entities with temporal metadata
- [ ] T041 Archive raw transcripts to cold storage (compressed .mv2 or SQLite FTS5)
  - `scripts/archive_cold.py` — moves processed raw files to `cold/` with metadata
- [ ] T042 Checkpoint 15% (~120 files): validate extraction quality, adjust prompt if < 70% approval
- [ ] T043 Checkpoint 30% (~240 files): cross-reference elements, merge duplicate concepts, update wikilink graph, validate embedding consistency
- [ ] T044 Checkpoint 50% (~400 files): schema evolution check, identify new metadata fields, detect new topic clusters, re-run classifier on previously "noise" elements
- [ ] T045 Checkpoint 75% (~600 files): graph density analysis, orphan detection, suggest new wikilinks, validate retrieval against query patterns
- [ ] T046 Checkpoint 100% (all 800 files): full lint, re-embed all elements if model improved, generate timelines, final quality report

**Dependencies**: T035 → T036 → T037 → T038 → (T039, T040, T041 in parallel) → T042 → T043 → T044 → T045 → T046

---

## Phase 4: Session Audit

**Goal**: Discover, classify, and timeline sessions across all agents

### User Story 4.1: Cross-Agent Session Discovery

**As a** developer using multiple AI agents (Claude Code, OpenClaw, Hermes, Qwen, OpenCode, Codex, web UIs)  
**I want** all my past sessions automatically discovered and linked  
**so that** I can see what I worked on, when, and in which tool

**Acceptance**:
- Sessions found in all 7 storage locations (Claude Code JSONL, OpenClaw JSON, Hermes state.db, Qwen context files, OpenCode JSON, Codex JSONL, Tampermonkey web captures)
- Each session tagged with: project_id (inferred), session_type (planning/coding/debugging/testing/docs/audit), key_decisions extracted
- Per-project timeline view shows related sessions chronologically

**Implementation Tasks**:
- [ ] T047 Discovery scanner: walk all known agent storage locations
  - `src/audit/discover.py`
  - Paths: `~/.claude/sessions/`, `~/.openclaw/sessions/`, `~/.hermes/state.db`, `~/qwen/context/`, `~/.opencode/sessions/`, `~/codex/rollouts/`, `~/knowledge/raw/web/`
  - Identifies session files by pattern (JSONL, JSON, SQLite tables)
- [ ] T048 Session parser: extract turns from each format
  - `src/audit/parser.py` — format-specific loaders for each agent
  - Normalizes to common `Turn` model (user prompt, model response, timestamp, agent, project context)
- [ ] T049 Session classifier: infer project_id from cwd/file paths + content topics
  - `src/audit/classify.py`
  - Uses file path heuristics + LLM topic classification on first 5 turns
  - Assigns session_type via keyword classification (test keywords → "testing", build keywords → "coding", etc.)
- [ ] T050 Key decision extractor: summarize decisions per session
  - `src/audit/extract_decisions.py`
  - LLM prompt: "What key decisions were made in this session? List as bullet points."
- [ ] T051 Project timeline assembler: chronological view with linked sessions
  - `src/audit/timeline.py`
  - Groups sessions by project, sorted by start_time
  - Links sessions within 48h window across different agents
  - Generates markdown timeline with session summaries and turn counts
- [ ] T052 CLI commands: `aaa-memory sessions`, `aaa-memory timeline --project X --last 7d`, `aaa-memory audit --update`
  - `src/cli/sessions.py`, `src/cli/timeline.py`, `src/cli/audit.py`
- [ ] T053 MCP tool: `memory_sessions(project_id)`, `memory_timeline(project_id, days)`
  - Expose via ClawMem MCP server
- [ ] T054 Embed session summaries for semantic search
  - Store compressed summary (chaff/bash spam stripped, key decisions preserved) as `Element` with `kind="session_summary"`

**Dependencies**: T047 → T048 → T049/T050 in parallel → T051 → T052/T053/T054

---

## Phase 5: Memory Router

**Goal**: Intent-aware retrieval across hot/warm/cold tiers

### User Story 5.1: Query Understanding & Tier Routing

**As a** developer searching for information  
**I want** queries automatically routed to the appropriate memory tier (hot/warm/cold/all)  
**so that** results are fast and relevant without manual tier selection

**Acceptance**:
- Intent classifier correctly routes "recent" queries to hot, "relationship" to warm, "archival" to cold
- Ambiguous queries search all tiers and fuse results
- Latency targets met (NFR-001)

**Implementation Tasks**:
- [ ] T055 Intent classifier service
  - `src/router/intent.py`
  - Nemotron 3 Super via OpenRouter (free tier) with structured output: `{intent: "recent|relationship|archival|factual|ambiguous"}`
  - Rule-based fallback: keyword matching ("recent", "last week" → recent; "how does X relate" → relationship; "6 months ago" → archival)
- [ ] T056 Three retrieval strategy implementations
  - Hot tier: ClawMem FTS5 + sqlite-vec hybrid (RRF) — `src/retrieval/hot.py`
  - Warm tier: Graphiti graph traversal — `src/retrieval/warm.py`
  - Cold tier: MemVid V2 compressed search — `src/retrieval/cold.py`
- [ ] T057 Score fusion engine (Reciprocal Rank Fusion)
  - `src/retrieval/fusion.py`
  - Configurable weights per tier
- [ ] T058 Cross-encoder reranker (qwen3-reranker-0.6B on Ryzen, CPU cosine fallback)
  - `src/retrieval/rerank.py`
  - Re-scores top-50 fused results
- [ ] T059 Token budget enforcement (default 2000 tokens)
  - `src/retrieval/budget.py`
  - Rank-ordered greedy selection, truncate at sentence boundaries
- [ ] T060 Echo-loop prevention (sentinel marker stripping on ingest, duplicate detection on retrieval)
  - `src/retrieval/echo.py`

**Dependencies**: T055 → T056 → T057 → T058 → T059/T060 parallel

---

### User Story 5.2: Progressive Disclosure UI

**As a** developer reviewing search results  
**I want** three expansion levels (collapsed, summary, full)  
**so that** I can quickly scan results and drill into details only when needed

**Acceptance**:
- Collapsed: icon, project, model, date, first 80 chars, ~300 tokens
- Summary: full summary, topic labels, intent category, failure mode indicators
- Full: raw text, tool calls, files touched, commands, related turns, [[wikilinks]]

**Implementation Tasks**:
- [ ] T061 Result formatter service
  - `src/ui/formatter.py`
  - `format_collapsed(result) -> embed dict`
  - `format_summary(result) -> embed dict`
  - `format_full(result) -> markdown thread`
- [ ] T062 Web UI component (Svelte) for expandable results — Phase 7 only, defer to later

**Dependencies**: T061 (T062 deferred to Phase 7)

---

## Phase 6: Agent Integration

**Goal**: Connect all major agent runtimes to aaa-memory

### User Story 6.1: Claude Code Integration

**As a** Claude Code user  
**I want** my sessions automatically captured  
**so that** I can search my Claude Code history later

**Implementation Tasks**:
- [ ] T063 Install ClawMem hooks: `clawmem setup hooks` registers `UserPromptSubmit`, `SessionStart`, `Stop`, `PreCompact` lifecycle listeners
  - Hooks write turns to `turns` table in real-time
- [ ] T064 MCP tool registration: `clawmem setup mcp` exposes 31+ tools (search, storage, context injection)
- [ ] T065 Validate capture: run test session, verify entries appear in `turns` within 60s

**Dependencies**: ClawMem already installed (Phase 1) → T063/T064 → T065

---

### User Story 6.2: OpenClaw Integration

**As an** OpenClaw user  
**I want** OpenClaw to use aaa-memory as its memory backend  
**so that** Discord/Telegram conversations are automatically archived and searchable

**Implementation Tasks**:
- [ ] T066 Write OpenClaw ContextEngine plugin
  - Plugin: `openclaw-plugin-aaa-memory`
  - Hooks: `before_prompt_build` (inject retrieved context), `afterTurn` (store turn), `compact` (summarize old context)
  - Configure via `openclaw.json` → `plugins: ["aaa-memory"]`
- [ ] T067 Replace OpenClaw cold storage with shared SQLite vault
  - Patch OpenClaw's `src/memory/cold-storage/` to read from `turns`, `elements`, `wiki_pages` tables
  - Maintain backward compatibility via view aliases if needed
- [ ] T068 Test: Discord message → stored in vault → `!memory search` finds it

**Dependencies**: T066 → T067 → T068

---

### User Story 6.3: Hermes Integration

**As a** Hermes user  
**I want** Hermes to persist sessions to aaa-memory  
**so that** my agent conversations are not lost to compaction

**Implementation Tasks**:
- [ ] T069 Recommend hermes-lcm plugin (lossless context management) — documentation only
- [ ] T070 Write Hermes MemoryProvider ABC plugin
  - Implements `search(query) -> [results]`, `store(turn)`, `health_check()`
  - `src/hermes/provider.py`
- [ ] T071 Test Hermes → aaa-memory pipeline

**Dependencies**: T069 (docs) → T070 → T071

---

### User Story 6.4: Qwen Code, OpenCode, Codex CLI Integration

**As a** user of non-Claude agents  
**I want** my Qwen/OpenCode/Codex sessions captured  
**so that** the archive is complete across all tools

**Implementation Tasks**:
- [ ] T072 Qwen Code: context file injection + MCP tool registration
  - `src/qwen/context.py` — writes `PROJECT_SUMMARY.md` refreshes
  - MCP: `memory_store`, `memory_search`
- [ ] T073 OpenCode: JSON session parser (`ses_*.json` files) + MCP
  - `src/opencode/parser.py`
- [ ] T074 Codex CLI: JSONL rollout parser
  - `src/codex/parser.py`

**Dependencies**: T072/T073/T074 independent, can run in parallel

---

## Phase 7: Tier Transitions + Sleep-Time Compute

**Goal**: Automated tier transitions and overnight improvement

### User Story 7.1: Tier Transition Daemons

**As a** power user with large memory  
**I want** old data automatically moved to appropriate tiers (hot→warm→cold)  
**so that** recent stays fast, old gets compressed

**Implementation Tasks**:
- [ ] T075 Weekly Hot→Warm transition daemon
  - `src/transitions/hot_to_warm.py`
  - Cron: `0 2 * * 0` (Sunday 2 AM)
  - Move turns >7 days old from `turns` (hot) to Graphiti episodes (warm)
  - Preserve provenance: keep `source_turn_id` references
- [ ] T076 Monthly Warm→Cold transition daemon
  - `src/transitions/warm_to_cold.py`
  - Cron: `0 3 1 * *` (1st of month, 3 AM)
  - Archive Graphiti episodes >90 days to MemVid V2 .mv2 files
  - Delete from warm tier after successful archival
- [ ] T077 Overnight improvement loop (re-encode low-confidence elements)
  - `src/overnight/improve.py`
  - Identify elements with confidence < 0.7 OR user correction history
  - Select S-tier reference for content type
  - Rewrite with local LLM (cost-free)
  - Accept if cosine similarity > 0.80 AND meaning preserved (LLM judge)
  - Write improved version with changelog entry
- [ ] T078 Post-transition reporting: Discord/status channel notification with stats
  - `src/reporting/transition_report.py`

**Dependencies**: T075 → T076 (monthly after weekly stable) → T077 (independent, can run nightly) → T078 (after both)

---

### User Story 7.2: Lint → Fix Pipeline

**As a** maintainer of the knowledge base  
**I want** wiki lint issues automatically fixed where safe  
**so that** the wiki stays healthy with minimal manual work

**Implementation Tasks**:
- [ ] T079 Wiki lint runner (already T034) scheduled daily at 3 AM
- [ ] T080 Auto-fix agent: reads `wiki_lint_report.md`, proposes fixes via Forge plan
  - `src/wiki/fixer.py` — safe fixes only: add missing wikilinks, update stale claims with LLM (requires approval)
  - Creates GitHub issues for complex fixes
- [ ] T081 Approval gate: require human confirmation before merge of auto-fixes

**Dependencies**: T079 → T080 → T081

---

## Final Phase: Polish & Cross-Cutting Concerns

**Goal**: Quality gates, testing, documentation, monitoring

- [ ] T082 End-to-end integration tests (10 end-to-end scenarios spanning ingestion → retrieval)
  - `tests/e2e/` — pytest with fixtures for each tier
- [ ] T083 Performance benchmarks (latency targets from NFR-001)
  - `tests/benchmarks/` — measure hot/warm/cold query times, track over time
- [ ] T084 Structured logging throughout (JSON, 50MB rotation, 5-file retention)
  - `src/logging/configure.py`
- [ ] T085 Error handling: dead letter queue for Graphiti failures, graceful degradation for embedding unavailability
- [ ] T086 Documentation: README.md with setup guide, architecture diagram, troubleshooting
- [ ] T087 CHANGELOG.md and versioning (semver)
- [ ] T088 Security audit: verify no secret leakage in stored turns, validate Tampermonkey script permissions
- [ ] T089 Accessibility review: ensure web UI (Phase 7) meets WCAG 2.1 AA (deferred to Phase 7)

**Dependencies**: T082/T083 after core retrieval complete, T084 throughout, T085 before release, T086/T087 final, T088 security review before production, T089 deferred

---

## Dependencies Summary (Execution Order)

```
Phase 1 (Setup): T001 → T003 → T004 → (T005, T006 in parallel)

Phase 2 (Ingestion): 
  Classifier: T021 → T022 → T023 → (T024 parallel with T023) → T025
  Extractor: T026 → T027 → T028 → T029 → (T030 parallel with T031) 
  Wiki: T032 (after T028) → T033 → T034

Phase 3 (Vault Migration): 
  T035 → T036 → T037 → T038 → (T039, T040, T041 parallel) → checkpoints (T042 → T043 → T044 → T045 → T046)

Phase 4 (Session Audit): T047 → T048 → (T049, T050 parallel) → T051 → (T052, T053, T054 parallel)

Phase 5 (Memory Router): T055 → T056 → T057 → T058 → (T059, T060 parallel) → T061

Phase 6 (Agent Integration - independent): 
  Claude: T063 → T064 → T065
  OpenClaw: T066 → T067 → T068
  Hermes: T070 → T071  (T069 docs separate)
  Others: T072, T073, T074 (independent)

Phase 7 (Transitions): T075 → T076 → (T077 independent) → T078

Polish: T082/T083 (after core), T084 (throughout), T085 (before release), T086/T087 (final), T088 (pre-prod), T089 (deferred)
```

---

## Parallel Execution Opportunities

Marked `[P]` in task descriptions above. Key parallelization points:

| Phase | Parallel Tasks | Reason |
|-------|----------------|--------|
| Phase 1 | T005 (Tampermonkey) and T006 (daily service) | Independent — different scripts, no shared state |
| Phase 2 | T024 (LLM fallback) parallel with T023 (rule tuning) | Once rules stable, LLM can run alongside final tuning |
| Phase 2 | T030 (manual review) and T031 (test writing) | Different resources — human reviews while coder writes tests |
| Phase 2 | T032 (wiki compiler) can start after T028, before T029 finishes | Compiler logic independent of embedding storage |
| Phase 3 | T039, T040, T041 (indexing, Graphiti ingest, cold archive) all parallel after batch extraction | Different backends, no dependencies |
| Phase 4 | T049 (classifier) and T050 (decision extractor) both consume parsed turns | Independent post-parse |
| Phase 4 | T052, T053, T054 (CLI, MCP, embedding) parallel after timeline built | Different output formats |
| Phase 5 | T059 (budget) and T060 (echo prevention) independent after fusion/rerank done | Separate refinements |
| Phase 6 | Claude/OpenClaw/Hermes/Qwen tasks all independent | Different agent systems |
| Phase 7 | T077 (overnight improvement) independent of daemon scheduling | Can run nightly regardless of weekly/monthly transitions |

**Estimated total tasks**: 89 checklist items (T001–T089) across all phases.

---

## Implementation Strategy (MVP First)

**Minimum Viable Product** (Phases 1–2 + Phase 5 minimal):
- Install ClawMem, basic classifier, rule-based extractor, wiki compiler, simple keyword search (no graph, no cold tier)
- Deliverable: Drop a transcript → get wiki article → searchable by keyword within 24h

**Incremental delivery**:
1. Sprint 1 (Week 1): Phases 1–2 complete → vault ingest pipeline works on 10 files
2. Sprint 2 (Week 2): Phase 3 pilot on 120 files (15% checkpoint) → validate at scale
3. Sprint 3 (Week 3): Phase 4 session audit → timeline view
4. Sprint 4 (Week 4): Phase 5 memory router → hybrid search
5. Sprint 5 (Week 5-6): Phase 6 agent integrations → Discord/Claude Code capture
6. Sprint 6 (Week 7): Phase 7 transitions → automated tiering

---

## Tests: Optional by Default

Per spec-kit, tests are optional unless explicitly requested in constitution. No explicit test requirement in aaa-memory constitution. Tasks include test writing as development practice but not full TDD mandate.

---

**Task Count**: 89 tasks (T001–T089) across 7 phases, 21 user stories.  
**Parallel opportunities**: 10 phases with concurrent workstreams.  
**Estimated effort**: ~6–7 weeks (with refinement checkpoint reviews).
