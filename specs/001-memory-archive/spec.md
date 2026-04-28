# Feature: Personal AI Interaction Archive

**Status**: Draft  
**Created**: 2026-04-08  
**Branch**: 001-memory-archive

---

## User Scenarios & Testing

### Scenario 1: Drop a document, get structured knowledge

**As a** developer working with multiple AI agents,  
**I want to** drop any document (PRD, transcript, research paper, YouTube extraction) into a single folder  
**so that** the system automatically classifies it, extracts knowledge elements, enriches them with metadata, and indexes them for retrieval.

**Acceptance**:
- Given a raw LLM transcript dropped into `raw/transcripts/`, when the system processes it, then it produces structured wiki articles in `wiki/decisions/`, `wiki/code/`, `wiki/prompts/`, and `wiki/concepts/` with YAML frontmatter, embeddings, and cross-links.
- Given a PRD dropped into `raw/prds/`, when the system processes it, then it creates project-specific wiki entries with architectural decisions indexed into Graphiti.

### Scenario 2: Find anything I ever told any model about a topic

**As a** developer who has worked with Claude Code, ChatGPT, Gemini, and multiple local agents,  
**I want to** search across all my interactions by topic, project, model, or semantic similarity  
**so that** I can find every time I described a problem, gave an instruction, or arrived at a solution — regardless of which tool I used.

**Acceptance**:
- Given a query like "everything I've told any model about agent slash commands", when the user searches, then the system returns relevant turns from Claude Code sessions, web UI chats, and local agent conversations, ranked by relevance and recency.
- Given a query like "the exact Python pattern I used in data-kiln three months ago", when the user searches, then the system returns the specific code snippet with full context, including the conversation it came from and the project metadata.

### Scenario 3: Recover abandoned work and avoid repeated failures

**As a** developer who frequently switches between projects,  
**I want to** be reminded of past solutions, repeated failure modes, and abandoned work  
**so that** I don't re-explain problems I've already solved or repeat paths that previously failed.

**Acceptance**:
- Given a new session where the user describes a problem similar to one solved before, when the system retrieves context, then it surfaces the prior solution with provenance (which agent, when, what worked).
- Given a conversation pattern that matches a known failure mode (partial fix, context loss, misunderstanding), when the system detects the pattern, then it flags it and suggests the corrective approach that worked previously.

### Scenario 4: Continuous capture across all agents

**As a** developer running multiple AI agents (Claude Code, OpenClaw, Hermes, Qwen, OpenCode),  
**I want** every session automatically captured into the archive  
**so that** no interaction is lost, even if I forget to manually ingest it.

**Acceptance**:
- Given a new Claude Code session, when the session ends, then all user prompts and model responses are captured in the archive within 60 seconds.
- Given a new OpenClaw session, when the session ends, then the session summary and key decisions are captured.
- Given a web UI chat (ChatGPT, Gemini, Claude web), when the user sends a message, then it is captured in real-time via Tampermonkey interception.

### Scenario 5: Multi-tier memory with automated transitions

**As a** power user with a large interaction history,  
**I want** recent sessions in fast searchable memory, medium-term knowledge in a graph for relationship queries, and old sessions in compressed archive  
**so that** retrieval stays fast and relevant regardless of archive size.

**Acceptance**:
- Given a session from today, when queried, then results return in < 50ms from the Hot tier.
- Given a relationship query ("how does my auth pattern relate to the token system?"), when queried, then the Warm tier returns connected entities and edges in < 150ms.
- Given an archival query ("that conversation from 6 months ago about WebSocket debugging"), when queried, then the Cold tier returns compressed results in < 500ms.

---

## Functional Requirements

### FR-001: Document Ingestion

The system shall accept documents from four input categories:
- **PRDs/Specs** (30-50k, structured) → classified, compiled into `wiki/projects/`
- **Knowledge Extractions** (<10k, semi-structured) → compiled into `wiki/research/`
- **Research Papers** (up to 50k, structured) → compiled into `wiki/concepts/`
- **Raw LLM Transcripts** (up to 200k, unstructured) → element extraction pipeline

### FR-002: Element Extraction

For raw transcripts, the system shall extract discrete knowledge elements:
- **Decisions** → `wiki/decisions/` + Graphiti episodes
- **Patterns** → `wiki/concepts/`
- **Code snippets** → `wiki/code/`
- **Working prompts** → `wiki/prompts/`
- **Facts** → Hot tier (indexed)
- **Concepts** → `wiki/concepts/`
- **Noise** → discarded

### FR-003: Metadata Injection

Every extracted element shall receive structured metadata including:
- Title, type, tags, confidence score, importance weight
- Source file, source type, extraction timestamp
- Related concepts (via `[[wikilinks]]`)
- Project, agent, session ID, git branch
- Content type classification

### FR-004: Embedding

Every element shall be embedded at extraction time using a pluggable provider:
- Primary: Qwen3-Embedding-8B (Ryzen desktop)
- Surface fallback: EmbeddingGemma-300M (Q8, local)
- Cloud fallback: Jina v3 API

Embeddings shall be stored alongside the element in three places:
- Markdown frontmatter (base64, git-trackable)
- SQLite vector index (sqlite-vec, fast retrieval)
- Element metadata JSON (provenance chain)

### FR-005: Hybrid Retrieval

The retrieval pipeline shall support four modes:
- **Lexical** (BM25/tsvector) for exact token matching
- **Semantic** (vector similarity) for meaning-based queries
- **Graph traversal** (wikilinks + knowledge edges) for relationship queries
- **Metadata filtering** (project, date, agent, type) for structured queries

Results shall be fused using Reciprocal Rank Fusion, then reranked by a cross-encoder model.

### FR-006: Progressive Disclosure

Search results shall display at three expansion levels:
- **Collapsed**: source icon, project, model, date, first ~80 chars
- **Summary**: full summary, topic labels, intent category
- **Full**: complete raw text, tool calls, files touched, commands, related turns

### FR-007: Intent-Based Routing

The system shall classify query intent and route to relevant memory tiers:
- "Recent/session" → Hot tier (ClawMem)
- "Relationship/pattern" → Warm tier (Graphiti)
- "Historical/archival" → Cold tier (MemVid V2)
- "Factual lookup" → Hot + Warm
- "Ambiguous" → All tiers (fallback)

### FR-008: Continuous Session Capture

The system shall automatically capture sessions from:
- Claude Code (JSONL files via filesystem watcher)
- OpenClaw (ContextEngine plugin)
- Hermes (MemoryProvider plugin)
- Qwen Code (context file + MCP)
- OpenCode (JSON session files + MCP)
- Codex CLI (JSONL parser)
- Web UIs (Tampermonkey: ChatGPT, Gemini, Claude web)

### FR-009: Daily Update Service

A scheduled service shall run daily to:
- Scan all agent session stores for new entries
- Parse user prompts and model response summaries
- Write to the shared memory vault
- Reconcile with batch exports from web providers

### FR-010: Anti Echo-Loop

All injected context shall be tagged with sentinel markers. During ingestion, content between sentinel markers shall be stripped before storage to prevent infinite self-referential growth.

### FR-011: Token Budget

The retrieval pipeline shall enforce a configurable token budget (default 2000 tokens). Results shall be ranked by fused score, then greedily selected until budget is exhausted. Partial results exceeding the budget shall be truncated at sentence boundaries.

### FR-012: Karpathy Wiki Operations

The system shall support three agent-driven wiki operations:
- **Ingest**: Drop source → compile into wiki → update index + log
- **Query**: Read index → follow [[wikilinks]] → synthesize answer with citations
- **Lint**: Scan wiki for contradictions, orphans, stale claims → output report → fix with approval

### FR-013: Sleep-Time Compute

A scheduled overnight process shall:
- Identify low-confidence documents from ClawMem feedback
- Select S-tier reference for content type
- Rewrite toward reference using local LLM
- Accept if distance decreased AND meaning preserved (cosine > 0.80)
- Write improved versions with changelog

### FR-014: Tier Transitions

The system shall manage automated tier transitions:
- Weekly: Hot (ClawMem) → Warm (Graphiti episodes)
- Monthly: Warm (Graphiti) → Cold (MemVid V2 archive)
- Each transition preserves provenance chains and metadata

### FR-015: MCP SQL Access

An MCP server shall expose the database schema to connected LLMs, allowing natural-language queries to be translated into SQL. The server shall publish full schema documentation with column comments so the LLM understands what it is querying.

### FR-016: Session Audit and Project Timeline

The system shall provide a session audit capability that:
- Discovers sessions across all agent storage locations (Claude Code JSONL, OpenClaw JSON, Qwen text logs, OpenCode JSON, Codex JSONL, Hermes state.db)
- Classifies each session by project (inferred from working directory, file paths, content topics), session type (planning, coding, debugging, testing, docs, audit), and extracts key decisions
- Assembles per-project timelines showing all related sessions chronologically with token summaries and duration
- Identifies related sessions (same project within 48-hour window) across different agents
- Provides CLI commands (`aaa-memory sessions --project X`, `aaa-memory timeline --project X --last 7d`, `aaa-memory audit --project X --update`)
- Exposes project timeline queries via MCP tools for other agents
- Embeds session summaries for semantic search ("what did I work on last week?")
- Stores compressed session summaries as elements (chaff/bash spam stripped, key decisions preserved)
- Links related sessions via knowledge_edges of type `related_to`

### FR-017: Adaptive Refinement Checkpoints

During vault ingestion, the system shall perform quality checks at defined milestones:
- **15%**: Analyze first ~120 files → validate extraction quality. If < 70% approval rate: adjust extraction prompt, re-extract batch. Update schema if new element types discovered.
- **30%**: Cross-reference extracted elements → detect duplicate concepts across sources → merge related wiki pages → update [[wikilink]] graph → validate embedding consistency.
- **50%**: Schema evolution check → identify uncaptured metadata fields → detect new topic clusters needing sub-indexes → re-run classification on previously "noise" elements → generate mid-point quality report.
- **75%**: Graph density analysis → identify orphan wiki pages (zero inbound links) → detect isolated concept clusters → suggest new [[wikilink]] connections → validate retrieval against actual query patterns.
- **100%**: Full audit → lint entire wiki → re-embed all elements if embedding model improved → generate project timelines from session data → produce final quality report.

---

## Non-Functional Requirements

### NFR-001: Latency Targets

| Operation | Target |
|-----------|--------|
| UserPromptSubmit hook total | < 300ms |
| ClawMem query (SQLite FTS5 + vector) | < 50ms |
| Graphiti query (Kuzu embedded) | < 150ms |
| MemVid V2 query (per .mv2 file) | < 50ms |
| Intent classification (Nemotron 3 Super) | < 200ms |
| Score fusion + reranking | < 50ms |
| Prefetch cache lookup | < 5ms |

### NFR-002: VRAM Budget (Surface Laptop Studio 2, RTX 4050 6GB)

- EmbeddingGemma-300M (Q8): ~0.4GB — acceptable
- Total with Headroom: < 2.0GB — acceptable
- SOTA stack (zembed-1 + zerank-2): ~10GB — does not fit, defer to Ryzen desktop

### NFR-003: Reliability

- Atomic writes via SQLite transactions (WAL mode)
- Graphiti ingestion failures go to dead letter queue, not dropped
- LLM classifier unavailability degrades to rule-based heuristics within 1 second
- Structured JSON logging with 50MB rotation, 5-file retention

### NFR-004: Concurrency

- ClawMem SQLite uses WAL mode for safe concurrent multi-session reads
- Graphiti writes serialized via file lock, reads are concurrent
- MemVid V2 writes hold exclusive lock during archival

---

## Success Criteria

1. **Retrieval accuracy**: 85% of queries return the exact conversation/element the user was looking for as the top result (measured via user feedback on progressive disclosure results).
2. **Capture coverage**: 95% of user interactions across all CLI agents are captured within 60 seconds of session end (measured via reconciliation against batch exports).
3. **Latency**: 90th percentile retrieval time under 300ms for Hot tier queries (measured via structured query logging).
4. **Element extraction quality**: 80% of extracted elements from raw transcripts are rated useful by the user (measured via review_extractions.py approval rate).
5. **Cross-session recall**: 84% of related conversations are correctly linked (measured via graph edge accuracy against manual annotation of 100 turn pairs).

---

## Assumptions

1. The user's 800-file Obsidian vault contains a mix of PRDs, YouTube extractions, research papers, and raw LLM transcripts (~35MB total).
2. The Ryzen desktop (RTX 3070 8GB) will be used for heavy embedding and re-encoding tasks.
3. The Surface (RTX 4050 6GB) handles lightweight inference with embedded models.
4. Nemotron 3 Super remains available as a free tier on OpenRouter/NVIDIA NIM.
5. The user prefers zero cloud dependency but accepts free-tier APIs as fallback.

---

## Key Entities

| Entity | Purpose |
|--------|---------|
| **Turn** | Atomic unit: a single user prompt or model response with metadata |
| **Element** | Extracted knowledge unit (decision, pattern, code, prompt, fact, concept) |
| **WikiPage** | LLM-compiled wiki article with YAML frontmatter and [[wikilinks]] |
| **WikiIndex** | Master or sub-index pointing to wiki pages (Karpathy pointer table) |
| **GraphEpisode** | Graphiti episode: an entity/relationship with temporal metadata |
| **ColdArchive** | MemVid V2 compressed archive entry with pre-computed embeddings |
| **Skill** | Reusable pattern extracted from interaction history |
| **SlashCommand** | Candidate auto-promoted from high-frequency operation |

---

## [NEEDS CLARIFICATION] Markers

1. **[NEEDS CLARIFICATION: Storage engine]** — The PRD addendum assumes PostgreSQL + pgvector. The aaa-memory spec uses SQLite + sqlite-vec. Decision pending from user on whether to unify on one engine or run both.
2. **[NEEDS CLARIFICATION: MemVid V2 availability]** — Whether MemVid V2 is installed and working, or if a cold tier alternative (compressed markdown + BM25) is needed as fallback.
3. **[NEEDS CLARIFICATION: MVP scope]** — Which single capability should work first: (A) ingest all 800 vault files and make them searchable, or (B) continuous capture from all agents starting today.
