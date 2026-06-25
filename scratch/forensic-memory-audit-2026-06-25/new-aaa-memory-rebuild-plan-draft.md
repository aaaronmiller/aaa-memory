---
date: 2026-06-25 00:00:00 PDT
ver: 0.2.0
author: Codex
model: GPT-5
tags: [audit, rebuild-plan, aaa-memory, wiki-memory, karpathy-wiki, cass, final-draft]
---
# new-aaa-memory Rebuild Plan Draft

## Executive Summary

The workspace does not contain one memory project. It contains a lineage. The original `aaa-memory` intent was a local memory program for capturing, searching, and recovering agent work. The Karpathy wiki branch then turned that into a compounding wiki system with sleep-time compilation, dream-agent refinement, and multiple installation surfaces. `aaa-memory-local-work` captured a separate but related PRD-unification effort. The result is not a clean product. It is a set of overlapping documents, path assumptions, and partial implementations that each preserve part of the intent while obscuring the whole.

The correct final target is not "keep all of it." The correct target is `new-aaa-memory`, a single project that:

1. preserves the original memory-program intent,
2. keeps the useful Karpathy-style wiki mechanics,
3. uses Cass as the raw session-history evidence/search layer,
4. documents the system so a third party can audit and rebuild it without reading the source repos first.

The key decision is how to judge divergence. A divergence is positive only if it improves quality, auditability, provenance, or operational reliability. It is negative if it fragments the product, makes the architecture harder to explain, or creates stale assumptions. That rule drives the rest of the plan.

## Lineage Map

The lineage is easiest to understand as four layers.

### 1. Original `aaa-memory`

This is the starting point. It is a memory program first. Its core behavior is to store useful context, recover prior work, and keep an agent from starting over every session. Its most important documents are the memory-archive spec and plan, plus later audit and session-resume docs that explain how the project was supposed to work.

### 2. Karpathy-style wiki skill/protocol

This is the second branch. It grows from the Karpathy wiki pattern: raw sources in, compiled wiki pages out, cross-links, indexes, linting, and sleep-time improvement. This branch is much more explicit about compilation and provenance. It is also where the dream agent and multi-tier memory architecture become central.

### 3. `wiki-memory`

This is the most complete Karpathy-style implementation in the workspace. It has the richest operational docs, the strongest dream-agent and ClawMem narrative, and the most detailed installation and integration notes. It is also the most path-drifted branch, because many docs still point at stale `skills-USER/karpathy-wiki` targets.

### 4. `aaa-memory-local-work`

This is not the runtime system. It is a separate synthesis and PRD-unification line. It is useful because it shows how docs were merged, normalized, and split into "future" and "discarded" artifacts. It helps explain the documentation style, but it should not be confused with the live memory stack.

## What the Current Code and Docs Actually Do

The current `aaa-memory` repo is already doing more than the original memory-program concept. The README and SETUP docs now treat Cass as part of the install surface. They tell the user to install Cass, run `cass triage --json`, refresh the index with a cron job four times a day, and treat Cass as the raw coding-agent session evidence tier. That is a real architectural change, not a cosmetic one.

At the same time, the repo still carries the original memory-archive spec, planning docs, and agent-integration guidance. That means the repo has become the likely anchor point for the final system, but it has not yet been normalized into a single, clean product definition.

The Karpathy branch and the `wiki-memory` branch add useful machinery:

- a compounding wiki model,
- dream-agent compilation,
- hot/warm/cold memory tiers,
- provenance-rich pages,
- scheduler and idle-time compute,
- cross-agent installation patterns.

Those are useful if they improve the final memory system. They are harmful if they become a separate product family the user never asked for.

## Divergence Analysis

### Positive Divergences

The following changes improve the final output quality.

First, Cass gives the project a concrete raw evidence layer. The repo docs show Cass as a local-first, multi-agent session search and preflight tool with robot-friendly JSON surfaces. That is a meaningful addition because the memory system becomes easier to audit and easier to bootstrap from actual history rather than only curated wiki pages.

Second, the Karpathy-style wiki machinery improves durability. A raw session log is not enough. Turning that log into a curated wiki with cross-links, provenance, and sleep-time improvements makes the stored knowledge more usable.

Third, the later docs improve operational clarity. `wiki-memory` and the unification work both add setup guides, installation flows, and explicit role separation. That matters because the original user complaint was not just about content. It was about the system no longer matching the intended shape.

### Neutral Divergences

Some divergences are mostly shape changes. For example, moving a concept from a note into a formal spec is not intrinsically good or bad. It only matters if the new shape helps a third party understand the system. Likewise, adding more docs can be neutral if those docs collapse ambiguity instead of adding it.

### Negative Divergences

The biggest negative divergence is fragmentation. The system now has multiple quasi-authoritative sources: `aaa-memory`, `wiki-memory`, the older Karpathy vault, and the unification worktree. They all describe overlapping intent, but not from the same center. A third party would not know which repo to trust first.

The second negative divergence is stale path drift. The wiki docs still point at `skills-USER/karpathy-wiki` and other locations that no longer exist. That is not merely annoying; it is evidence that the docs have diverged from the live environment.

The third negative divergence is architectural sprawl. A lot of future-facing ideas were folded into the wiki docs at once: council deliberation, MemVid, S-tier quality loops, multi-agent orchestration, Graphiti-like graph layers, and more. Some of these are worth keeping, but they need prioritization. Otherwise the final plan becomes a museum of aspirations instead of a buildable system.

## How Cass Fits

Cass should be a first-class part of `new-aaa-memory`, but only in one role: the raw session-history and preflight evidence tier.

That means Cass should:

- be installed when `aaa-memory` is installed,
- be kept current by a cron refresh,
- be queried before the system falls back to weaker memory assumptions,
- provide searchable historical evidence across agent sessions,
- feed durable compilation and audit layers.

Cass should not:

- replace the wiki,
- replace durable structured memory,
- become the only memory system,
- be treated as a synonym for the project itself.

That distinction matters. Cass improves the system when it is the current evidence substrate. It degrades the system if it is allowed to absorb the entire identity of the project.

## Proposed Final Target

`new-aaa-memory` should be the final combined system.

### The stack

The final stack should be:

1. `aaa-memory` at the top-level identity and package name,
2. Cass as the raw search/evidence layer,
3. wiki-style compilation as the durable curated layer,
4. a single documentation set that explains how the two layers interact.

### The operating rule

The final system should never force the user to care which internal branch produced a memory item. The user should only care that the system can:

- recover the right historical evidence,
- compile it into durable knowledge,
- explain where it came from,
- keep itself current,
- and be rebuilt from scratch by another engineer.

## Keep / Change / Delete

### Keep

- the original `aaa-memory` memory-program intent,
- the requirement for durable, searchable session history,
- the wiki-style compounding and cross-linking mechanics,
- provenance and auditability,
- local-first operation,
- the Cass integration as raw evidence search,
- the installation and refresh automation.

### Change

- collapse the current doc family into one canonical project narrative,
- replace stale `skills-USER` and broken symlink assumptions with the live paths,
- make the architecture explicit about what is source evidence, what is derived knowledge, and what is archival,
- separate current runtime behavior from future ideas,
- normalize the docs so a third party can follow them without code archaeology.

### Delete

- duplicate source-of-truth claims,
- dead path references,
- unprioritized "future" ideas that have crept into the mainline docs,
- any doc structure that hides the distinction between runtime, evidence, and synthesis.

## Rebuild From Existing Projects

If rebuilding from the current repos, the path should be:

1. Start from `aaa-memory` because it is the active installed source and the correct project name anchor.
2. Pull in only the Karpathy/wiki mechanics that improve the final product.
3. Treat `wiki-memory` as the richest source of operational detail, but mine it selectively.
4. Treat `aaa-memory-local-work` as documentation lineage only.
5. Write one canonical rebuild plan and then revise the repo docs to point at it.

## Rebuild From Scratch

If rebuilding with no existing code, the path should be:

1. Define the user intent in plain language.
2. Define Cass as the evidence search layer.
3. Define a durable memory archive and a wiki compilation layer.
4. Define the file layout, update cadence, and provenance chain.
5. Add the install workflow.
6. Add the cron refresh.
7. Add the rebuild and audit docs.
8. Only then decide whether any dream-agent or improvement loops are needed.

## Audit Conclusion

The project is not broken because the underlying ideas were wrong. It is broken because the ideas were allowed to multiply without a clean resolution. Cass is useful, but only if it is absorbed into a clear final architecture. The final plan must say exactly why each divergence happened, whether it helped or hurt, and what the new canonical shape is.

## Expanded Evidence Record

This section exists so a third party can understand the audit without opening every source file first. It is not a substitute for the source tree, but it is a compressed map of what was read, what it means, and how each branch should influence `new-aaa-memory`.

### Project roots and status

| Root | Role in lineage | Current judgment |
|---|---|---|
| `/home/cheta/code/aaa-memory` | Active installed package and correct final identity anchor | Keep as the root project. Normalize docs and architecture around this tree. |
| `/home/cheta/code/wiki-memory` | Richest Karpathy-style implementation and dream-agent spec source | Mine for mechanics. Do not preserve as separate final identity. |
| `/home/cheta/code/aaa-memory-local-work` | PRD-unification and schema-reference lineage | Keep as historical evidence and method reference. Do not treat as runtime truth. |
| `/home/cheta/git/karpathy-obsidian-vault` | Older, simpler Karpathy wiki vault pattern | Use as baseline for simple raw-to-wiki behavior and wiki operating rules. |
| `/home/cheta/code/agents/skills/karpathy-wiki` | Broken skill symlink | Do not trust as install source. Fix or replace through final project setup. |

### Runtime facts

The installed `aaa-memory` command resolves to the local `aaa-memory` package. That makes `/home/cheta/code/aaa-memory` the live source tree. Cass is installed at `/home/cheta/.local/bin/cass`. The Karpathy wiki skill symlink points to a missing target and the old `skills-USER/karpathy-wiki` path no longer exists. These facts matter because the final plan must privilege runtime reality over stale documentation.

### Documents used as evidence

The primary `aaa-memory` evidence set consists of the memory archive spec and plan, the integration audit, session resume notes, unified PRD notes, agent integration README, and the live README/SETUP diff that adds Cass. These documents preserve the original memory-program intent and show how the current repo is already drifting toward a Cass-aware architecture.

The primary Karpathy/wiki evidence set consists of the older vault README and `CLAUDE.md`, the `wiki-memory` root docs, skill docs, `spec-as-built.md`, master spec, requirements, architecture, completion plan, test strategy, ClawMem integration, MemVid cold storage, Pi integration, and changelog. These documents preserve the wiki compiler and dream-agent mechanics.

The primary `aaa-memory-local-work` evidence set consists of the unified PRD creation spec, plan, tasks, discarded-content log, future-scope log, schema reference, and mem2 bridge plan. These documents show how prior synthesis work distinguished included material, rejected material, and deferred material. That distinction should be copied into the final plan.

## Reconstructed Original Intent

The original user intent is consistent across all branches. The user wanted a memory system that makes agents less amnesic. The system should capture useful work, recover old context, preserve decisions, and help future sessions start from accumulated evidence rather than blank state. It should work across agent tools because the user's work happens across multiple CLIs, repos, and local integrations.

The original plan was not merely "build search." It was a memory program with several properties:

1. It should ingest documents, agent sessions, raw transcripts, PRDs, specs, notes, research, and source artifacts.
2. It should extract durable facts, decisions, patterns, code references, prompts, errors, and open work.
3. It should attach metadata: source, project, agent, session, branch, timestamp, citation, confidence, and related concepts.
4. It should retrieve across lexical search, semantic search, graph relationships, metadata filters, and archive lookup.
5. It should enforce token budgets and progressive disclosure so agent prompts get useful context without transcript dumps.
6. It should avoid echo loops by distinguishing user-confirmed durable memory from generated summaries and raw transcript text.
7. It should expose memory through CLI, MCP, hooks, and agent integrations.
8. It should compile durable human-readable knowledge, not just retain raw logs.

That original intent remains correct. The divergence problem is not that later work added Cass or a wiki. The problem is that later work added several partially overlapping systems without collapsing them back into one coherent architecture.

## What Each Existing Version Does and Does Not Do

### `aaa-memory`

`aaa-memory` is the correct anchor because it is the installed package and closest match to the original memory-program goal. It provides the active Python package, CLI memory tools, MCP surfaces, vault storage, and the current README/SETUP install path. It now explicitly documents Cass as the raw coding-agent session evidence tier, including `cass triage`, `cass index`, robot-mode search, and a six-hour cron refresh.

What it does well:

- anchors the final project identity,
- stores explicit durable memory,
- exposes MCP and CLI integration surfaces,
- preserves the memory-archive spec lineage,
- now documents Cass as part of setup.

What it does not yet do:

- provide one canonical architecture narrative,
- explain how wiki compilation and Cass evidence interact,
- fully reconcile with `wiki-memory`,
- make every retrieval result evidence-backed,
- remove ambiguity between current implementation and proposed future features.

### Karpathy wiki lineage

The Karpathy branch contributes the compounding wiki pattern. The simple version is powerful: immutable raw sources, compiled wiki pages, master indexes, related links, linting, and query results that can be filed back into the wiki. This solves a different problem than Cass. Cass finds the raw session. The wiki turns durable pieces of that raw evidence into readable knowledge.

What it does well:

- defines a human-readable compiled layer,
- makes cross-links and indexes part of the operating model,
- keeps raw sources immutable,
- gives agents a schema for writing maintainable knowledge.

What it does not yet do:

- install cleanly in the current filesystem,
- know about Cass,
- resolve its own broken symlink and stale path assumptions,
- distinguish baseline wiki behavior from later speculative dream-agent expansion.

### `wiki-memory`

`wiki-memory` is the most complete Karpathy-style branch. It contains detailed specs for dream-agent phases, ClawMem integration, MemVid cold storage, S-tier improvement, testing, Pi integration, and multi-agent setup. It is valuable because it translates high-level wiki ideas into operational surfaces.

What it does well:

- describes the dream-agent compile loop,
- defines hot/warm/cold memory roles,
- documents provenance-rich wiki pages,
- proposes a test strategy,
- documents scheduler and multi-agent integration behavior.

What it does not yet do:

- collapse back into `aaa-memory`,
- remove stale `skills-USER` references,
- classify speculative features separately from required ones,
- incorporate Cass as the raw session evidence tier,
- provide a single source of truth.

### `aaa-memory-local-work`

`aaa-memory-local-work` is mostly a documentation-synthesis artifact. It should not become the final product. Its value is methodological: it shows how to merge many source docs, record discarded content, defer future scope, and normalize schema references.

What it does well:

- preserves unification criteria,
- records discarded content and why it was discarded,
- keeps future ideas separate from current requirements,
- provides a detailed metadata schema reference.

What it does not yet do:

- represent the live installed package,
- provide runtime behavior,
- resolve the memory-project lineage by itself.

## Current Changes Since the Original Plan

The most important current change is Cass. The live README and SETUP diffs add Cass as required raw session evidence support. They document installation, initial triage, indexing, a four-times-daily cron refresh, robot-mode search, robot-mode packs, and optional semantic model installation.

This change is positive because it improves historical evidence recovery. It also reduces the need to hand-maintain bespoke parsers for every agent session format. Cass is purpose-built for local coding-agent session search and supports many providers. It gives `aaa-memory` a stronger raw history layer than the original repo had.

The change is incomplete because it currently lives mostly in docs and hook-adjacent thinking. The final architecture must promote Cass into a proper evidence source adapter and trust boundary. Otherwise it remains a useful CLI that is not fully integrated into retrieval, provenance, wiki compilation, or audit output.

## Research Findings and Alternatives

Research was used to verify Cass and adjacent tools as of June 25, 2026.

### Cass / `coding_agent_session_search`

Official repository: https://github.com/Dicklesworthstone/coding_agent_session_search

Finding: Cass is a local TUI and CLI for indexing and searching local coding-agent history across many providers, including Codex, Claude Code, Gemini CLI, OpenCode, Cursor, Aider, Pi-Agent, Copilot surfaces, Hermes, Qwen Code, and others. GitHub shows release `v0.6.17` as latest on June 24, 2026.

Plan effect: Cass should be the raw session evidence tier. `new-aaa-memory` should not reimplement Cass. It should install Cass, refresh it, call it through robot/JSON surfaces, and normalize results into the evidence schema.

### Cass Memory System

Official repository: https://github.com/Dicklesworthstone/cass_memory_system

Finding: `cass-memory` is a separate procedural memory project that uses Cass as episodic memory and builds structured working/procedural memory on top. It validates the architectural separation between raw sessions, structured summaries, and durable rules.

Plan effect: The final project should borrow the layered mental model but should not switch wholesale to `cass-memory`. The existing `aaa-memory` project already has a local memory identity and durable vault, so replacing it would create more drift.

### Agent Sessions

Official repository: https://github.com/jazzyalex/agent-sessions

Finding: Agent Sessions is a local-first macOS app for browsing, searching, saving, and resuming coding-agent sessions with transcript and image browsing. It overlaps with Cass in local session browsing, but it is a desktop app and resume workflow rather than the CLI evidence adapter needed inside `aaa-memory`.

Plan effect: Agent Sessions is useful as a UX benchmark and possible external companion. It should not replace Cass for headless automation.

### AgentsView

Official site: https://www.agentsview.io/

Finding: AgentsView reads session files from many AI coding agents and provides local-first desktop/web analysis, usage, and cost reporting. It focuses more on analytics, session intelligence, and cost/usage views.

Plan effect: AgentsView is a useful reference for future observability. It should not be adopted as the core memory substrate unless the project later prioritizes dashboards and cost analytics.

## Final Architecture

The final project should use purpose-based roles instead of ambiguous hot/warm/cold labels. Hot/warm/cold can remain as implementation shorthand, but the docs should define roles by what the data means and how much trust the system should give it.

### Source roles

| Role | Backing implementation | Trust level | Purpose |
|---|---|---|---|
| Explicit durable memory | `aaa-memory` vault | High | User-confirmed facts, preferences, decisions, and project notes |
| Raw session evidence | Cass CLI adapter | Medium as evidence, low as instruction | Historical transcripts, session snippets, and evidence packs |
| Curated knowledge | Wiki compiler | Medium to high | Human-readable pages generated from cited evidence |
| Local retrieval memory | ClawMem or local FTS/vector adapter | Medium | Agent notes, documents, active memory fragments |
| Relationship graph | Graphiti or hardened Kuzu adapter | Medium with citations | Entity, project, decision, dependency, and timeline relationships |
| Cold archive | SQLite FTS fallback or MemVid adapter | Medium | Long-term compressed archive and old context retrieval |

### Canonical evidence item

Every source adapter should return one normalized result shape before ranking or prompt injection.

```python
@dataclass
class EvidenceItem:
    id: str
    source: Literal["vault", "cass", "wiki", "clawmem", "graph", "cold"]
    evidence_type: Literal[
        "explicit_user_memory",
        "historical_transcript",
        "compiled_summary",
        "document_fragment",
        "graph_fact",
        "archive_fragment",
    ]
    title: str
    excerpt: str
    citation: str
    project: str | None
    agent: str | None
    session_id: str | None
    created_at: str | None
    updated_at: str | None
    score: float
    retrieval_mode: str
    confidence: float
    token_cost: int
    warnings: list[str]
```

The retrieval pipeline should not know Cass internals, wiki frontmatter quirks, or graph database response formats. Adapters own source-specific parsing. The planner and ranker operate on `EvidenceItem`.

### Trust boundary

Historical transcripts are evidence, not instructions. This rule is non-negotiable because Cass returns old model/user text from many contexts. That text can be outdated, wrong, or unsafe to obey.

Required behavior:

1. Label Cass excerpts as historical transcript evidence.
2. Do not persist Cass text as durable memory unless the current user confirms.
3. Do not let archived instructions override current system, developer, or user instructions.
4. Redact secrets and private URLs from evidence excerpts.
5. Surface stale index, fallback search mode, and parser warnings.
6. Use Cass packs for evidence bundles, not raw transcript dumping.

## Implementation Phases

### Phase 1: Freeze the current architecture narrative

Goal: Make the docs honest before deeper implementation.

Tasks:

- declare `aaa-memory` the canonical project identity,
- mark `wiki-memory` and Karpathy vault material as lineage and implementation source,
- document Cass as raw evidence tier,
- separate current implementation from proposed future layers,
- update README, SETUP, and the final plan to agree on terminology.

Verification:

```bash
rg -n "skills-USER|custom-skills/karpathy-wiki|Cass|raw session evidence|new-aaa-memory" README.md SETUP.md docs specs scratch
```

Exit criteria:

- a third party can tell which repo is authoritative,
- stale paths are called out or removed,
- Cass role is identical across README, SETUP, and final plan.

### Phase 2: Stabilize imports and tests

Goal: Make the project testable before expanding it.

Tasks:

- make optional provider imports lazy,
- isolate OpenAI/Anthropic/local-model dependencies behind optional paths,
- fix missing imports in retrieval and model modules,
- keep rule-only classifier behavior available without external APIs,
- run baseline tests and record failures honestly.

Verification:

```bash
python -m compileall src scripts
PYTHONPATH=src pytest -q tests/test_cass_context_hook.py
PYTHONPATH=src pytest -q tests/test_retrieval_pipeline.py
pytest -q
```

Exit criteria:

- full test collection works,
- optional providers do not break unrelated tests,
- existing Cass hook tests still pass.

### Phase 3: Add canonical evidence models

Goal: Stop source-specific drift in retrieval.

Tasks:

- add `EvidenceItem`,
- add `RetrievalPlan`,
- add source adapter interfaces,
- define trust levels,
- define allowed persistence behavior for each evidence type.

Verification:

```bash
PYTHONPATH=src pytest -q tests/test_evidence_models.py tests/test_source_adapters.py
```

Exit criteria:

- every source returns the same evidence schema,
- every evidence item has citation, source type, confidence, token cost, and warnings.

### Phase 4: Promote Cass into a first-class adapter

Goal: Make Cass more than a setup note or prompt hook.

Tasks:

- implement `CassSourceAdapter` using subprocess calls to installed `cass`,
- run `cass triage --json` before searches or on scheduled health checks,
- call `cass search --robot --robot-meta` for result lists,
- call `cass pack --robot` for bounded evidence bundles,
- parse freshness, fallback, warning, citation, agent, project, session, and timestamp fields,
- surface timeouts and unavailable-Cass states as warnings rather than crashes.

Verification:

```bash
cass triage --json
cass search "aaa-memory cass original plan divergence" --robot --robot-meta --fields summary --limit 5 --max-tokens 2000
PYTHONPATH=src pytest -q tests/test_cass_source_adapter.py
```

Exit criteria:

- Cass unavailable, stale, lexical fallback, empty result, and successful result cases are tested,
- Cass evidence is never stored as durable memory by default,
- the cron refresh remains documented and verifiable.

### Phase 5: Unify retrieval planning and fusion

Goal: Replace duplicate retrieval logic with one auditable planner.

Tasks:

- centralize intent classification,
- keep rule-based routing as required fallback,
- add optional LLM refinement later,
- rank evidence via normalized source score, freshness, trust level, RRF, citation quality, and token cost,
- add golden queries that test source selection.

Verification:

```bash
PYTHONPATH=src pytest -q tests/test_retrieval_planner.py tests/test_retrieval_fusion.py
```

Exit criteria:

- query planning is deterministic in tests,
- golden queries choose the expected source mix,
- prompt budget is respected.

### Phase 6: Unify wiki compilation

Goal: Make the durable wiki layer evidence-backed.

Tasks:

- choose one compiler entrypoint,
- require evidence IDs and citations for compiled pages,
- track raw evidence -> extracted element -> compiled page lineage,
- prevent generated summaries from citing only generated summaries,
- downgrade confidence when source evidence conflicts.

Verification:

```bash
PYTHONPATH=src pytest -q tests/test_wiki_compiler.py tests/test_provenance_chain.py
```

Exit criteria:

- every compiled page has source citations,
- stale Cass warnings block automatic promotion to durable claims,
- wiki pages can be traced back to source evidence.

### Phase 7: Add graph and cold archive only after evidence quality stabilizes

Goal: Restore the original relationship and archive ambitions without reintroducing drift.

Tasks:

- benchmark Graphiti against existing Kuzu/local graph behavior,
- feed graph only cited `EvidenceItem` data,
- keep SQLite cold archive as baseline,
- prototype MemVid behind an adapter,
- benchmark latency, disk use, recall, and citation fidelity.

Verification:

```bash
PYTHONPATH=src pytest -q tests/test_graph_source_adapter.py tests/test_cold_source_adapter.py
```

Exit criteria:

- graph facts cite evidence,
- cold archive returns `EvidenceItem`,
- MemVid is adopted or rejected based on benchmark evidence.

### Phase 8: Release gate and third-party audit

Goal: Make the final product buildable and auditable.

Tasks:

- update final docs,
- add acceptance-story checklist,
- document install from scratch,
- document migration from current repos,
- verify Cass cron,
- verify no secrets are printed,
- verify source citations.

Verification:

```bash
cass triage --json
crontab -l | grep "aaa-memory cass refresh"
python3 scripts/mem.py recall "cass integration aaa-memory" --limit 5
rg -n "PENDING|skills-USER|custom-skills/karpathy-wiki" README.md SETUP.md docs specs scratch
```

Exit criteria:

- a third party can rebuild from scratch,
- a third party can migrate from current repos,
- the eight user stories can be run as acceptance tests.

## Migration Path From Current Repos

1. Preserve `/home/cheta/code/aaa-memory` as the root project.
2. Keep the current README/SETUP Cass installation edits.
3. Pull selected `wiki-memory` mechanics into docs and code only after classifying them as required, deferred, or rejected.
4. Move stale Karpathy path references into a migration note or replace them with final paths.
5. Keep `aaa-memory-local-work` as historical reference and link it from an appendix, not main install docs.
6. Create source adapters and evidence models before adding new retrieval features.
7. Retire duplicated retrieval/wiki compilation paths after tests cover the replacement.

## Rebuild From Scratch

The from-scratch build should be simpler than the current lineage:

1. Create a Python package named `aaa_memory`.
2. Add a local durable vault with SQLite FTS for explicit user memory.
3. Add CLI commands: `save`, `recall`, `inject`, `search`, `status`, `audit`.
4. Install Cass and configure a six-hour cron refresh.
5. Add a Cass adapter that returns normalized evidence.
6. Add a wiki compiler that writes cited markdown pages from evidence.
7. Add MCP tools for memory search, timeline, session search, and memory store.
8. Add source adapters for vault, Cass, wiki, local retrieval, graph, and cold archive.
9. Add retrieval planner, source ranking, token packing, and trust-boundary formatting.
10. Add acceptance tests for the eight user stories.
11. Add graph and cold archive only after baseline tests pass.

## Required Documentation Structure

The final project should have one documentation structure:

| File | Purpose |
|---|---|
| `README.md` | User install, architecture overview, and common commands |
| `SETUP.md` | Detailed install including Cass and cron |
| `docs/ARCHITECTURE.md` | Source roles, evidence model, trust boundary, retrieval planner |
| `docs/MIGRATION.md` | How `aaa-memory`, `wiki-memory`, Karpathy wiki, and local-work map into final project |
| `docs/AUDIT.md` | How to verify install, Cass freshness, source citations, and test health |
| `docs/FUTURE.md` | Deferred ideas like Graphiti, MemVid, S-tier improvement, and dashboards |
| `docs/DISCARDED.md` | Explicitly rejected or superseded ideas with rationale |
| `specs/new-aaa-memory-rebuild-plan.html` | Browser-ready plan |

## Acceptance Metrics

| Area | Target |
|---|---|
| Install | `aaa-memory`, Cass, and cron setup are documented and verifiable |
| Test health | Full test collection succeeds without optional dependency failures |
| Cass adapter | Handles unavailable, stale, fallback, empty, and successful Cass results |
| Provenance | 100 percent of injected evidence has source and citation |
| Trust boundary | 0 historical transcript snippets stored as explicit memory without confirmation |
| Golden queries | Expected source mix appears in top results |
| Prompt budget | Default injected context respects configured token budget |
| Docs | Current, deferred, and rejected pieces are clearly separated |

## Third-Party Audit Checklist

A third-party auditor should be able to answer these questions after reading the final docs:

1. Which repo is authoritative?
2. What does `aaa-memory` store directly?
3. What does Cass provide?
4. What does the wiki compiler generate?
5. Which source is trusted most for user preferences?
6. Which source is trusted most for raw historical evidence?
7. How is Cass kept current?
8. How are stale Cass indexes reported?
9. How are historical transcript instructions prevented from becoming live instructions?
10. How does a compiled wiki page cite its source evidence?
11. What features are implemented now?
12. What features are deferred?
13. What ideas were discarded and why?
14. How does a new machine install the system?
15. How does the project migrate from the current repos?
16. Which tests prove the eight user stories?

## Open Decisions

### Repo shape

Opinion: Keep `aaa-memory` as the root repo and package identity. Do not create a permanent `new-aaa-memory` repo unless the user wants a clean break after the plan is implemented.

Rationale: The installed source is already `aaa-memory`, and changing names now would add more fragmentation.

### Cass requirement level

Opinion: Cass should be required for the final intended install, but the core vault should degrade gracefully if Cass is missing.

Rationale: Cass is central to raw session evidence, but durable explicit memory should not be unusable when Cass is temporarily unavailable.

### Graph layer

Opinion: Defer graph adoption until evidence models and Cass adapter are stable.

Rationale: Graphs amplify whatever quality you feed them. Feeding uncited summaries into a graph would make the system harder to audit.

### MemVid

Opinion: Keep as a cold archive candidate and benchmark behind an adapter.

Rationale: The idea fits the original cold tier, but adoption should be evidence-based.

### Wiki automation level

Opinion: Start with cited compilation and linting. Add self-improvement only after provenance is reliable.

Rationale: Improving bad or uncited pages creates false confidence. Provenance must come first.

## Appendix A: User Stories

### Story 1: Recover a recent session

A developer asks the system what happened in a session from earlier today. The system searches Cass first, retrieves the relevant transcript fragments, and returns a concise answer with the exact evidence trail.

### Story 2: Rebuild an old decision

A developer asks why a path or design choice was made last month. The system uses Cass to find the original session, then uses the curated memory layer to summarize the decision and point to the durable note.

### Story 3: Audit a repo drift

An engineer notices that the docs mention a path that no longer exists. The system shows the stale reference, the updated live path, and the doc that needs to be corrected.

### Story 4: Seed a new project from memory

A developer starts a new project and wants a clean restart from prior experience. The system surfaces the relevant patterns and prior architecture notes instead of forcing a blind rebuild.

### Story 5: Verify a claimed improvement

A user asks whether a new change improved the system. The audit layer compares the old and new docs, classifies the divergence as positive or negative, and explains why.

### Story 6: Install on a fresh machine

A user installs the final memory system on a new machine. The installer sets up the core package, Cass, the refresh cron, and the documentation pointers in one pass.

### Story 7: Trace a memory back to its source

A user opens a wiki page and wants the original evidence. The system follows the provenance chain back through the curated layer and Cass search results to the underlying session.

### Story 8: Continue after an interrupted run

An agent runs out of quota mid-audit. Another agent opens the handoff note, sees the current files and next steps, and continues the work without re-reading every source file from scratch.
