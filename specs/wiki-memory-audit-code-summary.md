# Wiki Memory — As-Built Audit Summary

Generated: 2026-06-08
Source: /home/cheta/code/wiki-memory (git repo, ~3500 lines across all source files)

---

## 1. Entrypoints

### Primary

| File | Role | CLI |
|------|------|-----|
| `dream/dream_agent.py` (1087 lines) | Dream agent — 6-phase background wiki compiler | `python3 dream/dream_agent.py --idle 600 [--quiet]` |
| `dream/scheduler.py` (206 lines) | Scheduler daemon / one-shot runner | `python3 dream/scheduler.py [--cycle N] [--daemon] [--idle-check]` |
| `setup.sh` (1236 lines) | Multi-agent installer (detects agents, symlinks, hooks, MCP, systemd) | `./setup.sh [--preset NAME] [--skip-*]` |
| `install.sh` (31 lines) | Thin wrapper — delegates all args to `setup.sh` | `./install.sh [...]` |

### Secondary

| File | Role | Lines |
|------|------|-------|
| `cli/install.sh` | Universal per-agent installer (detects commands, runs per-agent scripts) | 45 |
| `cli/install-claude-code.sh` | Symlinks plugin.json + skill + hooks + dream into `~/.claude/plugins/` | 10 |
| `cli/install-pi.sh` | Copies TS extension, writes root-marker JSON, symlinks skill | 29 |
| `cli/install-hermes.sh` | Symlinks SKILL.md into `~/.config/hermes/skills/` | 12 |
| `cli/install-codex.sh` | Adds `dream-wiki` alias to shell rc, symlinks data dir | 29 |
| `cli/install-opencode.sh` | Symlinks hook scripts into `.opencode/hooks/` | 18 |
| `cli/install-kilocode.sh` | Writes `.kilocode/config` with hook + dream paths | 21 |
| `cli/install-antigravity.sh` | Writes `.antigravity/config.json` pointing at plugin.json | 22 |

---

## 2. Environment Variables

| Variable | Default | Used In | Purpose |
|----------|---------|---------|---------|
| `AI_WIKI` | `~/ai-wiki` → `~/.local/share/ai-wiki` | dream_agent.py, hooks/*.py, Pi extension | Canonical wiki data root |
| `CLAWMEM_URL` | `http://localhost:7438` | dream_agent.py, setup.sh | ClawMem REST API endpoint |
| `CLAWMEM_COLLECTION` | `wiki` | dream_agent.py | Collection name for wiki content |
| `CLAWMEM_SOURCE_DIR` | `~/git/ClawMem` | setup.sh | Where ClawMem is cloned |
| `CLAWMEM_BINARY` | `$CLAWMEM_SOURCE_DIR/bin/clawmem` | setup.sh | Path to ClawMem binary |
| `PI_SKILLS_DIR` | `~/.pi/agent/skills` | dream_agent.py (`_create_skill_from_pattern`) | Where auto-skills are written |
| `ANTE_SKILLS_DIR` | `~/.config/ante/skills` | dream_agent.py (defined but unused) | Ante skills target |
| `WIKI_MEMORY_ROOT` | `~/code/wiki-memory` | Pi extension (wiki-memory-hooks.ts) | Project root resolution |
| `SKILL_CREATION_THRESHOLD` | `3` | dream_agent.py | Pattern count before auto-skill |
| `CONFIDENCE_AUTO` | `0.8` | dream_agent.py | Auto-compile threshold |
| `CONFIDENCE_FLAG` | `0.5` | dream_agent.py | Flag-for-review threshold |
| `CONFIDENCE_REJECT` | `0.5` | dream_agent.py | Rejection threshold (same as FLAG) |
| `COUNCIL_LOW` / `COUNCIL_HIGH` | `0.5` / `0.6` | dream_agent.py | Council escalation range |
| `IMPROVEMENT_BUDGET_RATIO` | `0.33` | dream_agent.py (defined, used as 0.50 in code) | **Doc/code mismatch — code hardcodes 0.50** |
| `MAX_ITERATIONS_PER_DOC` | `10` | dream_agent.py (defined, unused) | Not wired to any loop |
| `MEANING_PRESERVATION_THRESHOLD` | `0.80` | dream_agent.py (defined, unused) | Not implemented |
| `CONVERGENCE_THRESHOLD` | `0.01` | dream_agent.py (defined, unused) | Not implemented |
| `DAEMON_INTERVAL` | `1800` | scheduler.py | Daemon loop interval (s) |
| `WALL_CLOCK_FALLBACK` | `14400` | scheduler.py | Max time without idle-triggered run (s) |
| `DEFAULT_IDLE` | `600` | scheduler.py | Default idle budget for dream cycle (s) |
| `MEMVID_SOURCE_DIR` | `~/git/memvid` | setup.sh (defined, never used) | MemVid repo path |

---

## 3. Default Paths

| Path | Resolved From | Purpose |
|------|---------------|---------|
| `~/.local/share/ai-wiki/` | `$AI_WIKI` default | Canonical wiki data root |
| `~/ai-wiki/` | Symlink → above | Short alias |
| `~/ai-wiki/raw/` | `$AI_WIKI/raw` | Immutable source intake |
| `~/ai-wiki/pages/` | `$AI_WIKI/pages` | Compiled wiki articles |
| `~/ai-wiki/pages/concepts/` | `$AI_WIKI/pages/concepts` | Concept articles |
| `~/ai-wiki/pages/entities/` | `$AI_WIKI/pages/entities` | Entity pages (people, orgs, tools) |
| `~/ai-wiki/pages/sources/` | `$AI_WIKI/pages/sources` | Source summaries |
| `~/ai-wiki/pages/queries/` | `$AI_WIKI/pages/queries` | Filed QA pairs |
| `~/ai-wiki/.meta/` | `$AI_WIKI/.meta` | Runtime state |
| `~/ai-wiki/.meta/skills/` | `$AI_WIKI/.meta/skills` | Auto-generated skill refs |
| `~/ai-wiki/.meta/skill_patterns.json` | Hardcoded | Pattern tracking DB |
| `~/ai-wiki/.meta/intake_log.jsonl` | Hardcoded | Intake processing log |
| `~/ai-wiki/.meta/scheduler_state.json` | scheduler.py | Last-run timestamps |
| `~/ai-wiki/pages/index.md` | Hardcoded | Wiki master index |
| `~/ai-wiki/pages/log.md` | Hardcoded | Append-only action log |
| `~/git/ClawMem/` | `$CLAWMEM_SOURCE_DIR` | ClawMem source repo |
| `~/.pi/agent/extensions/wiki-memory-hooks.ts` | Copied by setup.sh | Pi lifecycle extension |
| `~/.pi/agent/extensions/wiki-memory-root.json` | Generated by setup.sh | Project root marker for Pi |

---

## 4. Hook Behavior

### `hooks/pre_compact.py` (31 lines)
- **Trigger:** Before context compaction
- **Input:** JSON from stdin (`session_id`, `transcript_path`)
- **Action:** Writes one line to `pages/log.md`: `## {date} | pre-compact | session={session_id}`
- **Does NOT** capture or save transcript content — just a log entry
- **Edge case:** If stdin is a TTY, uses empty dict and logs "unknown"
- **Output:** JSON `{"status": "ok", "log_entry": "..."}`

### `hooks/session_end.py` (38 lines)
- **Trigger:** Session end/shutdown
- **Input:** JSON from stdin (`session_id`, `transcript_path`, `cwd`)
- **Action:** If `transcript_path` exists and is readable, copies it to `raw/{date}-session-{timestamp}.md`, then logs to `pages/log.md`
- **Safe:** Handles missing transcript_path, missing source file, copy errors gracefully
- **Output:** JSON `{"status": "ok", "captured": bool}`

### `.pi/extensions/wiki-memory-hooks.ts` (165 lines, TypeScript)
- **`session_start`:** Shows notification "🧠 Wiki-memory dream hooks active" if context has UI
- **`resources_discover`:** Returns `{skillPaths: [SKILL_DIR]}` for auto-discovery
- **`before_agent_start`:** Reads `pages/index.md` (first 60 lines) and `.meta/skills/*.md` (first 5 files × 40 lines), injects as custom `wiki-memory-context` message with `display: false`
- **`session_before_compact`:** Runs `python3 dream_agent.py --quiet --idle 60` with 65s timeout, passes `WIKI_MEMORY_ROOT` and `AI_WIKI` env vars; reports errors via UI notification
- **`session_shutdown`:** Spawns `python3 dream_agent.py --quiet` as detached child (`spawn` with `detached: true`, `unref()`) for fire-and-forget
- **Project root resolution** (priority): `$WIKI_MEMORY_ROOT` env → `wiki-memory-root.json` marker → `~/code/wiki-memory` → `cwd/wiki-memory` → `cwd` → fallback to `$WIKI_MEMORY_ROOT` or `cwd`

### `plugin/plugin.json` (62 lines, legacy Claude Code format)
- **SessionStart:** Two hook commands:
  1. `cat ~/.local/share/ai-wiki/pages/index.md | head -60` (inject wiki index, 5s timeout)
  2. `cat ~/.local/share/ai-wiki/.meta/skills/*.md | head -40 || true` (inject auto-skills, 5s timeout)
- **PreCompact:** `bash -lc 'root="${WIKI_MEMORY_ROOT:-$HOME/code/wiki-memory}"; python3 "$root/dream/dream_agent.py" --quiet --idle 60 2>/dev/null || true'` (30s timeout)
- **SessionEnd:** `bash -lc 'root="${WIKI_MEMORY_ROOT:-$HOME/code/wiki-memory}"; python3 "$root/dream/dream_agent.py" --quiet 2>/dev/null' &` (2s timeout, background `&`)
- **agentConfig:** `target: ~/.pi/agent/AGENTS.md`, `injectAt: end`

---

## 5. Dream Agent — 6-Phase Cycle (`dream/dream_agent.py`)

| Phase | Function | Lines | Behavior | Notable |
|-------|----------|-------|----------|---------|
| 0 — Budget | `allocate_budget()` | 195-222 | `idle_seconds × 0.25`, capped 7200s. Dynamic intake/refine ratio (80%→33% as wiki matures). Sub-allocates refine→compile 40%, improve 50%, lint 10% | Refinement state estimated via regex on page files |
| 1 — Extract | `extract_from_clawmem()` + `_extract_from_raw()` | 226-344 | Primary: GET `/health`, POST `/search` on ClawMem, fetch docs. Fallback: scan `raw/` for unprocessed files (tracked via `intake_log.jsonl`). Entity/concept extraction via regex | Falls back gracefully if ClawMem unavailable |
| 2 — Refine | `refine_claim()` | 348-466 | 4-factor confidence: 35% self-consistency, 25% freshness, 25% cross-ref, 15% evidence count. Thresholds: ≥0.8 auto, 0.5-0.8 flagged, <0.5 rejected | Council escalation is a **stub** that always returns False (fails closed) |
| 3 — Compile | `compile_to_wiki()` | 470-588 | YAML frontmatter (title, created/updated, tags, confidence, status, sources, wikilinks). Pages routed to concepts/ or entities/ based on entity/concept presence. Append to existing pages | Status: stable≥0.8, needs_review≥0.5, draft<0.5 |
| 4 — Pattern Detect | `detect_patterns()` | 592-779 | 7 regex patterns: code-review, deployment, testing, debugging, database, api-development, research. At threshold 3, auto-creates SKILL.md with hardcoded templates | Templates are hardcoded strings, not configurable |
| 5 — Re-index | `trigger_reindex()` | 783-797 | POST `/reindex` to ClawMem with collection name | No-op if ClawMem unavailable |
| 6 — Improve | `improve_wiki()` | 801-887 | Structural fixes only: missing confidence, missing status, missing sources, broken wikilinks, stale dates. Max 5 per cycle. Budget-gated. | Embedding-guided gap analysis is **scaffolding only** — uses regex, no actual embedding |
| Lint | `lint_wiki()` | 891-937 | Orphan pages (not in index), broken wikilinks, stale drafts (>30 days), contradictions | Reporting only — no auto-fix |

### Git Integration (`git_commit()`, lines 941-958)
- Called after: Phase 3 (compile), Phase 4 (skills), Phase 6 (improve)
- `git add -A`, then `git diff --cached --quiet` to avoid empty commits
- Commit messages: `"compile: N created, M updated"`, `"create: N auto-skills from patterns"`, `"improve: N documents"`, `"cycle: lint — N issues found"`
- Silent failure on errors

---

## 6. Classification Behavior

**No classification or intent routing is implemented in the wiki-memory codebase.**

- `config.schema.yaml` defines an elaborate `intent_router` section with 2-stage classifier (heuristic regex + LLM refinement), 5 routing categories, and per-session cache
- No Python/TS code in this repo implements any classify() or route() function
- The dream agent's "classification" is limited to the Phase 2 confidence scoring (heuristic, keyword-based) and Phase 4 pattern detection (7 hardcoded regexes)
- The intent router implementation lives in ClawMem (external dependency) — this repo only has the config schema

---

## 7. Response Append / Context Injection Behavior

**No response-level append exists.** Content is injected into session **context** (before agent response generation), not appended to agent responses.

Three injection mechanisms:

1. **Pi extension `before_agent_start`** (TypeScript): Injects `customType: "wiki-memory-context"` message containing wiki index (60 lines) + auto-skills (5 files × 40 lines) as a system message before each agent turn

2. **`plugin/plugin.json` SessionStart hooks** (legacy Claude Code): Runs `cat` commands to dump wiki index and skills into session context at session start. The `injectAt: "end"` field in `agentConfig` suggests append behavior but is never consumed by any code in this repo — it's a Claude Code plugin.json convention field

3. **PreCompact/SessionEnd hooks**: Run dream agent, which writes to disk. No response injection.

---

## 8. Tests (`tests/test_deployment_integration.py`, 98 lines)

| Test | What It Checks | Real/Dry |
|------|---------------|----------|
| `test_pi_extension_is_global_safe_and_not_cwd_bound` | TS extension uses `WIKI_MEMORY_ROOT` env, not `process.cwd()` | Static analysis of file contents |
| `test_pi_installer_registers_extension_and_skill` | install-pi.sh references paths correctly | Static analysis |
| `test_main_setup_installs_pi_extension_when_hooks_enabled` | setup.sh references wiki-memory-hooks.ts and wiki-memory-root.json | Static analysis |
| `test_universal_cli_installer_covers_requested_agents` | cli/install.sh covers all 7 agent scripts | Static analysis |
| `test_codex_installer_links_shared_wiki_data_dir` | install-codex.sh uses AI_WIKI, not hardcoded `$PLUGIN_DIR/wiki` | Static analysis |
| `test_plugin_hooks_are_relocatable_by_environment` | plugin.json dream commands use `WIKI_MEMORY_ROOT` not `~/code/wiki-memory` | Static + JSON parse |
| `test_dream_cycle_reports_budget_used_for_real_cycle` | **Actual** dream cycle run with tmp_path — verifies claims_extracted ≥ 1, budget_used > 0 | Real execution |

**Coverage gaps:**
- No tests for any individual phase function (allocate_budget, refine_claim, compile_to_wiki, etc.)
- No tests for hooks/pre_compact.py or hooks/session_end.py
- No tests for scheduler.py
- No tests for setup.sh (bash)
- No mock-based integration tests (test_dream_cycle works but requires no real ClawMem)
- Total: 7 tests, all passing, ~0% coverage of actual business logic

---

## 9. Missing Dependencies

| Dependency | Required By | Notes |
|------------|-------------|-------|
| `clawmem` binary | setup.sh MCP, dream agent extract phase | Not bundled; npm or source install |
| `bun` | SETUP.md, setup.sh | Required for ClawMem source build |
| `memvid` | SETUP.md (optional) | Cold storage; not implemented in code |
| `systemd --user` | scheduler.py, setup.sh | Linux idle timer (primary trigger) |
| `loginctl` | scheduler.py | Idle detection (Linux only) |
| `@earendil-works/pi-coding-agent` (TypeScript types) | `.pi/extensions/wiki-memory-hooks.ts` | Pi ExtensionAPI types |
| `@lnilluv/pi-ralph-loop` | `plugin/goal/index.ts` (comment) | npm package for /goal loop |
| `pi-agent-suite` | `plugin/goal/index.ts` (comment) | npm package |
| GPU + llama-server | SETUP.md (optional) | For embedding/expansion/reranker at ports 8088-8090 |
| `python3` + `git` | setup.sh prerequisite check | Minimum requirements |

---

## 10. Gaps Between Docs and Code

### Config/Preset System (high severity)
- **`--preset` flag** (setup.sh lines 1172-1182): Validates the preset name (`lightweight|balanced|high-quality|max-context`) but **never applies any configuration changes** — it's a no-op label. The full preset definitions in `config.schema.yaml` (with model sizes, budget ratios, enabled agents, etc.) are never read by setup.sh or dream_agent.py.
- **`--config` flag**: Accepted but never used — no YAML/JSON parsing exists in setup.sh. The config file path is stored in `$CONFIG_FILE` but never read.
- **`config.schema.yaml`**: A comprehensive 881-line schema defining all parameters, but **no code in this repo reads it** at runtime.

### Intent Router (high severity)
- **Doc claim:** Two-stage classifier (heuristic + LLM) routing queries to memory tiers
- **Reality:** No Python/TS implementation. Config schema exists but no code.

### Council Escalation (medium severity)
- **Doc claim:** Two-model adversarial deliberation for borderline claims
- **Reality:** `_convene_council()` (line 456) is a 7-line stub that always returns `False` (fails closed into human review). Never consults an LLM.

### MemVid Cold Storage (medium severity)
- **Doc claim:** Tier 3 cold storage with QR-coded video archives
- **Reality:** `--skip-memvid` flag exists in setup.sh, but **no `setup_memvid()` function exists**. No code writes to MemVid.

### S-Tier Reference Improvement (medium severity)
- **Doc claim:** Embedding-guided quality scoring against S-tier reference documents
- **Reality:** Phase 6 (`improve_wiki`) only does structural regex fixes (missing frontmatter, broken dates). The embedding-based improvement is not implemented.

### Cross-Platform Idle Detection (medium severity)
- **Doc claim:** Cross-platform (Linux, macOS, WSL2)
- **Reality:** `scheduler.py` uses `loginctl` (Linux-only) with no macOS/WSL2 fallback. On non-Linux, it returns `DEFAULT_IDLE=600`.

### Improvement Budget Ratio (low severity)
- **Doc claim:** `IMPROVEMENT_BUDGET_RATIO = 0.33` env var configurable
- **Reality:** `allocate_budget()` hardcodes `improve_share = 0.50` (line 213). The env var is defined but unused in the budget calculation.

### Unused Env Vars (low severity)
- `MAX_ITERATIONS_PER_DOC`, `MEANING_PRESERVATION_THRESHOLD`, `CONVERGENCE_THRESHOLD` are all defined but never referenced in any function body.

### README.md Symlink Claim (low severity)
- **Doc claim:** `install.sh` creates `~/.pi/wiki` → `wiki-memory/wiki/` symlink
- **Reality:** The actual `install.sh` delegates to `setup.sh` which creates `WIKI_DATA` → `$HOME/ai-wiki` and `$WIKI_DATA` → `$HOME/.pi/wiki` — note the target is `WIKI_DATA` (the canonical data dir), not `wiki-memory/wiki/`.

### AGENTS_WIKI.md (low severity)
- **Doc claim:** Schema file copied from `$CANONICAL_DIR/AGENTS_WIKI.md`
- **Reality:** This file doesn't exist in the repo. If missing from both source and data dir, setup.sh generates a minimal default (lines 231-248).

### "7 Agents" Claim — Antigravity (low severity)
- Antigravity is listed as an installed agent but there is no Antigravity-specific integration code beyond the config.json symlink. It gets no skills, no MCP, no hooks.

### Setup Phase Count
- **Doc claim:** 8 phases (Phase 7 includes MemVid)
- **Reality:** 7 phases in `main()`. No MemVid phase. Phase numbering in comments says "3/8" etc. but there are only 7 numbered phases in `main()`.

---

## 11. File Summary

| File | Lines | Type | Status |
|------|-------|------|--------|
| `dream/dream_agent.py` | 1087 | Python | Core logic, fully functional but council stub, improve scaffolding |
| `dream/scheduler.py` | 206 | Python | Functional, Linux-only idle detection |
| `setup.sh` | 1236 | Bash | Functional installer, preset/config flags are no-op |
| `install.sh` | 31 | Bash | Thin wrapper |
| `hooks/pre_compact.py` | 31 | Python | Minimal log-only hook |
| `hooks/session_end.py` | 38 | Python | Transcript capture hook |
| `.pi/extensions/wiki-memory-hooks.ts` | 165 | TypeScript | Full Pi lifecycle extension |
| `plugin/plugin.json` | 62 | JSON | Legacy Claude Code plugin manifest |
| `plugin/goal/index.ts` | 117 | TypeScript | /goal command (wraps ralph loop) |
| `cli/install.sh` | 45 | Bash | Universal installer dispatcher |
| `cli/install-*.sh` (7) | 10-29 each | Bash | Per-agent installers |
| `skill/SKILL.md` | 182 | Markdown | Agent-facing skill documentation |
| `SKILL.md` | 23 | Markdown | Lightweight skill pointer |
| `tests/test_deployment_integration.py` | 98 | Python | 7 tests, static analysis + 1 real cycle |
| `config.schema.yaml` | 881 | YAML | Unused config schema |
| `spec-as-built.md` | 372 | Markdown | Prior audit, partially accurate |
| `INSTALL.md` | 304 | Markdown | Install docs |
| `SETUP.md` | 211 | Markdown | Setup guide |
| `README.md` | 123 | Markdown | Project overview |
| `CLAUDE.md` | 35 | Markdown | Brief reference |

---

## 12. Runtime Architecture (As-Built)

```
Agent session lifecycle
  │
  ├── SessionStart / before_agent_start
  │     └── Inject wiki index + auto-skills into context
  │         (via plugin.json cat commands or Pi TS extension)
  │
  ├── PreCompact / session_before_compact
  │     ├── Run dream_agent.py --quiet --idle 60 (≤65s budget)
  │     └── hooks/pre_compact.py logs to pages/log.md
  │
  ├── SessionEnd / session_shutdown
  │     ├── hooks/session_end.py copies transcript → raw/
  │     └── Fire dream_agent.py --quiet (background, no wait)
  │
  ├── Idle timer (systemd 30min / daemon loop)
  │     └── scheduler.py → dream_agent.py --idle 600
  │
  └── Manual
        └── python3 dream/dream_agent.py --idle N

Dream cycle flow:
  Phase 0: allocate_budget()
  Phase 1: extract_from_clawmem() → _extract_from_raw() fallback
  Phase 2: refine_claim() → confidence scoring
  Phase 3: compile_to_wiki() → YAML pages
  Phase 4: detect_patterns() → auto-skill if ≥3
  Phase 5: trigger_reindex() → ClawMem
  Phase 6: improve_wiki() → structural fixes
  Lint: lint_wiki() → report only
  Git: git_commit() after phases 3, 4, 6
```

Key architectural note: The only autonomous runtime is `scheduler.py`. The hooks are passive triggers. Without a running daemon or systemd timer, the dream agent only runs on session lifecycle events (compact/shutdown) with a 60s budget.

---

## 13. Key Version Info

- `plugin.json` version: `3.1.0`
- `config.schema.yaml` version: `1.0.0`
- SKILL.md claims "v3" throughout
- Skill auto-generation templates version: `1.0.0` (hardcoded in dream_agent.py line 739)
- No `__version__` in dream_agent.py or scheduler.py
- Spec-as-built.md: v1.0 (dated 2026-05-22)

---

## 14. Current aaa-memory Update and Proposal

---
date: 2026-06-24 19:48:49 PDT
ver: 2.0.0
author: codex
model: gpt-5
tags: [aaa-memory,cass,audit,proposal,retrieval,wiki,clawmem,verification]
---

# Current-State Delta

This June 8 audit describes the old `wiki-memory` repo. The active project is now `aaa-memory` at `/home/cheta/code/aaa-memory`, and it is current with its cloud repo: `git fetch origin` completed and `git rev-list --left-right --count main...origin/main` returned `0 0`.

## 14.1 Current Entrypoints

| File | Current role | Status |
|------|--------------|--------|
| `scripts/mem.py` | User CLI for save, recall, inject, list, forget, capture, stats | Active hot-memory interface |
| `src/aaa_memory/mcp.py` | MCP server with `memory_search`, `memory_sessions`, `memory_timeline`, `memory_store` | Active, stdio and SSE daemon modes |
| `src/aaa_memory/retrieval/pipeline.py` | Unified tier search: hot vault, wiki FTS, ClawMem warm, cold fallback | Implemented, but has correctness gaps |
| `src/aaa_memory/warm/dream.py` | Sleep-time wiki compiler from ClawMem and raw files | Implemented scaffold, not fully original intent |
| `scripts/cass_context_hook.py` | Claude `UserPromptSubmit` hook using cass search history | Implemented, bounded, tested |
| `aaa-memory` | System CLI from `src/aaa_memory/cli.py` | Present in package metadata |

## 14.2 What Now Meets Original Intent

| Intended capability | Current evidence | Assessment |
|--------------------|------------------|------------|
| One shared memory vault | `~/.cache/aaa-memory/vault.sqlite`, `VaultMemoryStore`, MCP store writes | Mostly met |
| Hot recall from explicit memories | `scripts/mem.py recall`, `hot_memories` scoring | Met for durable facts |
| Multi-tier retrieval | `pipeline.search()` queries hot, wiki, ClawMem, cold fallback | Partly met |
| RRF fusion | `rrf_fusion()` exists in `retrieval/pipeline.py` and `retrieval/fusion.py` | Partly met, duplicated |
| Token budget | `enforce_token_budget()` exists | Partly met, currently weak after fusion |
| Intent routing | `router/intent.py` has rule plus OpenRouter fallback; pipeline has separate local classifier | Partly met, split implementation |
| Wiki compilation | `warm/dream.py` writes YAML pages with wikilinks; `wiki/compiler.py` compiles extracted elements | Partly met, duplicate compilers |
| Cass-based session recall | `scripts/cass_context_hook.py` uses `cass search` and tested mock behavior | Initial hook met |
| Agent access | MCP tools, CLI, Hermes provider package, parser modules | Partly met |

## 14.3 Defects Against Current Intent

| Priority | Defect | Impact | Fix |
|----------|--------|--------|-----|
| P0 | Cass is only a prompt-history hook, not a retrieval tier or pack source | The system misses the strongest session search source when `memory_search` runs | Add a cass adapter with `triage`, `search`, `pack`, `view`, and `expand`; include cass in RRF as a session-history tier |
| P0 | `pytest -q` fails at collection due missing `openai` package | Cannot claim test suite passes in the current environment | Install project deps or make `openai` an optional/lazy import for tests that do not need LLM calls |
| P1 | Pipeline has duplicate intent classifiers and duplicate RRF helpers | Behavior drifts and tests may validate the wrong code path | Make `router.intent.classify_intent` and `retrieval.fusion.rrf_fusion` the single implementations used by pipeline |
| P1 | Fusion output drops or truncates enough payload metadata that token budgeting becomes approximate | Context limits can be exceeded or useful evidence can be dropped poorly | Preserve `raw_text`, source path, line, title, timestamps, and score components through fusion |
| P1 | `config.py` defaults use `~/knowledge/wiki/raw`, while README storage docs still say `~/ai-wiki` | Operators and hooks can read/write different trees | Pick one canonical data root and document migration. Recommended: keep `~/.cache/aaa-memory` for DB, keep `~/ai-wiki` compatibility symlink, make config env-driven |
| P1 | Dream agent still reads from ClawMem/raw, not cass or the vault as first-class evidence with provenance | Compiled wiki misses current agent history and exact session citations | Feed dream extraction from vault plus cass packs, then ClawMem/wiki |
| P2 | ClawMem role is ambiguous across docs and code | Tier semantics remain confusing | Reframe ClawMem as warm indexed document tier, cass as session-history tier, vault as explicit durable memory tier |
| P2 | Cold storage remains local SQLite fallback, not MemVid V2 | Original cold-storage goal is not met | Defer MemVid; define cold tier as explicit future work |

## 14.4 Cass Findings

Runtime source of truth:

- `command -v cass` returned `/home/cheta/.local/bin/cass`.
- `cass api-version --json` returned crate version `0.6.14`, API version `1`, contract version `1`.
- `cass triage --json` reports initialized but unhealthy because the lexical index is stale, with next command `cass index --json --no-progress-events --data-dir /home/cheta/.local/share/coding-agent-search`.
- `cass capabilities --robot-format compact` reports connectors for Codex, Claude Code, Gemini, OpenCode, Pi Agent, Factory, OpenClaw, Antigravity, Qwen, Hermes, and more.
- `cass` official robot docs say agents should start with `cass triage --json`, use `cass search ... --robot --robot-meta` for bounded search, and use `cass pack ... --robot` for cited handoff evidence.

Local repo source:

- Repo path is `/home/cheta/git/coding_agent_session_search`.
- After fetch, local `main` is behind upstream by `377` commits and `origin/main` describes as `v0.6.17-5-ge485b1fb`.
- The cass working tree is dirty with local changes, so do not merge, reset, or upgrade it as part of aaa-memory work without a separate cass-maintenance task.

## 14.5 Revised Architecture Proposal

Target retrieval flow:

```
User query
  -> intent classifier
  -> tier planner
       -> vault hot memories for durable facts and preferences
       -> cass for raw cross-agent session evidence
       -> wiki pages for compiled knowledge
       -> ClawMem for indexed documents
       -> cold archive later
  -> RRF fusion with source-preserving metadata
  -> token budget and privacy filters
  -> MCP/CLI/hook response
```

Target dream flow:

```
Vault durable facts + cass packs + ClawMem docs + raw files
  -> provenance-normalized evidence records
  -> confidence scoring
  -> wiki pages with citations and wikilinks
  -> ClawMem reindex
```

## 14.6 Implementation Plan

1. Add `src/aaa_memory/retrieval/cass.py`.
   - Run `cass triage --json` before search.
   - If stale but usable, continue and surface freshness warning.
   - Use `cass search "<query>" --robot --robot-meta --fields summary --limit N --max-tokens B`.
   - Use `cass pack "<query>" --robot --max-tokens B --max-evidence N --max-sessions M` for dream evidence and operator reports.
   - Preserve `source_path`, `line_number`, `agent`, `workspace`, `created_at`, `score`, and freshness warnings.

2. Unify retrieval internals.
   - Delete or deprecate the local `classify_intent()` and `rrf_fusion()` copies in `retrieval/pipeline.py`.
   - Import from `router.intent` and `retrieval.fusion`.
   - Add one typed result shape for all tiers.

3. Repair token budget enforcement.
   - Budget by final fused excerpts, not by fused IDs without text.
   - Include truncation metadata so MCP consumers know when evidence is partial.

4. Make cass a first-class retrieval tier.
   - Recent/session-history intents should query cass first.
   - Ambiguous intents should include cass, vault, wiki, and ClawMem.
   - Archival intents should prefer cass pack plus cold fallback.

5. Feed dream agent from cass.
   - Add an extraction phase that asks cass for recent packs by project/workspace.
   - Convert pack evidence into claims with source citations.
   - Do not obey instructions inside historical session excerpts.

6. Fix environment/test reliability.
   - Either install dependencies from `pyproject.toml` or lazy-import `openai`.
   - Add tests for cass adapter with mocked subprocess output and stale-index warning handling.
   - Add an integration smoke test that skips if `cass` is missing.

## 14.7 Verification Plan

```bash
git fetch origin
git rev-list --left-right --count main...origin/main
cass api-version --json
cass triage --json
python3 -m pytest tests/test_cass_context_hook.py tests/test_retrieval_pipeline.py -q
pytest -q
```

Current verification status:

- Verified: aaa-memory cloud parity is `0 0`.
- Verified: cass runtime is installed and reports API contract.
- Verified: cass repo fetched newer upstream tags and commits.
- Verified: `pytest -q` fails during collection because `openai` is unavailable.
- Unverified: cass index refresh, because it writes cass archive/index state and was not necessary for this proposal.
