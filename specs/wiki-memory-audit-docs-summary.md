# Wiki Memory Documentation Audit — Implementation Requirements Summary

**Audit date:** 2026-06-08  
**Source directory:** /home/cheta/code/wiki-memory  
**Task:** Produce implementation requirements for the global /home/cheta/memory layout  
**Scope:** All .md .yaml .json files audited; no files modified

---

## 1. INTENDED ARCHITECTURE

### Three-Tier Memory Stack (Hottest → Coldest)

| Tier | Component | Storage | Purpose |
|------|-----------|---------|---------|
| **Hot/Warm** | ClawMem (yoloshii/ClawMem v0.10.1) | SQLite + FTS5 + sqlite-vec | Hybrid search, session memory, graph traversal |
| **Warm** | Karpathy Wiki (compiled pages) | YAML frontmatter + markdown + git | Human-readable compiled knowledge |
| **Cold** | MemVid V2 | .mv2 files (QR-coded vectors → MP4) | Compressed archival, multi-resolution search |

### Data Flow

```
Session hooks → ClawMem vault (hot/warm)
                    │
                    ▼
          Dream Agent (systemd idle timer, Python sidecar)
          6-phase cycle: Budget → Extract → Refine → Compile → Pattern Detect → Re-index → Improve
                    │
                    ▼
          Karpathy Wiki pages/ (warm, git-backed)
                    │ (monthly)
                    ▼
          MemVid V2 .mv2 files (cold archival)
```

### Key Design Decisions (from specs/ARCHITECTURE.md, specs/MASTER_SPEC.md)

- DD1: Replace custom SQLite adapter stub with real yoloshii/ClawMem (used as-is, no fork)
- DD2: Skip Graphiti (KuzuDB archived) — ClawMem natively has entity-level time-travel
- DD3: Dream agent as Python sidecar, NOT ClawMem plugin (needs systemd timer, budget, council, file I/O)
- DD4: Monthly MemVid re-encode creates natural 30-day refinement window
- DD5: Never modify backends (ClawMem, MemVid, inference servers used as-is)

---

## 2. REQUIRED COMPONENTS

### 2.1 ClawMem (Hot/Warm Memory)

| Attribute | Detail |
|-----------|--------|
| **Install** | `npm install -g clawmem` or `git clone + bun install` to `~/git/ClawMem` |
| **Binary** | `~/git/ClawMem/bin/clawmem` (source build) or `~/.bun/bin/clawmem` (npm) |
| **Service** | `clawmem serve --port 7438` (REST API) |
| **DB location** | `~/.cache/clawmem/index.sqlite` |
| **Collection** | `clawmem collection add <pages-dir> --name wiki` |
| **MCP server** | `clawmem mcp --register` (port 7438) |
| **Backend ports** | Embed: 8088, LLM/expand: 8089, Reranker: 8090 |

### 2.2 Dream Agent (Sleep-Time Compute)

| Attribute | Detail |
|-----------|--------|
| **File** | `dream/dream_agent.py` (~1085 lines, 6-phase cycle) |
| **Scheduler** | `dream/scheduler.py` (systemd idle timer / daemon / cron) |
| **Trigger** | systemd --user idle timer (30min check, 5min idle threshold) |
| **Budget** | `idle_seconds × 0.25`, capped at 7200s (2h) per cycle |
| **Dynamic ratio** | Intake/refinement shifts: 80/20 at raw wiki → 50/50 mid → 33/67 mature |
| **6 phases** | 0: Budget → 1: Extract → 2: Refine (confidence scoring) → 3: Compile (wiki pages) → 4: Pattern Detect (auto-skill at 3+ occurrences) → 5: Re-index → 6: Improve (structural lint + S-tier embedding improvement) |

### 2.3 Wiki Pages (Compiled Knowledge)

| Attribute | Detail |
|-----------|--------|
| **Structure** | `pages/concepts/`, `pages/entities/`, `pages/sources/`, `pages/queries/` |
| **Format** | YAML frontmatter + markdown + `[[wikilinks]]` |
| **Frontmatter** | title, created, updated, tags, confidence (0-1), status, sources (clawmem://docid/...), wikilinks, contradictions, entity_types, expires |
| **Git** | Auto-committed by dream agent after each compile/improve cycle |
| **Index** | `pages/index.md` (content catalog), `pages/log.md` (append-only action log) |

### 2.4 MemVid V2 (Cold Storage — Optional)

| Attribute | Detail |
|-----------|--------|
| **Format** | `.mv2` tar-based container (HDF5 vectors, ndjson metadata, zstd compression) |
| **Multi-resolution** | 256, 768, 1568, 2064, 4096 dimensions |
| **Schedule** | Monthly (1st of month) |
| **Tool** | Planned: `mv2` CLI (not yet implemented) |
| **Join table** | `memvid_indices.ndjson` → `chunks.metadata_json` for provenance chain |

### 2.5 Session Lifecycle Hooks

| Hook | File | Trigger | Action |
|------|------|---------|--------|
| **Pre-compact** | `hooks/pre_compact.py` | Before context compaction | Logs session ID, captures current context |
| **Session-end** | `hooks/session_end.py` | Session shutdown | Copies transcript to `raw/` |
| **Pi extension** | `.pi/extensions/wiki-memory-hooks.ts` | Pi lifecycle events | Registers skill, injects wiki index, runs dream agent on compact/shutdown |
| **plugin.json** | `plugin/plugin.json` | Claude Code hooks | SessionStart (inject wiki index), PreCompact (run dream agent), SessionEnd (fire-and-forget) |

### 2.6 Skills

| File | Purpose |
|------|---------|
| `skill/SKILL.md` | Full skill definition for Claude Code / Pi / Hermes |
| `SKILL.md` (root) | Stub skill descriptor (frontmatter only, references skill/SKILL.md) |
| `skill/SETUP.md` | Duplicate of root SETUP.md (agent-facing setup guide) |

### 2.7 CLI Install Scripts

| Script | Purpose |
|--------|---------|
| `install.sh` | Lightweight wrapper delegating to `setup.sh` |
| `setup.sh` | 8-phase multi-agent installer (data dir, symlinks, ClawMem, MCP, skills, hooks, dream, env vars) |
| `cli/install.sh` | Universal per-agent installer hub |
| `cli/install-pi.sh` | Pi-specific: copies extension .ts, writes root marker JSON, links skill |
| `cli/install-hermes.sh` | Hermes-specific: links SKILL.md, tells user to add hooks to config.yaml |
| `cli/install-claude-code.sh` | Claude Code: links plugin.json, skill, hooks, dream |
| `cli/install-codex.sh` | Codex: links wiki data directory |
| `cli/install-opencode.sh` | OpenCode: links hooks |
| `cli/install-kilocode.sh` | Kilocode: links dream directory |
| `cli/install-antigravity.sh` | Antigravity: writes plugin registration JSON |

### 2.8 SOUL / Context Changes

- **Pi integration**: `~/.pi/agent/AGENTS.md` should have a wiki section with instructions
- **AGENTS_WIKI.md**: Schema document at `~/.local/share/ai-wiki/AGENTS_WIKI.md` read at session start
- **Wiki MCP server** (port 7439, spec'd but disabled): Would expose `wiki_search`, `wiki_query`, `wiki_ingest` as MCP tools

---

## 3. CURRENT DEFAULT PATHS

### Data Directory (Canonical)

| Path | Purpose |
|------|---------|
| `~/.local/share/ai-wiki/` | **Primary wiki data location** (all specs agree) |
| `~/ai-wiki/` | User-friendly symlink → `~/.local/share/ai-wiki/` |

### Internal Data Structure

| Path | Purpose |
|------|---------|
| `~/.local/share/ai-wiki/raw/` | Immutable source documents |
| `~/.local/share/ai-wiki/pages/` | Compiled wiki articles |
| `~/.local/share/ai-wiki/pages/index.md` | Content catalog |
| `~/.local/share/ai-wiki/pages/log.md` | Append-only action log |
| `~/.local/share/ai-wiki/pages/concepts/` | Atomic concept articles |
| `~/.local/share/ai-wiki/pages/entities/` | People, orgs, tools |
| `~/.local/share/ai-wiki/pages/sources/` | Source summaries |
| `~/.local/share/ai-wiki/pages/queries/` | Filed QA pairs |
| `~/.local/share/ai-wiki/.meta/` | Runtime state |
| `~/.local/share/ai-wiki/.meta/skills/` | Auto-generated skill refs |
| `~/.local/share/ai-wiki/.meta/references/` | S-tier reference exemplars |
| `~/.local/share/ai-wiki/.meta/intake_log.jsonl` | Processing log |
| `~/.local/share/ai-wiki/.meta/skill_patterns.json` | Pattern tracking |
| `~/.local/share/ai-wiki/.git/` | Git repo |

### Agent Integration Paths

| Agent | Skill Path | MCP Config | Hooks |
|-------|-----------|------------|-------|
| Pi | `~/.pi/agent/skills/karpathy-wiki/` | `~/.pi/agent/mcp-cache.json` | `.pi/extensions/wiki-memory-hooks.ts` |
| Claude Code | `~/.claude/plugins/karpathy-wiki/` | `~/.claude/mcp.json` | `plugin/plugin.json` |
| Hermes | `~/.config/hermes/skills/karpathy-wiki.md` | `hermes mcp add` | `~/.config/hermes/config.yaml` |
| Ante | `~/.ante/skills/karpathy-wiki/` | `~/.ante/settings.json` | (none) |
| KiloCode | `~/.kilocode/skills/karpathy-wiki/` | `~/.kilocode/mcp_servers/` | `~/.kilocode/hooks/` |
| OpenCode | `~/.config/opencode/skills/karpathy-wiki/` | `~/.config/opencode/opencode.json` | `.opencode/hooks/` |
| Antigravity | n/a (direct file system) | n/a | n/a |

### ClawMem Paths

| Path | Purpose |
|------|---------|
| `~/.cache/clawmem/index.sqlite` | ClawMem SQLite vault |
| `~/git/ClawMem/` | Source code (if installed from git) |
| `~/.config/clawmem/index.yml` | ClawMem configuration |

### Code Directory

| Path | Purpose |
|------|---------|
| `~/code/wiki-memory/` | **Project root** (current location) |
| `~/code/wiki-memory/dream/dream_agent.py` | Dream agent executable |
| `~/code/wiki-memory/dream/scheduler.py` | Scheduler daemon |
| `~/code/wiki-memory/hooks/pre_compact.py` | Pre-compact hook |
| `~/code/wiki-memory/hooks/session_end.py` | Session-end hook |
| `~/code/wiki-memory/skill/SKILL.md` | Skill definition |
| `~/code/wiki-memory/plugin/plugin.json` | Plugin manifest |
| `~/code/wiki-memory/config.schema.yaml` | Central configuration |
| `~/code/wiki-memory/setup.sh` | Multi-agent installer |

---

## 4. CONFIGURABLE PATH KNOBS

### From `config.schema.yaml`

| Config Key | Default Value | Env Var | Description |
|------------|--------------|---------|-------------|
| `wiki.data_dir` | `~/.local/share/ai-wiki` | `AI_WIKI`, `WIKI_DATA_DIR` | All wiki data (raw, pages, .meta) |
| `wiki.code_dir` | `skills-USER/karpathy-wiki` | — | Canonical code location |
| `clawmem.binary` | `~/git/ClawMem/bin/clawmem` | — | Path to clawmem binary |
| `clawmem.install_path` | `~/git/ClawMem` | — | Where ClawMem is cloned |
| `clawmem.url` | `http://localhost:7438` | `CLAWMEM_URL` | REST API URL |
| `clawmem.collection` | `wiki` | `CLAWMEM_COLLECTION` | Collection name |
| `memvid.archive_dir` | `~/.cache/memvid` | — | Where .mv2 files live |
| `agents.pi.skill_path` | `~/.pi/agent/skills/karpathy-wiki` | — | Pi skill link target |
| `agents.pi.plugin_path` | `~/.pi/agent/plugins/goal` | — | Pi plugin link target |

### From `setup.sh`

| Variable | Default | Description |
|----------|---------|-------------|
| `WIKI_DATA` | `$HOME/.local/share/ai-wiki` | Data directory |
| `CLAWMEM_SOURCE_DIR` | `$HOME/git/ClawMem` | ClawMem source |
| `CLAWMEM_BINARY` | `$CLAWMEM_SOURCE_DIR/bin/clawmem` | ClawMem binary |
| `MEMVID_SOURCE_DIR` | `$HOME/git/memvid` | MemVid source |

### From `.pi/extensions/wiki-memory-hooks.ts`

| Variable | Default | Description |
|----------|---------|-------------|
| `WIKI_MEMORY_ROOT` | (detected) | Project root directory |
| `AI_WIKI` | `~/ai-wiki` | Wiki data directory |

### From `plugin/plugin.json`

| Variable | Default | Description |
|----------|---------|-------------|
| `WIKI_MEMORY_ROOT` | `$HOME/code/wiki-memory` | Project root (used in hook commands) |

### From `hooks/pre_compact.py` and `hooks/session_end.py`

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_WIKI` | `~/ai-wiki` | Wiki data directory |

---

## 5. VERIFICATION COMMANDS

### Path Verification
```bash
# Data directory
ls -la ~/.local/share/ai-wiki/
ls -la ~/ai-wiki/                    # Should be symlink

# Symlinks
readlink -f ~/ai-wiki
readlink -f ~/.pi/agent/skills/karpathy-wiki
readlink -f ~/.pi/wiki

# Project code
ls ~/code/wiki-memory/dream/dream_agent.py
ls ~/code/wiki-memory/config.schema.yaml
```

### ClawMem Verification
```bash
# Health check
curl http://localhost:7438/health

# Collection check
clawmem collection list | grep -q '"wiki"' && echo "wiki collection exists"

# MCP check
ls ~/.claude/mcp.json && grep -q clawmem ~/.claude/mcp.json
ls ~/.pi/agent/mcp-cache.json && grep -q clawmem ~/.pi/agent/mcp-cache.json
```

### Dream Agent Verification
```bash
# Manual run
python3 ~/code/wiki-memory/dream/dream_agent.py --idle 60 --quiet

# Help
python3 ~/code/wiki-memory/dream/dream_agent.py --help

# Check compiled output after run
ls ~/.local/share/ai-wiki/pages/concepts/
cat ~/.local/share/ai-wiki/pages/index.md
cat ~/.local/share/ai-wiki/pages/log.md
```

### Hook Registration (Agent-specific)
```bash
# Pi Agent
grep -q "wiki-memory-hooks" ~/.pi/agent/extensions/wiki-memory-hooks.ts
# Claude Code
/plugin list                          # Should show karpathy-wiki
# Hermes
hermes skill list                     # Should show karpathy-wiki
```

### Setup Script (dry-run)
```bash
cd ~/code/wiki-memory && ./setup.sh --dry-run
```

### Full Integration Test Sequence
```bash
# 1. Create test source
echo -e "---\ntitle: Test\n---\nTest observation" > ~/.local/share/ai-wiki/raw/test-note.md

# 2. Run dream agent
python3 ~/code/wiki-memory/dream/dream_agent.py --idle 600

# 3. Check output
ls ~/.local/share/ai-wiki/pages/concepts/
git -C ~/.local/share/ai-wiki log --oneline -5
```

---

## 6. CONTRADICTIONS AND INCONSISTENCIES

### 6.1 Path Dialect Inconsistency

The docset uses **three different path conventions** interchangeably for the same project root:

| Convention | Used In | Example |
|------------|---------|---------|
| `skills-USER/karpathy-wiki/` | `SETUP.md`, `skill/SETUP.md`, `config.schema.yaml` (symlinks section) | The *canonical placeholder path* per original project layout |
| `~/code/wiki-memory/` | `plugin/plugin.json` (WIKI_MEMORY_ROOT default), `.pi/extensions/wiki-memory-hooks.ts`, actual filesystem, `CHANGELOG.md` | The **current actual location** |
| `~/code/wiki-memory/karpathy-wiki/` | `PI_INTEGRATION_PLAN.md`, `COMPLETION_PLAN.md` (Phase 0 items) | The **old nested structure** (now flattened; no longer exists) |

**Status:** `COMPLETION_PLAN.md` Phase 0.1 says "Flatten directory structure ✅ DONE" but Phase 0.2-0.3 still reference `~/code/wiki-memory/karpathy-wiki/` paths that no longer exist.

### 6.2 Old Canonical Location Referenced in Specs

Multiple specs (`ARCHITECTURE.md`, `TIER_INTEGRATION.md`) reference `/home/cheta/code/karpathy-wiki/wiki/pages/` as the data path. This has been superseded by:
- Code: `~/code/wiki-memory/`
- Data: `~/.local/share/ai-wiki/`

### 6.3 Symlink Target vs Actual Path

- `config.schema.yaml` symlinks section uses `skills-USER/karpathy-wiki/skill` as the target — but the actual skill directory is at `~/code/wiki-memory/skill/`
- `setup.sh` resolves the canonical path dynamically via `pwd -P`, so the symlinks should be functional, but the config schema still references the old placeholder convention

### 6.4 Wiki Data Directory: Multiple Locations Mentioned

| Document | Path | Notes |
|----------|------|-------|
| `README.md` | `~/ai-wiki` (via `wiki/ → ~/ai-wiki` symlink) | Claims wiki/ symlinks to ~/ai-wiki |
| `SETUP.md` | `~/.local/share/ai-wiki/` | Canonical data dir |
| `INSTALL.md` | `~/ai-wiki/` | Uses ~/ai-wiki as primary |
| `ARCHITECTURE.md` | `/home/cheta/code/karpathy-wiki/wiki/pages/` | Outdated |
| `TIER_INTEGRATION.md` | `/home/cheta/code/karpathy-wiki/wiki` | Outdated |
| `config.schema.yaml` | `~/.local/share/ai-wiki` | Canonical (config default) |
| `plugin/plugin.json` | `~/.local/share/ai-wiki/pages/index.md` | Calls ~/.local/share/ai-wiki directly |
| `hooks/pre_compact.py` | `~/ai-wiki` (via AI_WIKI env var, defaulting to ~/ai-wiki) | Uses ~/ai-wiki |

### 6.5 MemVid Status

- Spec documents (`specs/requirements.md`, `specs/ARCHITECTURE.md`, `specs-next/MEMVID_COLD_STORAGE.md`) describe MemVid as a fully spec'd cold storage tier
- `spec-as-built.md` lists MemVid integration as "config-only — no code writes to it"
- `COMPLETION_PLAN.md` Phase 2 is the MemVid implementation, estimated 5-7 days
- `config.schema.yaml` has `memvid.enabled: false` by default
- **The `mv2` CLI tool does not exist yet**; neither MemVid encoding nor the memvid_indices join table are implemented

### 6.6 Council Deliberation Status

- Multiple specs describe a sophisticated adversarial council (v1.0.0 with 2 models, v2.0.0 with 8-10 personas)
- `spec-as-built.md` states council is a stub `_convene_council()` that always returns `True`
- `COMPLETION_PLAN.md` Phase 4 is dedicated to implementing this (3-5 days)
- `DELIBERATIVE_IMPROVEMENT_SCAN.md` Finding #1 confirms it's a 7-line stub

### 6.7 S-Tier Improvement Engine Status

- `VAULT_IMPROVEMENT.md`, `STIER_IMPROVEMENT_ENGINE.md` specify full embedding-guided quality engine
- `spec-as-built.md` says scaffolding only (structural fixes, no S-tier comparison)
- `COMPLETION_PLAN.md` Phase 3 is the implementation (5-7 days)
- Current Phase 6 only fixes: missing confidence → add 0.5 placeholder, missing update date → add today

### 6.8 Test Coverage

- **Zero tests** across all 24 features (confirmed by `spec-as-built.md`, `DELIBERATIVE_IMPROVEMENT_SCAN.md`)
- One test file exists at `tests/test_deployment_integration.py` (7 tests, integration-focused, runs against actual dream agent import)
- `TEST_STRATEGY.md` defines comprehensive plan (40+ tests, 5 categories)
- `COMPLETION_PLAN.md` Phase 5 is test implementation (3-5 days)

### 6.9 Intent Router Status

- Config schema has full `intent_router` section with 3 strategies, 6 intent signals, heuristic + LLM classification
- `spec-as-built.md`: "Config schema defines it clearly, but the actual heuristic + LLM classifier implementation is in ClawMem (external dependency), not in the wiki codebase itself"
- Not implemented inside the wiki codebase — depends entirely on ClawMem's built-in classification

### 6.10 Cross-Platform Idle Detection

- `REDESIGN.md` and specs favor systemd idle timer (Linux-only)
- `spec-as-built.md`: Cross-platform support is "low completeness — scheduler.py uses loginctl (Linux only) with no macOS/WSL2 fallback"
- `COMPLETION_PLAN.md` Phase 6.1 adds cross-platform idle detection (abstracted via `idle_detector.py`)
- WSL2 detection exists in `setup.sh` (`grep -qi microsoft /proc/version`) but no WSL2-specific idle detection

### 6.11 Plugin.json Hardcoded Paths

`plugin/plugin.json` hook commands use `WIKI_MEMORY_ROOT` env var with a fallback of `$HOME/code/wiki-memory`. The test at `tests/test_deployment_integration.py:60-71` asserts that these commands DO NOT contain `~/code/wiki-memory` and DO use `WIKI_MEMORY_ROOT`, confirming the intent to make them relocatable.

### 6.12 Two SETUP.md Files

- `/home/cheta/code/wiki-memory/SETUP.md` — root-level setup guide, references `skills-USER/karpathy-wiki/` paths
- `/home/cheta/code/wiki-memory/skill/SETUP.md` — identical content in the skill directory
- Both appear to be duplicates

---

## 7. IMPLEMENTATION SUMMARY FOR /home/cheta/memory LAYOUT

### What the Docs Say About the Target Layout

The user wants to transition from `~/.local/share/ai-wiki/` to `/home/cheta/memory/` as the global memory root. Based on the documentation audit, the components that need paths updated are:

**Data paths to map under `/home/cheta/memory/`:**
- `/home/cheta/memory/wiki/` — wiki output (replaces `~/.local/share/ai-wiki/pages/`)
- `/home/cheta/memory/.meta/` — runtime state (replaces `~/.local/share/ai-wiki/.meta/`)
- `/home/cheta/memory/raw/` — source documents (replaces `~/.local/share/ai-wiki/raw/`)

**Data paths that stay under existing convention (shared global):**
- `~/.cache/clawmem/index.sqlite` — ClawMem SQLite vault (hot/warm DB)
- `~/.cache/memvid/` — MemVid cold storage archives

**Code path:**
- `~/code/wiki-memory/` — code stays here (already the correct location)

### Required Configuration Changes (per spec)

1. **`config.schema.yaml`**: Change `wiki.data_dir` default from `~/.local/share/ai-wiki` to `/home/cheta/memory`
2. **`setup.sh`**: Change `WIKI_DATA` default from `$HOME/.local/share/ai-wiki` to `/home/cheta/memory`
3. **CLI install scripts** (`cli/install-*.sh`): Update `AI_WIKI` default from `$HOME/.local/share/ai-wiki` to `/home/cheta/memory`
4. **`hooks/pre_compact.py`** and **`hooks/session_end.py`**: Update fallback from `Path.home() / "ai-wiki"` to `/home/cheta/memory`
5. **`.pi/extensions/wiki-memory-hooks.ts`**: Update `AI_WIKI` default from `join(homedir(), "ai-wiki")` to `/home/cheta/memory`
6. **`plugin/plugin.json`**: Update `cat ~/.local/share/ai-wiki/...` paths to `/home/cheta/memory/...`
7. **`~/ai-wiki` symlink**: Should now point to `/home/cheta/memory` instead of `~/.local/share/ai-wiki`
8. **`~/.pi/wiki` symlink**: Should now point to `/home/cheta/memory`
9. **`clawmem collection`**: Must be re-added pointing to `/home/cheta/memory/wiki/` (or keep pointing to the wiki subdirectory)
10. **`AGENTS_WIKI.md`**: Should be at `/home/cheta/memory/AGENTS_WIKI.md`
11. **All CLI verification commands**: Update `~/.local/share/ai-wiki` → `/home/cheta/memory`
12. **All documentation** (README.md, SETUP.md, INSTALL.md, SKILL.md, all specs): Update path references

### Configurable Path Knobs Summary

| What | Current Default | Target Default | Configuration Point |
|------|----------------|---------------|-------------------|
| Wiki data root | `~/.local/share/ai-wiki` | `/home/cheta/memory` | `config.schema.yaml` `wiki.data_dir`, env `AI_WIKI`, env `WIKI_DATA_DIR` |
| Wiki wiki output | `~/.local/share/ai-wiki/pages` | `/home/cheta/memory/wiki` (or `/home/cheta/memory/pages`) | Subdirectory under data_dir |
| ClawMem vault | `~/.cache/clawmem/index.sqlite` | No change needed | `clawmem` CLI init |
| MemVid archive | `~/.cache/memvid` | No change needed | `config.schema.yaml` `memvid.archive_dir` |
| ClawMem binary | `~/git/ClawMem/bin/clawmem` | No change needed | `config.schema.yaml` `clawmem.binary`, env `CLAWMEM_BINARY` |
| S-tier references | `~/.local/share/ai-wiki/.meta/references` | `/home/cheta/memory/.meta/references` | Subdirectory under wiki data_dir |
| Auto-skills | `~/.local/share/ai-wiki/.meta/skills` | `/home/cheta/memory/.meta/skills` | Subdirectory under wiki data_dir |
| Symlink ~/ai-wiki | → `~/.local/share/ai-wiki` | → `/home/cheta/memory` | `setup.sh` / manual |
| Symlink ~/.pi/wiki | → `~/.local/share/ai-wiki` | → `/home/cheta/memory` | `setup.sh` / manual |

---

## 8. IMPLEMENTATION PRIORITY (from COMPLETION_PLAN.md)

| Phase | Focus | Est. Time | Ready for Implementation? |
|-------|-------|-----------|--------------------------|
| 0 | Pi Integration & Data Seeding | 1-2 days | ⚡ Partially (paths need update) |
| 1 | ClawMem Integration | 3-5 days | ⚡ Config done, embedding pipeline needed |
| 2 | MemVid Cold Storage | 5-7 days | ❌ Not implemented (CLI tool + join table) |
| 3 | S-Tier Reference Improvement | 5-7 days | ❌ Not implemented (embedding comparator + rubric analyzer) |
| 4 | Council Deliberation | 3-5 days | ❌ Stub replacement needed |
| 5 | Tests | 3-5 days | ❌ Near-zero coverage |
| 6 | Polish (cross-platform, wiki MCP, backup, notifications) | 3-5 days | ❌ Mostly unimplemented |

**Total estimated effort:** 23-36 days (single developer)

---

## 9. KEY FILES AUDITED

| File | Type | Key Content |
|------|------|-------------|
| `README.md` | Overview | Quick install, structure, components, symlinks |
| `INSTALL.md` | Install guide | Per-CLI installation (Claude Code, Pi, Hermes, Codex, OpenCode, Kilocode, Antigravity) |
| `SETUP.md` | Setup guide | Prerequisites, ClawMem install, MemVid install, GPU services, config presets, verification |
| `skill/SETUP.md` | Skill setup | Duplicate of root SETUP.md |
| `SKILL.md` | Skill descriptor stub | Frontmatter referencing skill/SKILL.md |
| `skill/SKILL.md` | Full skill definition | 6-phase dream cycle, data locations, critical rules, page format |
| `config.schema.yaml` | Central configuration (881 lines) | All config knobs: wiki, clawmem, memvid, dream_agent, intent_router, classifier, LLM, rerank, MCP, agents, symlinks, hooks, env, presets |
| `plugin/plugin.json` | Plugin manifest (62 lines) | Hook definitions with env-var-relocatable dream_agent paths |
| `specs/ARCHITECTURE.md` | Architecture v4.1 | 3-tier stack, data flow, metadata pipeline, intent router model options, key DDs |
| `specs/MASTER_SPEC.md` | Master spec v1.0 | Cross-reference table, 6 user stories, architecture summary, implementation phases, risk summary |
| `specs/requirements.md` | Requirements v1.0 | 6 user stories, 37+ FR, 14 acceptance criteria, glossary |
| `specs/DREAM_AGENT_V2.md` | Dream agent v2.0 | 6-phase cycle, budget allocation, confidence scoring, council escalation, MemVid schedule integration |
| `specs/TIER_INTEGRATION.md` | Tier integration v1.0 | Component dependency matrix, ClawMem adapter rewrite, startup order, status checklist |
| `specs/METADATA_PIPELINE.md` | Metadata pipeline v1.1 | YAML per layer, SQLite columns, memvid_indices join chain, read/write paths |
| `specs/VAULT_IMPROVEMENT.md` | Vault improvement v1.0 | 8 document types, S-tier references, embedding loss function, rubric scoring, 6 improvement phases |
| `spec-as-built.md` | As-built audit (372 lines) | 24 features catalogued with completeness/consistency/coverage ratings |
| `REDESIGN.md` | Old redesign doc | Dual-agent pattern, idle timer, percentage budget, deliberative refinement, YAML schema |
| `PI_INTEGRATION_PLAN.md` | Pi integration plan | Skill registration, plugin hooks, MCP, data dir, dream agent activation, seed/test |
| `COMPLETION_PLAN.md` | Master completion plan (415 lines) | 6 phases with detailed tasks, dependency map, feature-completeness matrix, open questions resolution |
| `specs-next/CLAWMEM_INTEGRATION.md` | ClawMem integration v1.0 | Deployment, REST API endpoints, embedding strategy, data flow, error handling, test checklist |
| `specs-next/MEMVID_COLD_STORAGE.md` | MemVid cold storage v1.0 | .mv2 format spec, embedding schema, snapshot lifecycle, memvid_indices join table, CLI tool |
| `specs-next/STIER_IMPROVEMENT_ENGINE.md` | S-tier improvement v1.0 | Reference corpus, embedding comparator, rubric gap analysis, LLM refinement, quality tracking |
| `specs-next/COUNCIL_DELIBERATION.md` | Council v1.0 | 2-model adversarial, 3 rounds, verdict handling, stub migration path |
| `specs-next/COUNCIL_DELIBERATION_V2.md` | Council v2.0 | 8-10 personas, evidence grounding, convergence detection, anti-sycophancy, decision packets |
| `specs-next/TEST_STRATEGY.md` | Test strategy v1.0 | 5 test categories, 40+ tests, CI integration, priority matrix |
| `specs-next/DELIBERATIVE_IMPROVEMENT_SCAN.md` | Improvement scan | 12 specific findings with line ranges, current behavior, proposed fixes |
| `hooks/pre_compact.py` | Pre-compact hook | Captures session before compaction (31 lines) |
| `hooks/session_end.py` | Session-end hook | Copies transcript to raw/ (38 lines) |
| `.pi/extensions/wiki-memory-hooks.ts` | Pi TypeScript extension | 4 lifecycle events, PROJECT_ROOT resolution with fallbacks (165 lines) |
| `dream/dream_agent.py` | Dream agent | ~1085 lines, 6 phases, confidence scoring, council stub, lint |
| `dream/scheduler.py` | Scheduler | systemd timer, daemon, cron modes |
| `setup.sh` | Multi-agent installer | 8 phases, ~1236 lines |
| `install.sh` | Quick install wrapper | Delegates to setup.sh (31 lines) |
| `cli/install.sh` | Universal CLI installer | Hub dispatching to per-agent scripts |
| `cli/install-pi.sh` | Pi installer | Copies .ts file, writes root marker JSON, links skill |
| `cli/install-hermes.sh` | Hermes installer | Links SKILL.md, instructs on hooks config |
| `tests/test_deployment_integration.py` | Integration test | 7 tests, dream_agent import test with temp wiki dir |
| `questions.md` | Audit questions | 15 questions targeting specific features |
| `questions.json` | Audit questions (JSON) | Same 15 questions in structured format |
| `council-plan.md` | Council plan | Recommended 8-agent parallel council formation |
| `USER_PROMPTS.md` | User prompts | Historical session context (8 project-specific prompts) |
| `CHANGELOG.md` | Changelog | Feature additions, changes, fixes, removals |
| `CLAUDE.md` | Per-session reference | Minimal quick reference for Claude Code |

---

## 10. Current Documentation Update and Revised Proposal

---
date: 2026-06-24 19:48:49 PDT
ver: 2.0.0
author: codex
model: gpt-5
tags: [aaa-memory,cass,docs,audit,proposal,architecture,paths,verification]
---

# Docs Delta

This file was written against the old `wiki-memory` documentation. The active implementation is now `aaa-memory` in `/home/cheta/code/aaa-memory`, and the docs need to describe what exists now rather than only the earlier ClawMem/wiki/MemVid plan.

## 10.1 Current Cloud and Runtime Facts

| Fact | Evidence | Status |
|------|----------|--------|
| aaa-memory local repo is current with cloud | `git fetch origin`; `git rev-list --left-right --count main...origin/main` returned `0 0` | Verified |
| aaa-memory working tree was clean before these markdown edits | `git status --short --branch` showed `## main...origin/main` | Verified before edits |
| cass installed runtime exists | `command -v cass` returned `/home/cheta/.local/bin/cass` | Verified |
| cass installed version | `cass api-version --json` returned `0.6.14`, API `1`, contract `1` | Verified |
| cass runtime health | `cass triage --json` says initialized but unhealthy because lexical index is stale | Verified |
| cass upstream has moved | `/home/cheta/git/coding_agent_session_search` fetched `origin/main`, now `377` commits ahead of local branch; latest fetched describe `v0.6.17-5-ge485b1fb` | Verified |
| cass local repo has uncommitted work | `git status --short --branch` in cass repo lists modified and untracked files | Verified |

## 10.2 Updated Intended Architecture

The intended architecture should be revised from a strict three-tier memory stack into a four-source retrieval stack:

| Source | Role | Why |
|--------|------|-----|
| aaa-memory vault | Explicit durable facts, preferences, MCP writes, hot memories | Best place for small persistent facts and user preferences |
| cass | Cross-agent raw session history and cited evidence packs | Best existing source for session data across Codex, Claude Code, Hermes, Pi, Qwen, OpenCode, Antigravity, and others |
| wiki pages | Human-readable compiled knowledge | Best place for stable distilled knowledge |
| ClawMem | Indexed documents and warm document retrieval | Useful as a document index, not the only source of truth |
| cold archive | Future MemVid or SQLite archive | Not yet original-intent complete |

Proposed flow:

```
Agent hooks and MCP stores
  -> aaa-memory vault for explicit memories
  -> cass index for raw session history
  -> dream agent reads vault + cass packs + ClawMem + raw files
  -> wiki pages with citations
  -> ClawMem reindex for document retrieval
```

## 10.3 Documentation Corrections Needed

| Priority | Docs problem | Impact | Correction |
|----------|--------------|--------|------------|
| P0 | README says all agents share `~/.cache/aaa-memory/vault.sqlite`, but cass is now the stronger source for raw cross-agent sessions | Understates the real session-search backend | Add cass as the session-history backend and explain vault vs cass |
| P0 | Current docs imply retrieval is fully intent-routed and RRF-fused | Code has partial and duplicated routing/fusion | Mark as implemented-but-needs-consolidation |
| P1 | Storage docs still mix `~/ai-wiki`, `~/knowledge/wiki`, and old `/home/cheta/memory` proposal | Operators can configure hooks against the wrong directory | Define canonical paths and compatibility symlinks |
| P1 | ClawMem is described as warm/cold differently across docs | Tier roles are hard to reason about | Define ClawMem as document index tier; define cass as session evidence tier |
| P1 | Dream agent docs do not mention cass | Future implementation would ignore the best source of session evidence | Add cass-pack ingestion to dream-agent design |
| P2 | MemVid is still described as planned architecture with no current implementation | Makes the project look more complete than it is | Move MemVid to deferred cold-storage roadmap |
| P2 | Test docs are stale | Current suite cannot pass without dependency setup | Document `pytest -q` collection failure if deps are missing |

## 10.4 Updated Path Policy

Recommended canonical paths:

| Purpose | Path | Notes |
|---------|------|-------|
| aaa-memory vault | `~/.cache/aaa-memory/vault.sqlite` | SQLite hot memory and turns |
| aaa-memory cache/logs | `~/.cache/aaa-memory/` | Existing config default |
| wiki output | `~/ai-wiki/pages` or configured `AAA_MEMORY_WIKI` | Keep `~/ai-wiki` for compatibility unless deliberately migrated |
| raw intake | `~/ai-wiki/raw` or configured `AAA_MEMORY_RAW` | Should align with dream agent |
| cass data | `~/.local/share/coding-agent-search/` | cass runtime data dir |
| cass repo | `/home/cheta/git/coding_agent_session_search` | Dirty local repo, do not mutate casually |
| ClawMem repo | `/home/cheta/git/ClawMem` | Existing local source |

Do not force `/home/cheta/memory` as a new default until the active code, README, `config.py`, hooks, and setup docs are changed together. The safer migration is:

1. Keep current defaults working.
2. Add one documented env-driven override set: `AAA_MEMORY_CACHE`, `AAA_MEMORY_WIKI`, `AAA_MEMORY_RAW`, `AI_WIKI`, `CASS_DATA_DIR`.
3. Add verification commands that print resolved paths from the live config.
4. Only then decide whether `/home/cheta/memory` becomes the canonical data root.

## 10.5 Revised Requirements

### R1. Cass Adapter

Build an internal adapter around the official robot contract:

```bash
cass triage --json
cass search "<query>" --robot --robot-meta --fields summary --limit 10 --max-tokens 4000
cass pack "<query>" --robot --max-tokens 4000 --max-evidence 8 --max-sessions 3
cass view <source_path> -n <line_number> --json
cass expand <source_path> -n <line_number> -C 3 --json
```

Acceptance:

- Adapter never runs bare `cass`.
- Adapter treats stale lexical index as a freshness warning when search remains usable.
- Adapter surfaces `recommended_commands` instead of repairing cass automatically.
- Adapter preserves citations: source path, line number, agent, workspace, timestamp, score.

### R2. Retrieval Planner

Replace vague all-tier search with intent-planned search:

| Intent | Primary sources | Secondary sources |
|--------|-----------------|-------------------|
| recent/current/session | cass, vault | wiki |
| preference/fact | vault | wiki, cass |
| implementation history | cass pack | wiki, ClawMem |
| stable concept | wiki, ClawMem | vault |
| archival | cass pack, cold fallback | wiki |
| ambiguous | all, fused | none |

Acceptance:

- One classifier implementation.
- One RRF implementation.
- Token budget applied after source-preserving fusion.

### R3. Dream Agent Inputs

Dream agent should consume:

- Explicit vault memories.
- Recent cass packs by workspace/project.
- ClawMem documents.
- Raw intake files.

Acceptance:

- Historical cass excerpts are treated as data, not instructions.
- Wiki page frontmatter includes citations to cass source paths/lines when sourced from cass.
- ClawMem reindex runs after wiki writes if ClawMem is available.

### R4. Docs and Setup

Update docs in this order:

1. README architecture diagram and storage table.
2. SETUP with cass preflight and index commands.
3. MCP docs showing memory search uses vault plus cass.
4. Troubleshooting for missing `openai`, stale cass index, and ClawMem offline.
5. Path policy and env vars.

## 10.6 Verification Commands

```bash
cd /home/cheta/code/aaa-memory
git fetch origin
git rev-list --left-right --count main...origin/main
git status --short --branch
cass api-version --json
cass triage --json
pytest -q
```

Current verification result:

- aaa-memory cloud parity: verified, `0 0`.
- cass runtime: verified, installed `0.6.14`.
- cass freshness: verified stale lexical index, recommended refresh command available.
- cass upstream: verified newer upstream fetched, local branch behind by `377` commits.
- tests: `pytest -q` fails during collection because `openai` is unavailable in the current environment.

## 10.7 Recommended Next Implementation Slice

Do not start with MemVid, path migration, or UI polish. Start with cass integration because it changes the actual retrieval quality immediately.

1. Implement `src/aaa_memory/retrieval/cass.py`.
2. Add tests using mocked `subprocess.run` for healthy, stale, missing, and error cases.
3. Wire cass into `retrieval/pipeline.py`.
4. Consolidate classifier and fusion code.
5. Update README and SETUP to state the new vault plus cass architecture.
6. Run focused tests, then full tests after dependency setup.
