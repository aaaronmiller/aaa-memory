# Wiki/Memory Current-State Audit — Readiness & Risk Report

**Date:** 2026-06-08
**Scope:** Full audit of Hermes agent memory/hook configuration plus shared agent context files, without modification.
**Goal:** Replace current Honcho/wiki plan with Karpathy-wiki memory under `/home/cheta/memory/wiki` shared globally.

---

## 1. CURRENT MEMORY SYSTEMS

### 1.1 `/home/cheta/memory/` — Flat-file memory (Honcho-era, April 2026)

| Component | Path | Status |
|-----------|------|--------|
| agents/   | `memory/agents/hermes.md` | 1 file, 320 bytes |
| archive/  | `memory/archive/` | Empty (only .gitkeep) |
| channels/ | `memory/channels/discord-alerts.md` | 315 bytes |
| daily/    | `memory/daily/2026-04-03.md` | 291 bytes, stale |
| handoffs/ | `memory/handoffs/` | Empty (only .gitkeep) |
| projects/ | `memory/projects/` | 12 files (3833–5479 bytes each) |
| topics/   | `memory/topics/lessons.md` + `rules.md` | Small, stale |
| root files | `README.md`, `heartbeat-task-loop.md`, `master-queue.json`, `clawhip.log` | Last active Apr 2-3 2026 |

**Assessment:** Stale. Last write was April 3, 2026. The heartbeat loop and master queue are inactive. This was designed for a task-loop agent pattern that has been superseded.

### 1.2 `/home/cheta/wiki/` — Personal wiki (May 2026)

| Component | Path | Status |
|-----------|------|--------|
| comparisons/ | `wiki/comparisons/` | Empty dir |
| concepts/ | `wiki/concepts/` | 5 files (May 26 2026) |
| entities/ | `wiki/entities/NVDA.md, TSM.md, MU.md` | May 15 2026, stock research |
| queries/ | `wiki/queries/semiconductor-correction-timing.md` | 1 query result |
| raw/ | `wiki/raw/articles/, assets/, papers/, transcripts/` | Empty subdirs |
| root files | `index.md`, `SCHEMA.md`, `log.md` | May 26 2026 |

**Assessment:** Active-ish (last edit May 26). Contains personal research wiki content for semiconductor/AI investments. Uses `[[page-name]]` wiki-link format. Separate from the memory system.

### 1.3 `/home/cheta/.local/share/ai-wiki/` — Karpathy-wiki target (already deployed)

| Component | Path | Status |
|-----------|------|--------|
| pages/ | `ai-wiki/pages/` | entities/, concepts/, queries/, sources/ subdirs |
| pages/index.md | `ai-wiki/pages/index.md` | 317 bytes |
| pages/log.md | `ai-wiki/pages/log.md` | 746 bytes |
| raw/ | `ai-wiki/raw/` | Empty |
| .meta/ | `ai-wiki/.meta/` | skill_patterns.json, step_counter.py, skills/ |
| root | `AGENTS_WIKI.md` | 8133 bytes |

**Symlinks pointing here:**
- `/home/cheta/ai-wiki -> /home/cheta/.local/share/ai-wiki`
- `/home/cheta/.pi/wiki -> /home/cheta/.local/share/ai-wiki`

**Assessment:** This is the actual shared wiki target. It's already set up with the Karpathy-wiki structure. Pi's `~/.pi/wiki` symlink already points here. The wiki-memory project at `~/code/wiki-memory` targets this same directory for its data.

---

## 2. HERMES MEMORY CONFIGURATION

### 2.1 Config.yaml (relevant sections)

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  provider: local
  memory_char_limit: 4000
  user_char_limit: 2000
  wiki_enabled: true
  wiki_path: ~/wiki

context:
  engine: compressor
```

**Key finding:** Hermes is currently configured with `wiki_path: ~/wiki` (the separate personal wiki, NOT `~/ai-wiki` or `~/.local/share/ai-wiki`). The memory provider is `local` (not Honcho).

### 2.2 Honcho configuration

**`~/.hermes/honcho.json`**:
```json
{
  "hosts": {
    "hermes": {
      "peerName": "cheta", "workspace": "hermesbot", "aiPeer": "hermes",
      "memoryMode": "hybrid", "writeFrequency": "async", "recallMode": "hybrid",
      "sessionStrategy": "per-session", "enabled": true, "saveMessages": true,
      "observationMode": "directional", "dialecticCadence": 2,
      "dialecticReasoningLevel": "low"
    }
  }
}
```
- `apiKey` value redacted — present but must not be printed
- Same config duplicated in `~/.honcho/config.json`

**`~/.hermes/gateway_state.json`**: Gateway `state: "stopped"` since 2026-06-07. Telegram platform disconnected. Honcho appears **dormant/unused**.

**`~/.hermes/agent_state.json`**: Stale from April 19, 2026 — references a `fix-all-issues` task for Honcho timeouts/429s.

**Assessment:** Honcho integration was attempted but appears to have been abandoned mid-April. The gateway is stopped. Honcho is not a live dependency.

### 2.3 Hermes Built-in Memories

**`~/.hermes/memories/`**: Contains only MEMORY.md (3898 bytes) and USER.md (2004 bytes) — the standard Hermes short-term context memories. Nothing wiki-related here.

---

## 3. HOOKS DIRECTORIES

### 3.1 Hermes Hooks (`~/.hermes/hooks/`)

| Hook | Type | Purpose |
|------|------|---------|
| `command-capture/HOOK.yaml` | agent:start | Captures every user command to JSONL log |
| `command-capture/handler.py` | Python | Logs command metadata (timestamp, platform, source) |

**Assessment:** Minimal. Only one hook active — command capture for logging. **No wiki/memory lifecycle hooks are deployed in Hermes.**

### 3.2 Hermes Hooks-hermes (`~/.hermes/hooks-hermes/`)

Empty directory. No active hooks.

### 3.3 Claude Code Hooks (`~/.claude/hooks/`)

| Hook | Type | Purpose |
|------|------|---------|
| `context-mode-cache-heal.mjs` | SessionStart | Fixes context-mode plugin cache symlink breakage |
| `damage-control/` | (various) | Tool damage-control scripts (bash/write/edit/test), backup patterns.yaml |

**Assessment:** No wiki-related hooks. The damage-control hooks are safety measures.

### 3.4 Ante Hooks (`~/.ante/hooks/`)

| Hook | Type | Purpose |
|------|------|---------|
| `pre_compact.py` | pre_compact | Logs memory/program usage before context compaction |
| `session_end.py` | session_end | Logs session summary (tokens, cost, duration) |
| `block-danger.sh` | pre_tool_use (Bash*) | Blocks dangerous commands (rm -rf /, fork bombs, etc.) |

**Assessment:** Ante has observability hooks (pre_compact, session_end) and a safety blocklist. The pre_compact and session_end patterns match what the wiki-memory project would need to replace.

---

## 4. SKILL LOCATIONS

### 4.1 Central Skills Repository

**Path:** `/home/cheta/code/agents/skills/`
Contains ~110+ skills, each a directory with a SKILL.md and supporting files.

### 4.2 Skills Deployments (all symlinked to central repo)

| Agent | Skills Location | Skills Count | Symlinked |
|-------|----------------|--------------|-----------|
| **Hermes** | `~/.hermes/skills/` | 163 dirs | 24 symlinked directly to `code/agents/skills/` |
| **Claude Code** | `~/.claude/skills/` | 101+ | All symlinked to `code/agents/skills/` |
| **Codex** | `~/.codex/skills/` | ~95 | All symlinked to `code/agents/skills/` |
| **Pi Agent** | `~/.pi/agent/skills/` | ~101 | Most symlinked, some local copies (auto-*) |

**Key finding:** The wiki-memory project ships a skill at `code/wiki-memory/skill/SKILL.md`. This skill is NOT currently symlinked into any agent's skills directory. It would need to be added.

---

## 5. CONTEXT SYMLINKS

### 5.1 Shared Agent Context

| Symlink | Target | Agents Affected |
|---------|--------|-----------------|
| `~/.claude/agents -> ~/code/agents/agents` | Shared agent definitions | Claude Code |
| `~/.claude/CLAUDE.md -> ~/code/agents/CLAUDE.md` | Shared root CLAUDE.md | Claude Code |
| `~/.claude/commands -> ~/code/agents/commands` | Shared commands | Claude Code |
| `~/.codex/agents -> ~/code/agents/agents` | Shared agent definitions | Codex |
| `~/.codex/AGENTS.md -> ~/code/agents/CLAUDE.md` | Shared root CLAUDE.md | Codex |
| `~/.codex/commands -> ~/code/agents/commands` | Shared commands | Codex |

**Assessment:** Claude Code and Codex share centralized agents, CLAUDE.md, and commands from `~/code/agents/`. Changing wiki/memory paths in one likely affects all.

### 5.2 Wiki/Memory Symlinks

| Symlink | Target | Purpose |
|---------|--------|---------|
| `~/ai-wiki -> ~/.local/share/ai-wiki` | Karpathy-wiki data | User-facing access |
| `~/.pi/wiki -> ~/.local/share/ai-wiki` | Pi wiki reference | Pi agent wiki access |
| `~/wiki` | (real dir, not symlink) | Current separate wiki |
| `~/memory` | (real dir, not symlink) | Current separate memory |

**Assessment:** The `~/ai-wiki` pathway already exists and is the intended target for the wiki-memory project. `~/wiki` and `~/memory` are still separate real directories that would need migration.

---

## 6. AGENT-SPECIFIC INSTALL TARGETS

### 6.1 Claude Code (`~/.claude/`)

| Component | Path | Notes |
|-----------|------|-------|
| Settings | `~/.claude/settings.json` | Model: opus, 29 plugins enabled |
| Hooks | `~/.claude/hooks/` | context-mode-cache-heal + damage-control |
| Skills | `~/.claude/skills/` | 101+ symlinked skills |
| Plugins | `~/.claude/plugins/` | 29 enabled (ralph-loop, context7, frontend-design, etc.) |
| Projects | `~/.claude/projects/` | Per-project config (19 dirs) |
| File History | `~/.claude/file-history/` | 26 versions |
| **CLAUDERC target** | `~/.claude/CLAUDE.md -> ~/code/agents/CLAUDE.md` | Shared |

**Key for wiki-memory:** The `plugin/plugin.json` in wiki-memory defines Claude Code hooks (SessionStart, PreCompact) that would need to be installed into `~/.claude/hooks/` and `~/.claude/settings.json`.

### 6.2 Codex (`~/.codex/`)

| Component | Path | Notes |
|-----------|------|-------|
| Config | `~/.codex/config.toml` | Model: gpt-5.5, trusted projects include `wiki-memory` |
| Skills | `~/.codex/skills/` | ~95 symlinked skills |
| Memories | `~/.codex/memories/` | 3 subdirs: .agents, .codex, .git (empty) |
| Rules | `~/.codex/rules/` | 1 file |
| **AGENTS.md** | `~/.codex/AGENTS.md -> ~/code/agents/CLAUDE.md` | Shared |

**Key for wiki-memory:** Codex's `config.toml` already has `~/code/wiki-memory` as a trusted project. The wiki-memory plugin hooks for Codex would target this directory.

### 6.3 Pi (`~/.pi/`)

| Component | Path | Notes |
|-----------|------|-------|
| Agent dir | `~/.pi/agent/` | Full agent deployment |
| Agent skills | `~/.pi/agent/skills/` | ~101 symlinked + some local |
| Agent extensions | `~/.pi/agent/extensions/` | 1 extension (astro.charter) |
| Settings | `~/.pi/agent/settings.json` | Model: deepseek-v4-flash, provider: opencode-go |
| Agents dir | `~/.pi/agent/agents/` | 20+ astro.* agent definitions |
| Wiki symlink | `~/.pi/wiki -> ~/.local/share/ai-wiki` | Already pointing to shared wiki |
| Plugins | `~/.pi/agent/plugins/` | 10 packages loaded |

**Key for wiki-memory:** Pi's wiki is already symlinked to `~/.local/share/ai-wiki`. The wiki-memory project has `~/.pi/extensions/wiki-memory-hooks.ts` for Pi-specific hooks. Pi would need the least configuration change — it's already pointing at the shared wiki target.

### 6.4 Ante (`~/.ante/`)

| Component | Path | Notes |
|-----------|------|-------|
| Settings | `~/.ante/settings.json` | Hooks defined, memory db at `~/ai-wiki/.meta/ante-memory.db` |
| Agents | `~/.ante/agents/` | 5 agent definitions (architect, writer, researcher, etc.) |
| Hooks | `~/.ante/hooks/` | pre_compact.py, session_end.py, block-danger.sh |
| Memory | `~/.ante/memory/ante-memory.db` | SQLite memory database |

**Key finding:** Ante's memory.db is ALREADY configured at `~/ai-wiki/.meta/ante-memory.db` (via `ai-wiki` symlink -> `~/.local/share/ai-wiki/.meta/ante-memory.db`). However, the actual file at `~/.local/share/ai-wiki/.meta/ante-memory.db` is only 2 bytes — essentially empty. This means the ante-memory.db path points to the shared wiki location but was never populated.

---

## 7. WIKI-MEMORY PROJECT — WHAT IT WOULD REPLACE

**Project path:** `/home/cheta/code/wiki-memory/`
**Last activity:** June 5-6, 2026 (recent)

### Components present:
- `plugin/plugin.json` — Plugin manifest targeting Claude Code, Pi, Hermes, Codex, Kilocode, Antigravity
- `skill/SKILL.md` — Skill definition (grade A, pi-community)
- `hooks/pre_compact.py` — Session context extraction before compaction
- `hooks/session_end.py` — Session transcript capture as raw wiki source
- `dream/dream_agent.py` — 42KB sleep-time compute agent
- `dream/scheduler.py` — 7KB scheduler
- `cli/install.sh` — Quick install wrapper → delegates to `setup.sh`
- `setup.sh` — 44KB full setup script
- `specs/` — 7 spec files (ARCHITECTURE, design, MASTER_SPEC, etc.)
- `.pi/extensions/wiki-memory-hooks.ts` — Pi-specific TypeScript hooks
- `.claude/settings.local.json` — Claude Code local settings

### Target data path: `~/.local/share/ai-wiki/` (or `~/ai-wiki`)

---

## 8. READINESS & RISK ASSESSMENT

### 8.1 What is already in place (readiness)

| Item | Status | Confidence |
|------|--------|------------|
| `~/code/wiki-memory` project | Fully present with hooks, dream agent, plugin, skill, install script | HIGH |
| `~/.local/share/ai-wiki/` directory | Already exists with pages, entities, concepts, sources | HIGH |
| `~/ai-wiki` symlink | Already points to `~/.local/share/ai-wiki` | HIGH |
| `~/.pi/wiki` symlink | Already points to `~/.local/share/ai-wiki` (Pi ready) | HIGH |
| Ante memory db path | Already configured to `~/ai-wiki/.meta/ante-memory.db` | MEDIUM |
| Codex trusts wiki-memory project | In `config.toml` as trusted | HIGH |
| Hermes `wiki_enabled=true` | Already configured, just points to wrong path | HIGH |

### 8.2 What needs to change (risks)

| Risk | Severity | Details |
|------|----------|---------|
| **1. Two wiki dirs exist** | HIGH | `~/wiki` (old personal wiki, real dir) and `~/ai-wiki -> ~/.local/share/ai-wiki` (new Karpathy wiki). Migration needed — content in `~/wiki` will be lost if simply replaced. |
| **2. Hermes config points to wrong wiki** | HIGH | `config.yaml` has `wiki_path: ~/wiki` — must be changed to `~/ai-wiki` or `~/.local/share/ai-wiki`. |
| **3. No wiki-memory skill deployed** | MEDIUM | The `code/wiki-memory/skill/SKILL.md` is not symlinked into any agent's skills directory. Must be added to Hermes, Claude, Codex, Pi. |
| **4. No wiki-memory hooks deployed** | MEDIUM | The hooks (pre_compact, session_end) exist in `code/wiki-memory/hooks/` but are NOT deployed to any agent's hooks directory except possibly Ante (which has its own pre-existing hooks). |
| **5. Claude Code settings need update** | MEDIUM | `plugin/plugin.json` defines SessionStart and PreCompact hooks — these must be installed into `~/.claude/settings.json` and `~/.claude/hooks/`. |
| **6. `~/memory/` contents orphaned** | MEDIUM | 18 files (projects, daily, agents, topics) in the old flat-file memory system. Need review for any salvageable content before replacing. |
| **7. Honcho config is dead but present** | LOW | `honcho.json` references a non-functional Honcho setup. Should be disabled/removed but doesn't block. |
| **8. Shared CLAUDE.md references** | LOW | Claude Code and Codex share `~/code/agents/CLAUDE.md`. If this file references old `~/memory` or `~/wiki` paths, those references would break. Must audit this file. |
| **9. Ante memory db is empty** | LOW | Path configured correctly at `~/ai-wiki/.meta/ante-memory.db` but the file is 2 bytes. Would need initialization. |
| **10. Rollback complexity** | MEDIUM | Changing `wiki_path` and adding hooks affects 4 agent systems simultaneously. A rollback plan must be prepared. |

### 8.3 Migration Path Summary

**Required changes for Karpathy-wiki replacement:**
1. Merge `~/wiki/` content into `~/.local/share/ai-wiki/pages/`
2. Change Hermes `config.yaml`: `wiki_path: ~/ai-wiki` (or `~/.local/share/ai-wiki`)
3. Symlink `code/wiki-memory/skill/SKILL.md` into: Hermes skills, Claude Code skills, Codex skills, Pi skills
4. Deploy `code/wiki-memory/hooks/pre_compact.py` and `session_end.py` to appropriate agent hooks dirs
5. Update Claude Code `settings.json` to add SessionStart/PreCompact hooks (from plugin.json)
6. Optionally: replace old `~/memory/` with a symlink to `~/.local/share/ai-wiki/pages/` or archive contents
7. Optionally: remove or disable Honcho configuration
8. Review and update `~/code/agents/CLAUDE.md` if it references old paths

---

## 9. SUMMARY OF KEY FINDINGS

1. **Two wiki paths conflict**: `~/wiki` (current Hermes wiki) vs `~/ai-wiki` (target wiki). Content migration needed.
2. **Honcho is dead**: Gateway stopped since June 7. No active Honcho integration. Safe to remove.
3. **Shared agent context is centralized**: Claude Code and Codex share `code/agents/agents/*`, `code/agents/CLAUDE.md`, and `code/agents/commands/`. Any changes there affect both.
4. **wiki-memory project is complete**: Code at `~/code/wiki-memory` has all components (plugin, skill, hooks, dream agent, install). Not yet deployed to any agent.
5. **Pi is already aligned**: `~/.pi/wiki -> ~/.local/share/ai-wiki`. Pi needs least migration effort.
6. **Ante already targets the right path**: `ante-memory.db` configured at `~/ai-wiki/.meta/ante-memory.db` but empty.
7. **Skill silos**: Each agent maintains its own `skills/` directory (163 in Hermes, 101 in Claude, 95 in Codex, 101 in Pi), all symlinked to the central `code/agents/skills/`.
8. **No data has been modified** during this audit — all information is observational.

---

*Report generated 2026-06-08 by Hermes Agent audit. No files modified.*
