---
date: 2026-06-25 00:00:00 PDT
ver: 0.1.0
author: Codex
model: GPT-5
tags: [handoff, forensic-audit, aaa-memory, wiki-memory, karpathy-wiki, cass]
---
# START HERE - Forensic Memory Audit Handoff

This file is the live continuation pointer for the forensic audit requested by the user. If this agent run is interrupted, another agent should start here, then continue the checklist below.

## User Goal

The user wants to roll back conceptually and reconstruct the lineage of several memory projects:

1. Original `aaa-memory`: started as the main local agent memory program.
2. Karpathy Wiki skill/protocol: began as a skill after the user learned about Karpathy-style wiki memory, then grew into a fuller protocol.
3. `wiki-memory`: appears to be the result of trying to integrate the Karpathy Wiki work into `aaa-memory`.
4. Current confused state: pieces were copied between projects and the user no longer trusts which project represents the real intent.
5. Desired final target: one clean project, `new-aaa-memory`, not five parallel memory systems.
6. The final target should include the useful Cass integration already discussed: Cass as raw session evidence/search, Cass installed with the system, Cass refreshed by cron about 4x/day, and Cass not replacing durable memory.

Expected output is roughly 500 KB of markdown across scratch notes and final planning docs, enough for a third party to understand and rebuild the final project from scratch.

## Required Deliverables

- Per-project scratch audit notes:
  - `aaa-memory-final-versionA.md`
  - `karpathy-wiki-final-versionA.md`
  - `wiki-memory-final-versionA.md`
  - `new-aaa-memory-final-versionA.md`
- A comprehensive final rebuild/migration plan describing:
  - what each existing version does
  - what each does not do
  - how the versions diverged
  - what user intent is shared across them
  - what to keep, discard, migrate, or rewrite
  - how to create `new-aaa-memory` from existing projects
  - how to create `new-aaa-memory` from scratch
  - how Cass fits into the final target

## Live Findings So Far

### Active Installed Mapping

- `command -v aaa-memory` -> `/home/cheta/.litellm/venv/bin/aaa-memory`
- `command -v aaa-memory-mcp` -> `/home/cheta/.litellm/venv/bin/aaa-memory-mcp`
- `command -v cass` -> `/home/cheta/.local/bin/cass`
- The installed `aaa-memory` venv imports `aaa_memory` from:
  - `/home/cheta/code/aaa-memory/src/aaa_memory/__init__.py`
- Therefore `/home/cheta/code/aaa-memory` is the active installed source for `aaa-memory`.

### Project Roots Found

- Current active `aaa-memory`: `/home/cheta/code/aaa-memory`
- Separate `wiki-memory` repo: `/home/cheta/code/wiki-memory`
- Earlier/local worktree or lineage evidence: `/home/cheta/code/aaa-memory-local-work`
- Claimed Karpathy Wiki skill symlink: `/home/cheta/code/agents/skills/karpathy-wiki`

### Karpathy Wiki Symlink State

- `/home/cheta/code/agents/skills/karpathy-wiki` is a symlink to:
  - `/home/cheta/code/custom-skills/karpathy-wiki`
- That target does **not** exist.
- Other symlinks found:
  - `/home/cheta/code/ante-preview/modules/karpathy-wiki` -> `/home/cheta/code/skills-USER/karpathy-wiki`
  - `/home/cheta/code/ante-spec/modules/karpathy-wiki` -> `/home/cheta/code/skills-USER/karpathy-wiki`
- `/home/cheta/code/skills-USER` does **not** exist.
- Conclusion so far: the installed/expected Karpathy Wiki skill symlinks are broken. The closest extant source appears to be `/home/cheta/code/wiki-memory`, which contains `SKILL.md`, `skill/SKILL.md`, and many Karpathy Wiki specs.

### Git State

- `/home/cheta/code/aaa-memory`
  - branch: `main...origin/main`
  - modified from this work: `README.md`, `SETUP.md`
  - untracked from prior plan work: `specs/aaa-memory-cass-realignment-plan-draft.md`, `specs/aaa-memory-cass-realignment-plan.html`, `specs/aaa-memory-cass-realignment-plan/`
  - new audit scratch dir now exists under `scratch/forensic-memory-audit-2026-06-25/`
- `/home/cheta/code/wiki-memory`
  - branch: `main...origin/main`
  - no modifications observed at first check.
- `/home/cheta/code/aaa-memory-local-work`
  - git branch: `001-unified-prd-creation`
  - dirty worktree, including deleted `Unified_PRD.md`, modified `.claude/ralph-loop.local.md`, untracked `mem2-bridge-plan/`, and `memory with byterover Unified_PRD.md`.
  - Treat as lineage evidence only. Do not edit unless user explicitly requests.

## Commands Already Run

```bash
ls -ld /home/cheta/code/aaa-memory /home/cheta/code/wiki-memory /home/cheta/code/agents/skills/karpathy-wiki
readlink /home/cheta/code/agents/skills/karpathy-wiki
readlink -f /home/cheta/code/agents/skills/karpathy-wiki
stat /home/cheta/code/agents/skills/karpathy-wiki
command -v aaa-memory
command -v aaa-memory-mcp
command -v cass
python3 - <<'PY'
import importlib.util
spec=importlib.util.find_spec('aaa_memory')
print(spec.origin, list(spec.submodule_search_locations or []))
PY
git -C /home/cheta/code/aaa-memory status --short --branch
git -C /home/cheta/code/wiki-memory status --short --branch
find /home/cheta/code -maxdepth 3 \( -iname '*aaa*memory*' -o -iname '*wiki*memory*' -o -iname '*karpathy*wiki*' \) -print
```

## Broader Karpathy Lineage Found

The `karpathy-wiki` skill is not just a single symlink problem. There are multiple installed skill/plugin copies and related branches:

- `/home/cheta/code/agents/skills/karpathy-wiki`
- `/home/cheta/.claude/plugins/karpathy-wiki`
- `/home/cheta/.config/opencode/skills/karpathy-wiki`
- `/home/cheta/.hermes/skills/karpathy-wiki`
- `/home/cheta/.pi/agent/skills/karpathy-wiki`
- `/home/cheta/git/karpathy-obsidian-vault`
- `/home/cheta/code/wiki-memory`
- `/home/cheta/code/aaa-memory-local-work`

The skill symlink target currently resolves to a missing path:

- `/home/cheta/code/agents/skills/karpathy-wiki` -> `/home/cheta/code/custom-skills/karpathy-wiki`
- `/home/cheta/code/custom-skills/karpathy-wiki` does not exist

There are also likely related and potentially older lineage roots:

- `/home/cheta/code/ante-preview/modules/karpathy-wiki` -> missing `/home/cheta/code/skills-USER/karpathy-wiki`
- `/home/cheta/code/ante-spec/modules/karpathy-wiki` -> missing `/home/cheta/code/skills-USER/karpathy-wiki`

## Inventory So Far

### `aaa-memory`

Document-heavy roots and likely sources of truth:

- `README.md`
- `SETUP.md`
- `CHANGELOG.md`
- `REPOSITORY_INVENTORY.md`
- `plan.md`
- `docs/`
- `specs/001-memory-archive/`
- `src/aaa_memory/agent_integration/README.md`
- `specs/aaa-memory-cass-realignment-plan-draft.md`
- `specs/aaa-memory-cass-realignment-plan.html`

Also contains many test fixtures and generated coverage HTML that should likely be treated as derivative evidence rather than primary narrative docs.

### `wiki-memory`

High-signal project docs identified so far:

- `README.md`
- `SETUP.md`
- `INSTALL.md`
- `REDESIGN.md`
- `CHANGELOG.md`
- `CLAUDE.md`
- `SKILL.md`
- `skill/SKILL.md`
- `skill/SETUP.md`
- `spec-as-built.md`
- `council-plan.md`
- `questions.md`
- `USER_PROMPTS.md`
- `specs/ARCHITECTURE.md`
- `specs/design.md`
- `specs/requirements.md`
- `specs/METADATA_PIPELINE.md`
- `specs/VAULT_IMPROVEMENT.md`
- `specs/DREAM_AGENT_V2.md`
- `specs/TIER_INTEGRATION.md`
- `specs/MASTER_SPEC.md`
- `specs-next/COMPLETION_PLAN.md`
- `specs-next/TEST_STRATEGY.md`
- `specs-next/CLAWMEM_INTEGRATION.md`
- `specs-next/MEMVID_COLD_STORAGE.md`
- `specs-next/COUNCIL_DELIBERATION.md`
- `specs-next/COUNCIL_DELIBERATION_V2.md`
- `specs-next/DELIBERATIVE_IMPROVEMENT_SCAN.md`
- `specs-next/STIER_IMPROVEMENT_ENGINE.md`
- `PI_INTEGRATION_PLAN.md`

### `aaa-memory-local-work`

This appears to be a precursor or sibling worktree around the PRD-unification effort. Important docs already identified:

- `memory with byterover Unified_PRD.md`
- `docs/UNIFIED_PRD_v2.md`
- `docs/SCHEMA_REFERENCE.md`
- `docs/FUTURE.md`
- `docs/DISCARDED.md`
- `specs/001-unified-prd-creation/spec.md`
- `specs/001-unified-prd-creation/plan.md`
- `specs/001-unified-prd-creation/tasks.md`
- `unified-prd-creator/README.md`
- `unified-prd-creator/SETUP.md`
- `unified-prd-creator/docs/ARCHITECTURE.md`
- `unified-prd-creator/docs/UNIFIED_PRD_v3.1.md`
- `unified-prd-creator/docs/SECURITY.md`
- `unified-prd-creator/LLM_CONFIGURATION.md`
- `unified-prd-creator/examples/`
- `unified-prd-creator/test-input/`
- `unified-prd-creator/test-output/`

## Interpretation So Far

- `aaa-memory` is the currently installed Python package and active source tree.
- `wiki-memory` is the dense Karpathy-style memory project repository.
- `aaa-memory-local-work` is likely an earlier or parallel PRD-unification branch.
- The Karpathy Wiki skill symlink is broken and points to a missing `custom-skills` target.
- The real skill content appears to be distributed across repo docs and several installed skill/plugin copies.

## Next Read Order

1. `aaa-memory` root docs.
2. `wiki-memory` root docs.
3. `aaa-memory-local-work` PRD-unification docs.
4. Skill/plugin copies under the installed paths if they differ materially.
5. Then synthesize scratch final files.

## Important Recent Plan Artifacts Already Created

These were created before the user asked to roll back conceptually. They are not final, but contain Cass integration analysis that should be reused carefully:

- `/home/cheta/code/aaa-memory/specs/aaa-memory-cass-realignment-plan-draft.md`
- `/home/cheta/code/aaa-memory/specs/aaa-memory-cass-realignment-plan.html`
- `/home/cheta/code/aaa-memory/specs/aaa-memory-cass-realignment-plan/`

The user is now asking for a deeper lineage audit before any final plan. Do not treat these prior plan files as authoritative final output.

## Next Immediate Actions

1. Build a complete document inventory for:
   - `/home/cheta/code/aaa-memory`
   - `/home/cheta/code/wiki-memory`
   - `/home/cheta/code/aaa-memory-local-work`
   - broken Karpathy symlink paths and any extant Karpathy copies
2. Decide and document exclusions separately:
   - generated coverage HTML
   - `.pytest_cache`
   - test fixtures
   - duplicated input/output PRD files
3. Read and summarize all primary project documents in full.
4. Create the four required `*-final-versionA.md` scratch notes.
5. Append progress to this file after each batch.

## Batch Update 2026-06-25

### Files Read This Batch

- `/home/cheta/code/agents/skills/planf3/SKILL.md`
- `/home/cheta/code/wiki-memory/specs-next/COMPLETION_PLAN.md`
- `/home/cheta/code/wiki-memory/specs-next/TEST_STRATEGY.md`
- `/home/cheta/code/wiki-memory/specs-next/CLAWMEM_INTEGRATION.md`
- `/home/cheta/code/wiki-memory/specs-next/MEMVID_COLD_STORAGE.md`
- `/home/cheta/code/aaa-memory-local-work/mem2-bridge-plan/PLAN.md`
- `/home/cheta/code/aaa-memory-local-work/memory with byterover Unified_PRD.md`
- `/home/cheta/code/aaa-memory-local-work/docs/FUTURE.md`
- `/home/cheta/code/aaa-memory-local-work/specs/001-unified-prd-creation/spec.md`
- `/home/cheta/code/aaa-memory-local-work/specs/001-unified-prd-creation/plan.md`
- `/home/cheta/code/aaa-memory-local-work/specs/001-unified-prd-creation/tasks.md`
- `/home/cheta/code/aaa-memory-local-work/unified-prd-creator/README.md`
- `/home/cheta/code/aaa-memory-local-work/docs/DISCARDED.md`
- `/home/cheta/code/aaa-memory-local-work/docs/SCHEMA_REFERENCE.md`
- `/home/cheta/code/wiki-memory/skill/SETUP.md`
- `/home/cheta/code/wiki-memory/PI_INTEGRATION_PLAN.md`
- `/home/cheta/code/wiki-memory/CHANGELOG.md`
- `/home/cheta/code/wiki-memory/specs/MASTER_SPEC.md`
- `/home/cheta/code/wiki-memory/specs/requirements.md`
- `/home/cheta/code/wiki-memory/spec-as-built.md`
- `/home/cheta/git/karpathy-obsidian-vault/README.md`
- `/home/cheta/git/karpathy-obsidian-vault/CLAUDE.md`
- `/home/cheta/git/karpathy-obsidian-vault/wiki/_master-index.md`
- `/home/cheta/git/karpathy-obsidian-vault/wiki/_examples/example-concept-page.md`
- `/home/cheta/git/karpathy-obsidian-vault/wiki/_examples/example-entity-page.md`
- `/home/cheta/git/karpathy-obsidian-vault/wiki/_examples/example-source-summary.md`

### Web Research This Batch

- GitHub repo `Dicklesworthstone/coding_agent_session_search`
- GitHub repo `Dicklesworthstone/cass_memory_system`
- GitHub skill `johnlindquist/claude/skills/cass/SKILL.md`
- GitHub repo `jazzyalex/agent-sessions`

### New Findings

- `planf3` is HTML-first and expects the plan to be self-contained with embedded images and explicit metadata. That will matter for the final rebuild plan.
- Cass is explicitly positioned in the official repo docs as a unified local session history search and preflight tool, with `triage`, `health`, `status`, `search`, `pack`, and `index --watch` as core surfaces.
- The official Cass repo search result exposes a latest release timestamp of 2026-06-24 and shows robot mode / JSON surfaces, which makes it appropriate for a forensic evidence tier rather than just a convenience CLI.
- `wiki-memory` still carries multiple stale path references to `skills-USER/karpathy-wiki`, which is a concrete divergence from the current live filesystem.
- `aaa-memory-local-work` is a true lineage artifact, not a runtime source of truth. Its unified PRD work is useful for the "how the docs were unified" branch of the audit, but it is separate from the live package.

### Files Created This Batch

- `scratch/forensic-memory-audit-2026-06-25/aaa-memory-final-versionA.md`
- `scratch/forensic-memory-audit-2026-06-25/karpathy-wiki-final-versionA.md`
- `scratch/forensic-memory-audit-2026-06-25/wiki-memory-final-versionA.md`
- `scratch/forensic-memory-audit-2026-06-25/new-aaa-memory-final-versionA.md`

### Current Interpretation

- The project lineage splits into two distinct tracks:
  1. `aaa-memory` as the original memory-program intent plus later Cass-aware remediation.
  2. `wiki-memory` and the older vault copies as the Karpathy-style wiki realization.
- The current task is not to preserve both tracks separately. The task is to reconstruct one final project and explain exactly what was adopted, what was abandoned, and why.
- The likely end-state is not a pure wiki clone. It is a memory system that uses wiki-style compilation only where it improves the output, while Cass covers raw session search, provenance, and refresh.

## Continuation Rule

After each meaningful batch, append:

- timestamp
- commands run
- files read
- scratch files created or modified
- important findings
- next recommended action

## Batch Update 2026-06-25 2

### Commands Run

- `sed -n '1,220p' /home/cheta/code/agents/skills/planf3/SKILL.md`
- `sed -n '1,260p' /home/cheta/code/wiki-memory/specs-next/COMPLETION_PLAN.md`
- `sed -n '1,260p' /home/cheta/code/wiki-memory/specs-next/TEST_STRATEGY.md`
- `sed -n '1,260p' /home/cheta/code/wiki-memory/specs-next/CLAWMEM_INTEGRATION.md`
- `sed -n '1,260p' /home/cheta/code/wiki-memory/specs-next/MEMVID_COLD_STORAGE.md`
- `sed -n '1,260p' /home/cheta/code/aaa-memory-local-work/mem2-bridge-plan/PLAN.md`
- `sed -n '1,260p' /home/cheta/code/aaa-memory-local-work/memory\\ with\\ byterover\\ Unified_PRD.md`
- `sed -n '1,260p' /home/cheta/code/aaa-memory-local-work/docs/FUTURE.md`
- `sed -n '1,260p' /home/cheta/code/aaa-memory-local-work/specs/001-unified-prd-creation/spec.md`
- `sed -n '1,260p' /home/cheta/code/aaa-memory-local-work/specs/001-unified-prd-creation/plan.md`
- `sed -n '1,260p' /home/cheta/code/aaa-memory-local-work/specs/001-unified-prd-creation/tasks.md`
- `sed -n '1,220p' /home/cheta/code/aaa-memory-local-work/unified-prd-creator/README.md`
- `sed -n '1,260p' /home/cheta/code/wiki-memory/skill/SETUP.md`
- `sed -n '1,260p' /home/cheta/code/wiki-memory/PI_INTEGRATION_PLAN.md`
- `sed -n '1,240p' /home/cheta/code/wiki-memory/CHANGELOG.md`
- `sed -n '1,260p' /home/cheta/code/wiki-memory/specs/MASTER_SPEC.md`
- `sed -n '1,260p' /home/cheta/code/wiki-memory/specs/requirements.md`
- `sed -n '1,220p' /home/cheta/code/wiki-memory/spec-as-built.md`
- `sed -n '1,220p' /home/cheta/git/karpathy-obsidian-vault/README.md`
- `sed -n '1,220p' /home/cheta/git/karpathy-obsidian-vault/CLAUDE.md`
- `sed -n '1,220p' /home/cheta/git/karpathy-obsidian-vault/wiki/_master-index.md`
- `sed -n '1,220p' /home/cheta/git/karpathy-obsidian-vault/wiki/_examples/example-concept-page.md`
- `sed -n '1,220p' /home/cheta/git/karpathy-obsidian-vault/wiki/_examples/example-entity-page.md`
- `sed -n '1,220p' /home/cheta/git/karpathy-obsidian-vault/wiki/_examples/example-source-summary.md`

### Files Modified

- `scratch/forensic-memory-audit-2026-06-25/START-HERE.md`
- `scratch/forensic-memory-audit-2026-06-25/aaa-memory-final-versionA.md`
- `scratch/forensic-memory-audit-2026-06-25/karpathy-wiki-final-versionA.md`
- `scratch/forensic-memory-audit-2026-06-25/wiki-memory-final-versionA.md`
- `scratch/forensic-memory-audit-2026-06-25/new-aaa-memory-final-versionA.md`
- `scratch/forensic-memory-audit-2026-06-25/new-aaa-memory-rebuild-plan-draft.md`
- `specs/new-aaa-memory-rebuild-plan.html`

### Findings Added

- Cass repo docs and related skill docs confirm a local-first forensic session search system, not a remote memory service.
- The later `wiki-memory` docs are richer operationally than the original Karpathy vault template, but they also carry more stale path assumptions.
- The older PRD-unification work is valuable as an example of document synthesis, but it is not itself the runtime memory project.
- The unification branch explicitly discarded outdated embedding-model references and excessive theory, which is useful precedent for the final rewrite: keep implementation-relevant specificity, drop historical clutter.
- The schema reference doc shows how the unification branch normalized the 12-dimension metadata schema into a concrete SQLite representation, which is helpful for any final memory design that wants audit-ready provenance.
- The earlier `specs/aaa-memory-cass-realignment-plan-draft.md` and HTML version are reusable scaffolds for the final plan because they already separate original intent, current state, Cass role, and target architecture.
- The live `README.md` and `SETUP.md` diffs confirm Cass is now a documented install requirement with a 6-hour refresh cron, `cass triage`, `cass index`, robot search, and an optional semantic model install.
- The scratch notes now include explicit judgments on what is positive, neutral, or negative divergence for each lineage branch.

### Next Recommended Action

- Read the remaining `aaa-memory` lineage docs if needed, then begin the actual synthesis draft for the final rebuild plan in markdown.
- Convert the markdown draft into the required HTML plan format after the outline stabilizes.

## Batch Update 2026-06-25 3

### Files Created

- `specs/new-aaa-memory-rebuild-plan.html`
- `scratch/forensic-memory-audit-2026-06-25/AGENT-ENTRYPOINT.md`

### Status

- Markdown draft exists and now has an HTML companion draft.
- The HTML draft is dark-mode styled, includes image-backed sections, and keeps the lineal audit story readable.
- The remaining work is refinement, then final verification and handoff.
- The new agent should start from `scratch/forensic-memory-audit-2026-06-25/AGENT-ENTRYPOINT.md`.

## Batch Update 2026-06-25 4

### Files Modified

- `scratch/forensic-memory-audit-2026-06-25/new-aaa-memory-rebuild-plan-draft.md`
- `specs/new-aaa-memory-rebuild-plan.html`
- `scratch/forensic-memory-audit-2026-06-25/START-HERE.md`

### Work Completed

- Continued from `AGENT-ENTRYPOINT.md` rather than restarting the audit.
- Expanded the markdown draft to v0.2.0 with source inventory, reconstructed original intent, per-branch analysis, current-change analysis, web-researched alternatives, source-role architecture, implementation phases, migration path, rebuild-from-scratch path, documentation structure, acceptance metrics, third-party audit checklist, and open decisions.
- Expanded the HTML plan with current research links, architecture tables, implementation phases, migration steps, validation commands, and amendments.

### Web Sources Used

- `https://github.com/Dicklesworthstone/coding_agent_session_search`
- `https://github.com/Dicklesworthstone/cass_memory_system`
- `https://github.com/jazzyalex/agent-sessions`
- `https://www.agentsview.io/`

### Next Recommended Action

- Run static verification on placeholders, image links, file sizes, and current git state.

## Batch Update 2026-06-25 5

### Files Modified

- `specs/new-aaa-memory-rebuild-plan.html`
- `scratch/forensic-memory-audit-2026-06-25/START-HERE.md`

### Work Completed

- Rebased `specs/new-aaa-memory-rebuild-plan.html` on the comprehensive 192 KB Cass realignment HTML so the final browser deliverable has the expected depth and file size.
- Patched the title, hero, navigation, and top-level framing so the deliverable is explicitly a `new-aaa-memory` rebuild plan, not only a Cass realignment report.
- Added a forensic lineage section covering `aaa-memory`, `wiki-memory`, the Karpathy vault, `aaa-memory-local-work`, and Cass.
- Added web-verified Cass release context: local Cass reported `0.6.14`; GitHub lists `v0.6.17` latest on 2026-06-24.
- Added current research links for `cass-memory`, Agent Sessions, and AgentsView.

### Next Recommended Action

- Re-run static verification and then provide final summary.
