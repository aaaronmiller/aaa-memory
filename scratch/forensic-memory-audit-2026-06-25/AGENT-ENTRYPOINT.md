---
date: 2026-06-25 00:00:00 PDT
ver: 0.2.0
author: Codex
model: GPT-5
tags: [handoff, forensic-audit, aaa-memory, wiki-memory, karpathy-wiki, cass]
---
# Agent Entry Point

Use this file if you are the next agent continuing the forensic audit and rebuild-plan work.

## Primary Objective

Reconstruct one clean final project, `new-aaa-memory`, from the lineage below:

1. original `aaa-memory` memory-program intent,
2. Karpathy-style wiki and dream-agent mechanics,
3. `wiki-memory` as the richest operational branch,
4. `aaa-memory-local-work` as PRD-unification lineage evidence,
5. Cass as the raw session-history evidence/search tier.

The user does not want five projects. The final output must explain what each existing version does, does not do, and how the final project should be rebuilt from scratch or migrated from current repos.

## What Is Already Verified

### Runtime and installed paths

- `command -v aaa-memory` -> `/home/cheta/.litellm/venv/bin/aaa-memory`
- `command -v aaa-memory-mcp` -> `/home/cheta/.litellm/venv/bin/aaa-memory-mcp`
- `command -v cass` -> `/home/cheta/.local/bin/cass`
- installed `aaa_memory` package source -> `/home/cheta/code/aaa-memory/src/aaa_memory/__init__.py`

### Main repos

- `aaa-memory` -> `/home/cheta/code/aaa-memory`
- `wiki-memory` -> `/home/cheta/code/wiki-memory`
- `aaa-memory-local-work` -> `/home/cheta/code/aaa-memory-local-work`
- older Karpathy vault -> `/home/cheta/git/karpathy-obsidian-vault`

### Broken path state

- `/home/cheta/code/agents/skills/karpathy-wiki` -> `/home/cheta/code/custom-skills/karpathy-wiki` and the target does not exist
- `wiki-memory/skill/SETUP.md` and multiple other docs still reference `skills-USER/karpathy-wiki`
- the current Karpathy skill path assumptions are stale and should not be treated as live install truth

## Cass Status

Cass is now part of the `aaa-memory` doc process.

Verified Cass role:

- local coding-agent session search and evidence packaging
- robot-friendly JSON / `--robot` surfaces
- `cass triage --json`
- `cass search "query" --robot --robot-meta`
- `cass pack "query" --robot --max-tokens ...`
- `cass index --json --no-progress-events --data-dir "$HOME/.local/share/coding-agent-search"`
- cron refresh every 6 hours, which is 4x/day

Cass should be treated as the raw evidence tier. It improves search and freshness, but it does not replace the durable memory store or the curated wiki layer.

## Files Already Created

### Scratch notes

- `scratch/forensic-memory-audit-2026-06-25/aaa-memory-final-versionA.md`
- `scratch/forensic-memory-audit-2026-06-25/karpathy-wiki-final-versionA.md`
- `scratch/forensic-memory-audit-2026-06-25/wiki-memory-final-versionA.md`
- `scratch/forensic-memory-audit-2026-06-25/new-aaa-memory-final-versionA.md`

### Draft plan files

- `scratch/forensic-memory-audit-2026-06-25/new-aaa-memory-rebuild-plan-draft.md`
- `specs/new-aaa-memory-rebuild-plan.html`

### Earlier Cass realignment artifacts

- `specs/aaa-memory-cass-realignment-plan-draft.md`
- `specs/aaa-memory-cass-realignment-plan.html`
- `specs/aaa-memory-cass-realignment-plan/`

## What The Current Repo Changes Mean

The user-facing edits in `README.md` and `SETUP.md` are real architectural changes:

- Cass is now documented as required for raw session evidence.
- Cass installation and `cass triage` / `cass index` are part of setup.
- A cron job refreshes Cass every 6 hours.
- `cass search --robot` and `cass pack --robot` are documented as the right non-interactive surfaces.
- semantic Cass models remain optional.

This is a positive divergence if the final project wants current raw evidence and forensic recall. It is negative only if the docs stop there and fail to normalize the rest of the architecture.

## Current Judgment Summary

### Positive divergences

- Cass as raw evidence tier
- better provenance and auditability
- better search freshness
- better install clarity
- stronger wiki compounding mechanics

### Negative divergences

- multiple source-of-truth claims
- stale `skills-USER` path assumptions
- repo fragmentation
- architecture sprawl
- unprioritized future ideas mixed into mainline docs

### Neutral divergences

- more formal documentation
- more explicit role separation
- more concrete config/install surfaces

## What Still Needs To Be Done

1. Finish the markdown synthesis in `scratch/forensic-memory-audit-2026-06-25/new-aaa-memory-rebuild-plan-draft.md`.
2. Tighten the HTML plan in `specs/new-aaa-memory-rebuild-plan.html` so it is the final browser-ready deliverable.
3. Make the final plan explicitly include:
   - original plan,
   - changes that were made,
   - divergence analysis,
   - Cass integration analysis,
   - keep/change/delete recommendations,
   - rebuild-from-scratch path,
   - migration path from current repos,
   - 8 user stories in an appendix.
4. Ensure the final deliverable is self-contained, readable, and auditable without needing to open the source repos first.

## Source Docs That Matter Most

Read these first if you need to continue the synthesis:

- `/home/cheta/code/aaa-memory/specs/001-memory-archive/spec.md`
- `/home/cheta/code/aaa-memory/specs/001-memory-archive/plan.md`
- `/home/cheta/code/aaa-memory/docs/INTEGRATION_AUDIT.md`
- `/home/cheta/code/aaa-memory/docs/SESSION-RESUME.md`
- `/home/cheta/code/aaa-memory/docs/UNIFIED_PRD.md`
- `/home/cheta/code/aaa-memory/src/aaa_memory/agent_integration/README.md`
- `/home/cheta/code/wiki-memory/README.md`
- `/home/cheta/code/wiki-memory/SETUP.md`
- `/home/cheta/code/wiki-memory/SKILL.md`
- `/home/cheta/code/wiki-memory/skill/SKILL.md`
- `/home/cheta/code/wiki-memory/skill/SETUP.md`
- `/home/cheta/code/wiki-memory/spec-as-built.md`
- `/home/cheta/code/wiki-memory/specs/ARCHITECTURE.md`
- `/home/cheta/code/wiki-memory/specs/requirements.md`
- `/home/cheta/code/wiki-memory/specs/MASTER_SPEC.md`
- `/home/cheta/code/wiki-memory/specs-next/COMPLETION_PLAN.md`
- `/home/cheta/code/wiki-memory/specs-next/TEST_STRATEGY.md`
- `/home/cheta/code/wiki-memory/specs-next/CLAWMEM_INTEGRATION.md`
- `/home/cheta/code/wiki-memory/specs-next/MEMVID_COLD_STORAGE.md`
- `/home/cheta/code/wiki-memory/PI_INTEGRATION_PLAN.md`
- `/home/cheta/code/aaa-memory-local-work/specs/001-unified-prd-creation/spec.md`
- `/home/cheta/code/aaa-memory-local-work/specs/001-unified-prd-creation/plan.md`
- `/home/cheta/code/aaa-memory-local-work/specs/001-unified-prd-creation/tasks.md`
- `/home/cheta/code/aaa-memory-local-work/docs/DISCARDED.md`
- `/home/cheta/code/aaa-memory-local-work/docs/FUTURE.md`
- `/home/cheta/code/aaa-memory-local-work/docs/SCHEMA_REFERENCE.md`
- `/home/cheta/code/aaa-memory-local-work/mem2-bridge-plan/PLAN.md`
- `/home/cheta/git/karpathy-obsidian-vault/README.md`
- `/home/cheta/git/karpathy-obsidian-vault/CLAUDE.md`

## Useful Existing Plan Artifact

The earlier Cass realignment draft already has a good structure:

- reconstructed original plan,
- evidence status table,
- current project state,
- divergence analysis,
- Cass investigation,
- target architecture,
- keep/change/delete,
- rebuild from current repos,
- rebuild from scratch,
- appendix user stories.

Use it as the conceptual scaffold, not as the final text.

## Commands That Help

```bash
git -C /home/cheta/code/aaa-memory status --short
git -C /home/cheta/code/aaa-memory diff -- README.md SETUP.md
sed -n '1,260p' /home/cheta/code/aaa-memory/scratch/forensic-memory-audit-2026-06-25/new-aaa-memory-rebuild-plan-draft.md
sed -n '1,260p' /home/cheta/code/aaa-memory/specs/new-aaa-memory-rebuild-plan.html
```

## Working Tree State

Current `aaa-memory` status at the time of handoff:

- modified: `README.md`
- modified: `SETUP.md`
- untracked: `scratch/`
- untracked: `specs/aaa-memory-cass-realignment-plan-draft.md`
- untracked: `specs/aaa-memory-cass-realignment-plan.html`
- untracked: `specs/aaa-memory-cass-realignment-plan/`
- untracked: `specs/new-aaa-memory-rebuild-plan.html`

## Continuation Rule

After each meaningful batch, append to `scratch/forensic-memory-audit-2026-06-25/START-HERE.md` and update this file if the entry point changes.

## Final Note

The next agent should not restart from scratch. They should finish the markdown/HTML synthesis, then verify the final plan against the source docs and the current repo diff.

## Continuation Update 2026-06-25 4

The markdown draft has been expanded to v0.2.0 and the HTML plan has been expanded with research, implementation phases, migration guidance, and an audit checklist. Remaining work is static verification, final cleanup, and final summary.
