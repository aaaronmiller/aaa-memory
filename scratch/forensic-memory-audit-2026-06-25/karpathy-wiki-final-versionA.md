---
date: 2026-06-25
ver: 0.1.0
author: Codex
tags: [audit, karpathy-wiki, lineage, wiki, dream-agent]
---
# Karpathy Wiki Final Version A

## Scope

This scratch file tracks the standalone Karpathy-wiki lineage as represented by the installed repo copies and the older vault template.

## Current Position

- The Karpathy wiki lineage is strongly represented in `wiki-memory`.
- The live skill path is broken and points at a missing `skills-USER/karpathy-wiki` target.
- The older vault template and the wiki-memory repo preserve the original wiki pattern better than `aaa-memory` does.

## What It Does

- Defines a Karpathy-style raw -> wiki compilation workflow.
- Uses dream-agent / idle compute ideas heavily.
- Preserves provenance, cross-links, and sleep-time improvements.
- Includes Pi and multi-agent installation intent.

## What It Does Not Yet Do

- It does not resolve its own path drift.
- It still assumes `skills-USER/karpathy-wiki` in multiple places.
- It does not yet acknowledge Cass as a first-class session-evidence layer.

## Evidence To Reuse

- `/home/cheta/git/karpathy-obsidian-vault/README.md`
- `/home/cheta/git/karpathy-obsidian-vault/CLAUDE.md`
- `/home/cheta/git/karpathy-obsidian-vault/wiki/_master-index.md`
- `/home/cheta/git/karpathy-obsidian-vault/wiki/_examples/example-concept-page.md`
- `/home/cheta/code/wiki-memory/spec-as-built.md`
- `/home/cheta/code/wiki-memory/specs/MASTER_SPEC.md`
- `/home/cheta/code/wiki-memory/specs/requirements.md`
- `/home/cheta/code/wiki-memory/skill/SKILL.md`
- `/home/cheta/code/wiki-memory/skill/SETUP.md`
- `/home/cheta/code/wiki-memory/PI_INTEGRATION_PLAN.md`

## Open Questions

- Should the Karpathy-wiki semantics be preserved as a subset inside the final project, or just mined for useful mechanics?
- Which parts of the dream agent are still correct once Cass becomes the raw evidence source?

## Deeper Read

The Karpathy lineage has two layers. The public-facing layer is the vault template and the wiki-memory docs that explain a Karpathy-style compounding wiki. The operational layer is the installed skill/plugin wiring that tries to make the wiki available to multiple agents. Both are useful, but both are currently brittle because the expected skill path no longer resolves.

The old vault template is important because it shows the simplest Karpathy pattern before the project acquired a lot of sleep-time machinery. It is closer to the original idea of "raw -> wiki -> index -> query" than the later multi-tier memory stack. That simplicity is valuable because it establishes the baseline intent: a readable, incrementally compiled knowledge base.

The `wiki-memory` repo is more ambitious. It adds ClawMem, MemVid, dream-agent phases, pattern detection, council deliberation, and a lot of multi-agent integration. That expansion is not automatically bad. Some of it is clearly positive because it makes the system more operationally complete. But some of it is drift because it shifts the project away from the clean memory-program frame and toward a self-improving wiki platform with many adjacent subsystems.

## Cass Read

Cass does not belong as a replacement for the Karpathy wiki. It belongs as the searchable session evidence substrate that feeds the wiki. That is the cleanest synthesis: Cass collects and refreshes the raw history, while the wiki compiles and curates the durable layer.

## Current Judgment

The Karpathy branch is not wrong. It is just incomplete on its own and path-drifted in several places. Its best pieces should survive, but they should be subordinated to the single final memory program instead of remaining a separate project family.

## Current Doc Delta

The biggest live defect is path drift. The wiki docs still assume the old `skills-USER/karpathy-wiki` location, while the actual symlink in the workspace now points somewhere else entirely and resolves to a missing target. That means the Karpathy branch is currently valuable as a design source, but unsafe as an install source.
