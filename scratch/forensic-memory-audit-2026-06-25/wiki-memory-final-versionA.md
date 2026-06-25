---
date: 2026-06-25
ver: 0.1.0
author: Codex
tags: [audit, wiki-memory, lineage, clawmem, memvid]
---
# wiki-memory Final Version A

## Scope

This scratch file tracks the `wiki-memory` repo as the current Karpathy-style memory project implementation.

## Current Position

- `wiki-memory` is the densest and most explicit Karpathy-wiki implementation in the workspace.
- Its specs strongly define a three-tier system with dream agent, ClawMem, MemVid, and improvement loops.
- It still reflects an architecture that is not the same as the user's original `aaa-memory` goal.

## What It Does

- Defines the memory wiki architecture in detail.
- Captures installation, skill, plugin, and Pi integration intent.
- Documents hot/warm/cold tiers, idle compute, and automatic skill creation.
- Preserves several higher-order plans: council, MemVid, S-tier improvements.

## What It Does Not Yet Do

- It does not reconcile its architecture back into `aaa-memory`.
- It still references stale symlink targets and older path conventions.
- It does not incorporate Cass as the raw session search and evidence index.

## Evidence To Reuse

- `README.md`
- `SETUP.md`
- `SKILL.md`
- `skill/SKILL.md`
- `skill/SETUP.md`
- `spec-as-built.md`
- `specs/ARCHITECTURE.md`
- `specs/requirements.md`
- `specs/MASTER_SPEC.md`
- `specs-next/COMPLETION_PLAN.md`
- `specs-next/TEST_STRATEGY.md`
- `specs-next/CLAWMEM_INTEGRATION.md`
- `specs-next/MEMVID_COLD_STORAGE.md`
- `PI_INTEGRATION_PLAN.md`
- `CHANGELOG.md`

## Open Questions

- Which components are intended architecture versus speculative expansion?
- Which docs should be considered superseded by the current `aaa-memory` repo changes?
- Does Cass replace anything in the hot tier, or only strengthen retrieval and auditability around it?

## Deeper Read

`wiki-memory` reads like the most complete expression of the Karpathy-style dream. It has the clearest articulation of the sleep-time compile loop, the strongest language around provenance and quality compounding, and the richest operational docs. If someone wanted to understand how the Karpathy wiki ecosystem was supposed to work, this repo is the best single source.

The downside is that the repo contains a lot of specification surface that is not yet stabilized against the current runtime reality. It still assumes old skill paths and older installation assumptions. It also carries several "future" ideas as if they are already part of the same product, which makes it harder for a third party to tell which pieces are essential and which pieces are aspirational.

The most important positive divergence in `wiki-memory` is the move away from vague memory talk into concrete operational surfaces: setup instructions, skill wiring, scheduler behavior, dream agent phases, and data layout. The most important negative divergence is that it has become a second source of truth competing with `aaa-memory`.

## Cass Read

Cass adds leverage here because it resolves the gap between "what happened in sessions" and "what the wiki compiled." The wiki can remain the durable curated layer, while Cass can answer the forensic question of how a session unfolded across tools.

## Current Judgment

`wiki-memory` should be treated as an implementation reservoir, not the final project name. It is valuable as a source of good mechanics, but the final shape should collapse back into one project with one documentation surface.

## Current Doc Delta

The docs are internally rich but externally inconsistent. The repo describes a strong multi-tier memory stack, yet multiple files still point at stale skill-paths and older installation assumptions. That makes it good source material for a rebuild plan and bad final-state documentation unless normalized.
