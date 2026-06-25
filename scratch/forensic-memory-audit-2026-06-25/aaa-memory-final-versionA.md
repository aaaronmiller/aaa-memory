---
date: 2026-06-25
ver: 0.1.0
author: Codex
tags: [audit, aaa-memory, lineage, memory, cass]
---
# aaa-memory Final Version A

## Scope

This scratch file tracks the current `aaa-memory` repository as the active installed source of truth.

## Current Position

- Installed entrypoint resolves to `/home/cheta/code/aaa-memory/src/aaa_memory/__init__.py`.
- The repo now contains Cass-oriented docs and the Cass refresh loop has been added to the human-facing install/setup docs.
- The repository is no longer just a raw memory store. It is already acting as the synthesis layer between the older memory archive plan and the later wiki/Cass lineage.

## What It Does

- Provides the active `aaa-memory` Python package.
- Contains original memory-archive requirements and plans.
- Contains session-resume and unified PRD docs that preserve lineage intent.
- Contains live Cass integration guidance in README/SETUP.

## What It Does Not Yet Do

- It does not yet present a single clean end-state for the lineage.
- It still mixes archival memory-archive intent with later wiki-style and PRD-unification intent.
- It does not yet explain the divergence between intent and as-built state in a way a third party can audit without re-reading the whole repo.

## Evidence To Reuse

- `specs/001-memory-archive/spec.md`
- `specs/001-memory-archive/plan.md`
- `docs/INTEGRATION_AUDIT.md`
- `docs/SESSION-RESUME.md`
- `docs/UNIFIED_PRD.md`
- `src/aaa_memory/agent_integration/README.md`
- `README.md`
- `SETUP.md`

## Open Questions

- Which parts of the current README/SETUP changes should be treated as a permanent shift versus temporary doc repair?
- Should Cass be documented as a required dependency for install, or as an optional but recommended forensic tier?
- Is the long-term target a memory system that preserves Karpathy-wiki semantics, or a more general session-search memory with wiki-style refinement?

## Deeper Read

The original `aaa-memory` intent is visible in `specs/001-memory-archive/spec.md` and `specs/001-memory-archive/plan.md`: a personal AI interaction archive with extraction, metadata, embeddings, progressive disclosure, intent routing, and tier transitions. That project is about durable memory infrastructure first, and wiki behavior second. It wants the system to remember sessions, recover abandoned work, and preserve context across time.

The later docs show the repo evolved into a synthesis layer. `docs/SESSION-RESUME.md` and `docs/UNIFIED_PRD.md` are not just product docs; they are lineage evidence. They show the repo being used to reconcile multiple memory concepts into one packaging surface. The current README and SETUP then push further by making Cass a required raw-session evidence tier, which is a real architectural choice and not a cosmetic doc tweak.

From an audit standpoint, the positive divergence is that the project now has a stronger retrieval substrate and better operational evidence handling. The negative divergence is that the repo now has multiple overlapping claims about what it is, and those claims are not yet normalized into one unambiguous target architecture.

## Cass Read

Cass fits `aaa-memory` best as the forensic evidence layer beneath durable memory. The current repo changes already reflect that: Cass is installed, triaged, and refreshed by cron, while the durable memory program remains the top-level project.

The Cass docs and repo show three properties that make it useful here:

1. It gives fast local session search across many agent histories.
2. It provides robot-friendly JSON and preflight surfaces.
3. It supports watch/index refresh behavior that makes evidence current without replacing the memory archive itself.

That means Cass adds utility if it is treated as an input and retrieval tier, not as the memory system itself.

## Current Judgment

`aaa-memory` is the right place to anchor the final project name, but not the right place to leave the architecture in its current mixed state. The final system should absorb the useful wiki mechanics and Cass evidence search, then republish itself as one coherent memory program.

## Current Doc Delta

The current README and SETUP changes are functionally important. They do three things:

- make Cass required for the install process,
- formalize a six-hour index refresh cron job,
- and teach the user to treat `cass triage` / `cass search --robot` as the entry points for raw session evidence.

That is a good divergence if the final project is meant to keep raw session evidence fresh and searchable. It is a bad divergence if it becomes another isolated install note that is not reflected in the architecture docs.
