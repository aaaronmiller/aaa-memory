---
date: 2026-06-25
ver: 0.1.0
author: Codex
tags: [audit, synthesis, new-aaa-memory, cass, rebuild]
---
# new-aaa-memory Final Version A

## Scope

This scratch file is the synthesis target: one project, one coherent memory system, built from the valid pieces of `aaa-memory`, `wiki-memory`, the Karpathy lineage, and Cass.

## Intended End State

- `new-aaa-memory` should be the only long-term project.
- It should preserve the original memory-program intent from `aaa-memory`.
- It should keep the useful wiki/dream mechanics where they improve quality.
- It should include Cass as the installed, refreshed raw session evidence/search tier.

## Core Question

The key audit question is not whether the current divergence is large. It is whether each divergence improves or degrades the final output quality relative to the original user intent.

## Working Thesis

- Positive divergences:
  - stronger provenance
  - broader session search
  - improved idle-time evidence refresh
  - more explicit install/runtime docs
- Negative divergences:
  - extra project fragmentation
  - stale symlink/path assumptions
  - multiple overlapping "source of truth" docs
  - architecture drift away from one unified memory system

## Required Synthesis Output

- original plan
- as-built changes
- divergence analysis
- Cass integration analysis
- keep / change / delete recommendations
- rebuild-from-scratch path
- migration path from current repos
- audit appendix for third parties

## Open Questions

- Should `new-aaa-memory` be implemented as a repo rename, a super-repo, or a clean rebuild with migration notes?
- Which current docs become canonical, and which become historical appendix only?

## Deeper Read

The best path forward is not to preserve every historical choice. It is to preserve the ones that improve output quality and remove the ones that add confusion without adding capability.

The likely canonical stack is:

1. `aaa-memory` as the root identity and memory-program namespace.
2. Cass as the local session-history evidence/search substrate.
3. Wiki-style compilation as the curated durable layer.
4. A single set of docs that explicitly separates original intent, current as-built reality, and proposed rebuild steps.

That stack works because it aligns the tools with the user intent. The user does not want five memory systems. They want one memory system that can explain itself, be audited, and be rebuilt from scratch.

## Divergence Rule

Any divergence from the original plan should be classified as one of three things:

- Positive: it improves durability, searchability, provenance, or rebuildability.
- Neutral: it changes shape but not outcome.
- Negative: it fragments the project, obscures lineage, or makes the system harder to audit.

That classification is the key lens for the final rebuild plan.
