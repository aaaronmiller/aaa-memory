# Topic: Sleep-Time Compute & Self-Improvement Loops

## Summary
The autonomous background processing system that runs during idle/sleep periods to refine skills, process memories, and improve the system without human intervention.

---

## Overview

> **Source:** [gemini-prd.md](../gemini-prd.md) (Sections 1.5, 3.3)

Sleep-Time Compute is the engine of self-improvement. It operates autonomously to upgrade the system's intelligence during idle periods, using free/cheap models via OpenRouter.

---

## Sleep-Time Daemon Architecture

> **Source:** [gemini-prd.md](../gemini-prd.md) (Section 3.3)

A background service (`launchd` on macOS) with a state machine:

| State | Trigger | Action |
|-------|---------|--------|
| IDLE | Default | Monitoring system load |
| DREAMING | Daily | Processing ByteRover inbox → Graphiti |
| EVOLVING | Nightly | Full refinement loop (5 steps) |

### EVOLVING Steps
1. **Curiosity Module** generates task list (identifies knowledge gaps)
2. **Creator** generates artifacts via OpenRouter (free models)
3. **Tribunal** (Parallel Async) critiques artifacts
4. **Mutator** updates `~/.skills/*.md` files
5. **Archivist** renders approved artifacts to `~/.memvid/staging`

---

## Loop 1: The Simulator (Correction)

> **Source:** [gemini-prd.md](../gemini-prd.md) (Section 1.5)

- **Input:** Failed tests/specs from the day's active work
- **Action:** Spawns a temporary git branch. Retries the failed spec using infinite time/retries.
- **Result:** Upon success, creates a "Solution Node" in Graphiti
- **Purpose:** Automatically fixes failures encountered during the day

---

## Loop 2: The Professor (Synthesis)

> **Source:** [gemini-prd.md](../gemini-prd.md) (Section 1.5)

- **Input:** High-quality external repositories (e.g., `shadcn/ui`, `actix-web`)
- **Action:** "Reverse Engineers" the code to generate Synthetic PRDs
- **Result:** Stores pairs of `{Synthetic_PRD} -> {Perfect_Code}` in MemVid for future RAG retrieval
- **Purpose:** Learns from exemplary codebases

---

## Loop 3: The Evolutionary Forge (Creation)

> **Source:** [gemini-prd.md](../gemini-prd.md) (Section 1.5)

- **Input:** "Madlib" Inspiration Queue (Randomized Topic + Style + Constraint)
- **Action:**
  1. **Draft:** Creator Model generates artifact
  2. **Gate:** Taste Oracle checks novelty (rejects if too similar/dissimilar to Gold Standard)
  3. **Critique:** Tribunal (Personas) attacks the draft
  4. **Mutate:** If score < 95, Mutator rewrites the Skill File (Prompt)
- **Result:** A graduated "Skill File" v2.0 and a high-quality artifact for the archive

---

## The Tribunal (The Critic)

> **Source:** [gemini-prd.md](../gemini-prd.md), [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md)

A dynamic graph of adversarial personas that critique generated artifacts:
- **Security Zealot** — Attacks security vulnerabilities
- **Pedant** — Checks correctness and precision
- **Visionary** — Evaluates innovation and forward-thinking

Runs in parallel async during sleep cycles.

---

## The Mutator (The Evolution)

> **Source:** [gemini-prd.md](../gemini-prd.md), [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md)

- Uses Genetic Algorithms to rewrite "Skill Files" (prompts)
- Based on Tribunal feedback scores
- Skill Files stored at `~/.skills/*.md`

---

## The Taste Oracle (The Quality Gate)

> **Source:** [gemini-prd.md](../gemini-prd.md), [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md)

- Vector-based novelty detector
- Compares outputs against a "Gold Standard" baseline in MemVid
- Rejects derivative or hallucinated work
- Uses cosine similarity in embedding space

---

## Cost Strategy

> **Source:** [gemini-prd.md](../gemini-prd.md) (Section 1.2)

- Uses Free/OpenRouter tiers for "Heavy Hitter" models during sleep cycles
- Models: DeepSeek, Qwen, Mistral (free tier)
- Zero-cost autonomous improvement

---

## Implementation Requirements

1. Implement sleep-time daemon (background service)
2. Build state machine (IDLE → DREAMING → EVOLVING)
3. Implement idle detection (system idle > 15 minutes)
4. Build the Simulator loop (failed test retry on temp branches)
5. Build the Professor loop (external repo analysis → synthetic PRDs)
6. Build the Evolutionary Forge loop (creation + critique + mutation)
7. Implement the Tribunal with configurable adversarial personas
8. Implement the Mutator with genetic algorithm-based prompt rewriting
9. Implement the Taste Oracle with vector novelty detection
10. Build the Curiosity Module (knowledge gap detection)
11. Create Skill File management system (`~/.skills/*.md`)
12. Integrate with OpenRouter free tier for sleep-time models

---

## Conflicts / Ambiguities

- **⚠️ Curiosity Module vs Madlib:** The Curiosity Module (Active Inference) is listed as a strategic integration in gemini-prd.md Section 2, replacing the random "Madlib" generator. But Loop 3 still references "Madlib Inspiration Queue." The Curiosity Module is the intended upgrade path.
- **⚠️ Score threshold:** Loop 3 uses "score < 95" as the mutation threshold. This seems very high — may need calibration.
- **⚠️ Platform dependency:** `launchd` is macOS-only. Needs cross-platform daemon support.
