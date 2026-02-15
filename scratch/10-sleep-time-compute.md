# Topic 10: Sleep-Time Compute & Self-Improvement Loops

## Summary
Implement the autonomous sleep-time daemon that runs three refinement loops (Simulator, Professor, Evolutionary Forge) plus the Tribunal critic system, Mutator evolution engine, and Taste Oracle quality gate.

---

## Overview
> Sources: `gemini-prd.md` (lines 76–113, 229–250), `docs/UNIFIED_PRD.md` (lines 58–60, 64–66)

The sleep-time compute system is the engine of self-improvement. It operates autonomously during idle periods to upgrade the system's intelligence, refine skills, and archive knowledge.

---

## Sleep-Time Daemon Architecture
> Source: `gemini-prd.md` (lines 229–250)

Background service (`launchd` on macOS) with a state machine:

| State | Trigger | Action |
|-------|---------|--------|
| `IDLE` | Default | Monitor system load |
| `DREAMING` (Daily) | Nightly or idle > 15m | Process ByteRover inbox → Graphiti |
| `EVOLVING` (Nightly) | After DREAMING completes | Run refinement loops |

### EVOLVING State Steps:
1. **Curiosity Module** generates task list (identifies knowledge gaps)
2. **Creator** generates artifacts via OpenRouter (free/cheap models)
3. **Tribunal** (Parallel Async) critiques generated artifacts
4. **Mutator** updates `~/.skills/*.md` skill files
5. **Archivist** renders approved artifacts to `~/.memvid/staging`

---

## Refinement Loops

### Loop 1: The Simulator (Correction)
> Source: `gemini-prd.md` (lines 80–86)

- **Input:** Failed tests/specs from the day's active work
- **Action:** Spawn a temporary git branch. Retry the failed spec using infinite time/retries
- **Result:** Upon success, create a "Solution Node" in Graphiti
- **Purpose:** Fix today's failures overnight

### Loop 2: The Professor (Synthesis)
> Source: `gemini-prd.md` (lines 89–95)

- **Input:** High-quality external repositories (e.g., `shadcn/ui`, `actix-web`)
- **Action:** "Reverse Engineer" the code to generate Synthetic PRDs
- **Result:** Store pairs of `{Synthetic_PRD} → {Perfect_Code}` in MemVid for future RAG retrieval
- **Purpose:** Learn from exemplary codebases

### Loop 3: The Evolutionary Forge (Creation)
> Source: `gemini-prd.md` (lines 98–112)

- **Input:** "Madlib" Inspiration Queue (Randomized Topic + Style + Constraint)
- **Action:**
  1. **Draft:** Creator Model generates artifact
  2. **Gate:** Taste Oracle checks novelty (rejects if too similar/dissimilar to Gold Standard)
  3. **Critique:** Tribunal (Personas) attacks the draft
  4. **Mutate:** If score < 95, Mutator rewrites the Skill File (Prompt)
- **Result:** A graduated "Skill File" v2.0 and a high-quality artifact for the archive

---

## Supporting Components

### The Tribunal (The Critic)
> Source: `gemini-prd.md` (lines 32, 108), `docs/UNIFIED_PRD.md` (line 58)

- Dynamic graph of adversarial personas that critique generated artifacts
- **Personas:** Security Zealot, Pedant, Visionary (and potentially more)
- Runs during sleep cycles
- Parallel async execution for speed
- Outputs: critique scores, specific feedback, pass/fail decisions

### The Mutator (The Evolution)
> Source: `gemini-prd.md` (lines 33, 110), `docs/UNIFIED_PRD.md` (line 59)

- Uses Genetic Algorithms to rewrite "Skill Files" (prompts)
- Based on Tribunal feedback
- Skill files stored at `~/.skills/*.md`
- Iterative improvement: v1.0 → v2.0 → v3.0...

### The Taste Oracle (The Quality Gate)
> Source: `gemini-prd.md` (lines 34, 106), `docs/UNIFIED_PRD.md` (line 60)

- Vector-based novelty detector
- Compares outputs against a "Gold Standard" baseline in MemVid
- Rejects derivative or hallucinated work
- Uses cosine similarity in embedding space
- Threshold: rejects if too similar (derivative) OR too dissimilar (hallucinated)

---

## Cost Efficiency
> Source: `gemini-prd.md` (lines 19)

- Utilizes Free/OpenRouter tiers for "Heavy Hitter" models during sleep cycles
- Models: DeepSeek, Qwen, Mistral (free tiers)
- Zero-cost operation during sleep time

---

## Implementation Tasks

1. Create `src/sleep/daemon.py` — Main sleep-time daemon with state machine (IDLE/DREAMING/EVOLVING)
2. Create `src/sleep/simulator.py` — Loop 1: Failed test retry on temporary branches
3. Create `src/sleep/professor.py` — Loop 2: Reverse-engineer external repos into synthetic PRDs
4. Create `src/sleep/forge.py` — Loop 3: Evolutionary creation with madlib inspiration
5. Create `src/sleep/tribunal.py` — Adversarial persona critique system (parallel async)
6. Create `src/sleep/mutator.py` — Genetic algorithm skill file rewriter
7. Create `src/sleep/taste_oracle.py` — Vector-based novelty detection against gold standard
8. Create `src/sleep/curiosity_module.py` — Knowledge gap detection and task generation
9. Create `src/sleep/archivist.py` — Render approved artifacts to MemVid staging
10. Create launchd plist for macOS daemon registration

---

## Conflicts & Ambiguities

1. **Curiosity Module source:** `gemini-prd.md` (line 241) mentions the Curiosity Module in the EVOLVING state, but the detailed description (lines 139–145) is under "Strategic Integrations" as "Active Inference Curiosity Module." It's unclear whether the basic version (madlib generator) or the advanced version (Free Energy-based) should be implemented first.

2. **Tribunal persona count:** The PRD mentions 3 personas (Security Zealot, Pedant, Visionary) but says it's a "dynamic graph" suggesting more can be added. The initial implementation should support a configurable set of personas.

3. **Score threshold:** The Evolutionary Forge uses "score < 95" as the mutation trigger, but the Taste Oracle's similarity thresholds are not specified numerically. Need to define concrete thresholds for both.

4. **macOS-specific:** The daemon uses `launchd` which is macOS-specific. For cross-platform support, consider `systemd` (Linux) or a generic process manager.
