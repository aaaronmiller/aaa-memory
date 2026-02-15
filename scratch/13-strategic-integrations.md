# Topic 13: Strategic Integrations (Advanced Features)

## Summary
Implement five advanced integrations that push the architecture from "Advanced" to "State-of-the-Art": Hypergraph Knowledge Representation, Active Inference Curiosity Module, Formal Verification Gate, Automated Model Merging, and Contrastive Value Alignment.

---

## Overview
> Source: `gemini-prd.md` (lines 126–173, 284–290)

These are future-proofing "add-ons" that enhance the core system. They are not required for MVP but represent the cutting-edge capabilities.

---

## Integration 1: Hypergraph Knowledge Representation
> Source: `gemini-prd.md` (lines 130–136, 288)

- **What:** Standard graphs use triplets (A → B). Hypergraphs allow a single edge to connect _multiple_ nodes (Code + PRD + Timestamp + Author).
- **Why:** Code is rarely binary. A function depends on a library, a requirement, and a specific node version simultaneously.
- **Integration:** Use Hypergraph RAG in the Graphiti layer to allow "n-ary" relationships, reducing the number of "hops" the agent needs.
- **Implementation:** Update Graphiti schema to support "Hyperedges" (Node-to-Edge connections).

## Integration 2: Active Inference Curiosity Module
> Source: `gemini-prd.md` (lines 139–145)

- **What:** Replaces the random "Madlib" generator in the Evolutionary Forge. The agent calculates "Free Energy" (uncertainty) across its knowledge base.
- **Why:** The agent should learn _what it realizes it doesn't know_. If it knows React but not Svelte, the Curiosity Module detects that gap and generates a targeted learning task.
- **Integration:** A "Curiosity Daemon" runs before Sleep Time, identifying sparse areas in the Graphiti vector space and generating targeted learning tasks.

## Integration 3: Formal Verification Gate (VeriGuard Protocol)
> Source: `gemini-prd.md` (lines 148–154, 286)

- **What:** Uses a mathematical prover (Coq, Lean, or lightweight Python-based CrossHair) to strictly verify code correctness.
- **Why:** "95% Confidence" is subjective. "Mathematically Proven" is absolute.
- **Integration:** The Tribunal gains a **Math-Persona**. It demands the agent write not just code, but _assertions_. If assertions fail formal verification, the artifact is rejected immediately.
- **Implementation:** Add a `verify.py` hook in the Tribunal loop.

## Integration 4: Automated Model Merging (The "Frankenstein" Strategy)
> Source: `gemini-prd.md` (lines 157–163, 290)

- **What:** Techniques like TIES-Merging or DARE allow merging weights of different fine-tuned models without retraining.
- **Why:** Instead of just refining prompts, the system can merge a "Security Expert" LoRA with a "Creative Writer" LoRA to create a custom daily driver.
- **Integration:** Monthly script that identifies top-performing specialized behaviors and merges them into a custom generic model for the "Creator" role.
- **Implementation:** A monthly script that checks HuggingFace for compatible LoRAs and merges them.

## Integration 5: Contrastive Value Alignment (Taste Oracle++)
> Source: `gemini-prd.md` (lines 166–172)

- **What:** Uses a learned Reward Model based on specific "Taste" vectors (Contrastive Learning).
- **Why:** Simple vector distance is a crude proxy for "Good." A trained Reward Model (small classifier) can learn the nuance of why you like "Brutalist" code but dislike "Spaghetti" code, even if they look vectorially similar.
- **Integration:** Train a small classifier (e.g., DeBERTa) on "Accepted" vs. "Rejected" tribunal outcomes to act as a highly accurate pre-filter for the Creation loop.

---

## Implementation Tasks

1. Create `src/integrations/hypergraph.py` — Hyperedge support for Graphiti
2. Create `src/integrations/curiosity_module.py` — Free Energy-based knowledge gap detection
3. Create `src/integrations/veriguard.py` — Formal verification hook for Tribunal
4. Create `src/integrations/model_merger.py` — Monthly LoRA merging script
5. Create `src/integrations/taste_classifier.py` — Contrastive value alignment classifier

---

## Conflicts & Ambiguities

1. **Priority:** These are explicitly described as future add-ons, not MVP requirements. The implementation order should be: Hypergraph and Curiosity Module first (they enhance existing components), then VeriGuard, then Model Merging and Value Alignment (most complex).

2. **FalkorDB hypergraph support:** FalkorDB is a property graph database. True hypergraph support may require a different backend or a workaround using intermediate "hyperedge" nodes.

3. **Formal verification scope:** CrossHair (Python) is mentioned as a lightweight option, but formal verification of arbitrary code is extremely challenging. The scope should be limited to assertion checking, not full program verification.

4. **Model merging feasibility:** TIES-Merging and DARE work on models with compatible architectures. This limits which LoRAs can be merged. The monthly script needs robust compatibility checking.
