# Topic: Strategic Integrations (Advanced Features)

## Summary
Five advanced integrations to push the architecture from "Advanced" to "State-of-the-Art": Hypergraph Knowledge, Active Inference, Formal Verification, Model Merging, and Contrastive Value Alignment.

---

## Overview

> **Source:** [gemini-prd.md](../gemini-prd.md) (Section 2)

These are future-proofing add-ons, not core requirements. They represent the upgrade path from the base system.

---

## 1. Hypergraph Knowledge Representation

> **Source:** [gemini-prd.md](../gemini-prd.md)

- **What:** Standard graphs use triplets (A → B). Hypergraphs allow a single edge to connect _multiple_ nodes (Code + PRD + Timestamp + Author).
- **Why:** Code is rarely binary. A function depends on a library, a requirement, and a specific node version simultaneously.
- **Integration:** Use Hypergraph RAG in the Graphiti layer to allow "n-ary" relationships, reducing the number of "hops" needed to understand complex dependencies.
- **Implementation:** Update Graphiti schema to support "Hyperedges" (Node-to-Edge connections)

---

## 2. Active Inference Curiosity Module (Frisstonian AI)

> **Source:** [gemini-prd.md](../gemini-prd.md)

- **What:** Replaces the random "Madlib" generator in the Evolutionary Forge. The agent calculates "Free Energy" (uncertainty) across its knowledge base.
- **Why:** The agent should learn _what it realizes it doesn't know_. If it knows React but not Svelte, the Curiosity Module detects that gap and generates a targeted learning task.
- **Integration:** A "Curiosity Daemon" runs before Sleep Time, identifying sparse areas in the Graphiti vector space and generating targeted learning tasks.

---

## 3. Formal Verification Gate (VeriGuard Protocol)

> **Source:** [gemini-prd.md](../gemini-prd.md)

- **What:** Uses a mathematical prover (Coq, Lean, or lightweight Python-based CrossHair) to verify code correctness.
- **Why:** "95% Confidence" is subjective. "Mathematically Proven" is absolute.
- **Integration:** The Tribunal gains a **Math-Persona** that demands the agent write assertions. If assertions fail formal verification, the artifact is rejected immediately.
- **Implementation:** Add a `verify.py` hook in the Tribunal loop.

---

## 4. Automated Model Merging (The "Frankenstein" Strategy)

> **Source:** [gemini-prd.md](../gemini-prd.md)

- **What:** Techniques like TIES-Merging or DARE allow merging weights of different fine-tuned models without retraining.
- **Why:** Instead of just refining prompts, the system can merge a "Security Expert" LoRA with a "Creative Writer" LoRA to create a custom daily driver.
- **Integration:** Monthly script that checks HuggingFace for compatible LoRAs and merges them.
- **Frequency:** Once a month

---

## 5. Contrastive Value Alignment (Taste Oracle++)

> **Source:** [gemini-prd.md](../gemini-prd.md)

- **What:** Uses a learned Reward Model based on user-specific "Taste" vectors (Contrastive Learning).
- **Why:** Simple vector distance is a crude proxy for "Good." A trained Reward Model can learn the nuance of why you like "Brutalist" code but dislike "Spaghetti" code, even if they look vectorially similar.
- **Integration:** Train a small classifier (e.g., DeBERTa) on "Accepted" vs. "Rejected" tribunal outcomes to act as a highly accurate pre-filter for the Creation loop.

---

## Implementation Priority

These are listed in suggested implementation order (after core system is built):

1. **Hypergraph Knowledge** — Enhances existing Graphiti layer (medium complexity)
2. **Active Inference Curiosity** — Replaces Madlib generator (medium complexity)
3. **Contrastive Value Alignment** — Improves Taste Oracle (medium complexity)
4. **Formal Verification** — Adds verification hook (small complexity)
5. **Model Merging** — Monthly automation (large complexity, requires ML expertise)

---

## Implementation Requirements

1. Research and select hypergraph library compatible with FalkorDB
2. Implement Free Energy calculation for knowledge gap detection
3. Integrate CrossHair or similar lightweight formal verifier
4. Build LoRA merging pipeline with HuggingFace integration
5. Train DeBERTa classifier on accepted/rejected outcomes
6. Create monthly automation for model merging

---

## Conflicts / Ambiguities

- **⚠️ These are aspirational:** Only gemini-prd.md describes these integrations. No other document references them. They should be treated as Phase 2+ features, not core requirements.
- **⚠️ Model merging feasibility:** Merging LoRAs requires access to model weights and significant ML infrastructure. May not be practical for a local-first system on M3 Max.
- **⚠️ Formal verification scope:** CrossHair (Python) is limited compared to Coq/Lean. The scope of what can be formally verified needs to be realistic.
