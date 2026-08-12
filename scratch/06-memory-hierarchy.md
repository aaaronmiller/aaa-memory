# Topic: Three-Tiered Memory Hierarchy

## Summary
The Hot/Warm/Cold memory architecture using ByteRover, Graphiti, and MemVid, including data lifecycle transitions and graduation protocols.

---

## Architecture Overview

> **Source:** [gemini-prd.md](../gemini-prd.md), [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md)

| Tier | Component | Role | Retention | Storage Format | Optimized For |
|------|-----------|------|-----------|---------------|---------------|
| Hot | ByteRover | Active Context | 0-24h | JSONL (filesystem) | Speed (grep/find) |
| Warm | Graphiti | Knowledge Graph | 7-90d | Property Graph (FalkorDB) | Relationships |
| Cold | MemVid | Deep Archive | 90d+ | H.265 video (QR frames) | Compression/Density |

---

## Hot Memory: ByteRover

> **Source:** [gemini-prd.md](../gemini-prd.md)

- **Type:** Filesystem-based active context
- **Location:** `~/.byterover/inbox/`
- **Format:** JSONL with strict Pydantic schema
- **Schema fields:** type, summary, content, tags, timestamp
- **Purpose:** Stores live "Working Memory," active Git branches, and "in-flight" ideas
- **Optimization:** Speed via grep/find (no database overhead)
- **Retention:** Purged nightly unless related to active Git branch

### Live Usage
During active work, ByteRover uses simple paragraph chunks — fast and good enough for real-time context injection.

---

## Warm Memory: Graphiti

> **Source:** [gemini-prd.md](../gemini-prd.md)

- **Type:** Temporal Knowledge Graph
- **Backend:** FalkorDB (via Bolt Protocol)
- **Node Types:** Concept, Pattern, Decision, DecisionNode, PatternNode
- **Edge Types:** IMPLEMENTS, DEPRECATES, DEPENDS_ON, MITIGATES
- **Purpose:** Stores structured relationships, "Skill" storage, and "Lineage"
- **Retention:** 7-90 days active; nodes >30 days inactive become candidates for archival

### Tombstone Pointers
When nodes are archived to MemVid, they are replaced with lightweight "Tombstone Pointers" (e.g., `See Archive W42`) to maintain graph connectivity.

---

## Cold Memory: MemVid

> **Source:** [gemini-prd.md](../gemini-prd.md), [opus-prd2-v3.md](../opus-prd2-v3.md)

See [scratch/05-memvid-storage.md](./05-memvid-storage.md) for full details.

- **Type:** Deep archive with video-encoded storage
- **Format:** H.265 compressed video with QR frames
- **Vector Index:** Quad-encoded (4 FAISS indices per video)
- **Metadata:** Sidecar JSON + SQLite memvid_indices table
- **Retention:** Permanent

---

## Transition A: The "Digest" (Hot → Warm)

> **Source:** [gemini-prd.md](../gemini-prd.md), [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md)

- **Trigger:** Nightly "Sleep Cycle" Daemon (or system idle > 15 minutes)
- **Input:** Raw interaction logs from ByteRover (cleaned via Proxy)
- **Process (The Dreamer):**
  1. **Structuring:** Convert raw logs into strict Graphiti Nodes (DecisionNode, PatternNode)
  2. **Filtering:** Discard "chatter" (conversational noise). Keep only "Solved Problems" and "Architectural Decisions"
- **Output:** New Nodes added to Graphiti. Raw logs purged from ByteRover (unless related to active Git branch)

---

## Transition B: The "Freeze" (Warm → Cold)

> **Source:** [gemini-prd.md](../gemini-prd.md), [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md)

- **Trigger:** Weekly "Archivist" Job (Sunday)
- **Input:** Stale Graphiti nodes (>30 days inactive) + Curated "Gold Standard" datasets
- **Process (The Renderer):**
  1. **Deconstruction:** Serialize nodes into JSON
  2. **Rendering:** Generate QR Code images (PNGs) of the JSON data
  3. **Quad-Encoding:** Generate 4 vector layers (Token, Fact, Context, Boundary)
  4. **Stitching:** Compile images into an H.265 `.mp4` video file
- **Output:** Portable MemVid archive file. Stale nodes replaced with Tombstone Pointers in Graphiti.

---

## Data Flow Diagram

```
User Input → Proxy → ByteRover (Hot, 0-24h)
                         ↓ [Nightly Digest]
                     Graphiti (Warm, 7-90d)
                         ↓ [Weekly Freeze]
                     MemVid (Cold, 90d+)
```

---

## Implementation Requirements

1. Implement ByteRover filesystem layer with JSONL read/write and Pydantic validation
2. Set up FalkorDB container for Graphiti (docker-compose)
3. Define Graphiti node and edge schemas
4. Implement the "Digest" transition daemon (Hot → Warm)
5. Implement the "Freeze" transition daemon (Warm → Cold)
6. Build Tombstone Pointer system for archived nodes
7. Implement idle detection trigger (system idle > 15 minutes)

---

## Conflicts / Ambiguities

- **⚠️ Retention periods:** gemini-prd.md says Hot is 0-24h and Warm is 7-90h (hours), but UNIFIED_PRD.md says Warm is 7-90d (days) and the Freeze trigger is >30 days inactive. The "h" in gemini-prd.md appears to be a typo — days is the intended unit based on context.
- **⚠️ Graph database:** gemini-prd.md specifies FalkorDB via Bolt Protocol. No other document confirms this choice. The containerization section mentions docker-compose for FalkorDB.
- **⚠️ ByteRover location:** gemini-prd.md uses `~/.byterover/inbox/` but this is macOS-specific. Should be configurable.
