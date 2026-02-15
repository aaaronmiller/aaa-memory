# Topic 06: Three-Tiered Memory Hierarchy

## Summary
Implement the Hot/Warm/Cold memory hierarchy with ByteRover (filesystem), Graphiti (knowledge graph), and MemVid (video archive), including the transition protocols between tiers.

---

## Overview
> Sources: `gemini-prd.md` (lines 24–73), `docs/UNIFIED_PRD.md` (lines 50–88)

| Tier | Component | Role | Retention | Storage | Optimized For |
|------|-----------|------|-----------|---------|---------------|
| Hot | ByteRover | Active context | 0-24h | Filesystem (JSONL) | Speed (grep/find) |
| Warm | Graphiti | Knowledge graph | 7-90d | FalkorDB (Bolt Protocol) | Structured relationships |
| Cold | MemVid | Deep archive | 90d+ | H.265 video (QR frames) | Compression & retrieval |

---

## Hot Memory: ByteRover
> Sources: `gemini-prd.md` (lines 29, 254), `docs/UNIFIED_PRD.md` (line 55)

- **Storage:** Filesystem-based, `~/.byterover/inbox/` directory
- **Format:** JSONL with strict Pydantic schema
- **Schema fields:** `type`, `summary`, `content`, `tags`, `timestamp`
- **Purpose:** Live "Working Memory," active Git branches, "in-flight" ideas
- **Optimization:** Speed via grep/find (no database overhead)
- **Lifecycle:** Purged nightly during "Digest" transition (unless related to active Git branch)

## Warm Memory: Graphiti
> Sources: `gemini-prd.md` (lines 30, 256), `docs/UNIFIED_PRD.md` (line 56)

- **Storage:** Temporal Knowledge Graph via FalkorDB (Bolt Protocol)
- **Node types:** `Concept`, `Pattern`, `Decision`, `DecisionNode`, `PatternNode`, `AuthPattern`, etc.
- **Edge types:** `IMPLEMENTS`, `DEPRECATES`, `DEPENDS_ON`, `MITIGATES`
- **Purpose:** Structured relationships, "Skill" storage, "Lineage" tracking
- **Retention:** 7-90 days
- **Lifecycle:** Stale nodes (>30 days inactive) archived to MemVid weekly

## Cold Memory: MemVid
> Sources: `gemini-prd.md` (lines 31, 258–264), `docs/UNIFIED_PRD.md` (line 57)

- **Storage:** H.265 compressed video with QR frames
- **Vector:** Quad-Index (4 separate FAISS indices per video)
- **Metadata:** Sidecar JSON linking Vector_ID → Video_Timestamp
- **Visual:** QR Code (Version 40, High Error Correction)
- **Purpose:** Deep archive, "Gold Standard" datasets
- **Retention:** Permanent (90d+)

---

## Transition Protocols

### Transition A: The "Digest" (Hot → Warm)
> Source: `gemini-prd.md` (lines 42–54)

- **Trigger:** Nightly "Sleep Cycle" Daemon (or system idle > 15 minutes)
- **Input:** Raw interaction logs from ByteRover (cleaned via Proxy)
- **Process (The Dreamer):**
  1. **Structuring:** Convert raw logs into strict Graphiti Nodes (e.g., `DecisionNode`, `PatternNode`)
  2. **Filtering:** Discard "chatter" (conversational noise). Keep only "Solved Problems" and "Architectural Decisions"
- **Output:** New Nodes added to Graphiti. Raw logs purged from ByteRover (unless related to active Git branch)

### Transition B: The "Freeze" (Warm → Cold)
> Source: `gemini-prd.md` (lines 57–73)

- **Trigger:** Weekly "Archivist" Job (Sunday)
- **Input:** Stale Graphiti nodes (>30 days inactive) + Curated "Gold Standard" datasets
- **Process (The Renderer):**
  1. **Deconstruction:** Serialize nodes into JSON
  2. **Rendering:** Generate QR Code images (PNGs) of the JSON data
  3. **Quad-Encoding:** Generate 4 vector layers (Token, Fact, Context, Boundary)
  4. **Stitching:** Compile images into an H.265 `.mp4` video file
- **Output:** Portable MemVid archive file. Stale nodes in Graphiti replaced with lightweight "Tombstone Pointers" (e.g., `See Archive W42`)

---

## Implementation Tasks

1. Create `src/memory/byterover.py` — Hot memory filesystem manager
   - JSONL read/write with Pydantic validation
   - Inbox directory management
   - Active Git branch awareness
2. Create `src/memory/graphiti.py` — Warm memory knowledge graph interface
   - FalkorDB connection via Bolt Protocol
   - Node/Edge CRUD operations
   - Staleness detection (>30 days inactive)
   - Tombstone pointer creation
3. Create `src/memory/transitions.py` — Tier transition logic
   - Digest (Hot → Warm): Log structuring, filtering, node creation
   - Freeze (Warm → Cold): Serialization, QR rendering, quad-encoding, stitching
4. Create `src/memory/query_router.py` — Route queries to appropriate memory tier(s)

---

## Conflicts & Ambiguities

1. **Graphiti retention period:** `gemini-prd.md` says "7-90h" (hours) in the component table but "30 days" for staleness threshold. The UNIFIED_PRD says "7-90d" (days). The 30-day staleness threshold is the actionable number; the "7-90" range likely refers to the typical active window.

2. **ByteRover vs direct SQLite:** The RAG pipeline (Topics 01-04) uses SQLite for chunk metadata. ByteRover uses filesystem JSONL. These are separate systems — ByteRover is for live interaction logs, SQLite is for the processed RAG corpus. They coexist.

3. **FalkorDB containerization:** `gemini-prd.md` (line 269) specifies `docker-compose.yml` for FalkorDB. This is an infrastructure dependency that needs to be documented in setup instructions.

4. **Tombstone pointers:** When nodes move from Warm to Cold, they're replaced with pointers. The query router needs to follow these pointers transparently when a warm-memory query hits a tombstone.
