# Topic 05: MemVid Storage (Video-Encoded Vectors)

## Summary
Implement the MemVid cold storage system that encodes chunks and their embeddings into H.265 compressed video files (MP4) with QR-code frames, providing massive compression for long-term archival with efficient retrieval via frame-seeking.

---

## Overview
> Sources: `gemini-prd.md` (lines 31, 57–73, 295–500+), `opus-prd2-v3.md` (lines 335–359), `docs/UNIFIED_PRD.md` (lines 57, 83–88), `chatgpt5.2-prd.md` (lines 25, 42)

MemVid is the **Cold Memory** tier (90d+ retention). It uses H.265 compressed video to store massive datasets as QR-code frames, with sidecar vector indices for retrieval.

---

## Encoder Configuration
> Source: `opus-prd2-v3.md` (lines 338–348)

```yaml
memvid:
  encoder:
    codec: "hevc"          # H.265
    crf: 18                # Constant Rate Factor (quality)
    gop: 30                # Group of Pictures
    preset: "medium"       # Encoding speed/quality tradeoff
  
  vector_config:
    input_dimensions: 4096   # From Qwen3-VL-Embedding-8B
    storage_dimensions: 1024 # MRL-truncated for efficiency
    similarity_sort: true    # Sort frames by similarity for better compression
  
  features:
    parallel_segments: true
    smart_recall: true
    text_search: true
    hnsw_index: true
```

## Domain-Specific Files
> Source: `opus-prd2-v3.md` (lines 356–359)

| File | Domain | Description |
|------|--------|-------------|
| `codebase.mp4` | Code | Multi-repository source code and configs |
| `research.mp4` | Research | Research papers, documentation, diagrams |
| `prompts.mp4` | Prompts | User inputs and prompts to LLMs |

---

## Encoding Process (The "Freeze" — Warm → Cold)
> Source: `gemini-prd.md` (lines 57–73)

**Trigger:** Weekly "Archivist" Job (Sunday)

**Input:** Stale Graphiti nodes (>30 days inactive) + Curated "Gold Standard" datasets

**Process (The Renderer):**
1. **Deconstruction:** Serialize nodes into JSON
2. **Rendering:** Generate QR Code images (PNGs) of the JSON data
   - QR Version 40, High Error Correction
3. **Quad-Encoding:** Generate 4 vector layers per content block:
   - Token (Word) — keywords & entities
   - Fact (Sentence) — discrete facts
   - Context (Paragraph) — local context
   - Boundary — relationships & flow between sections
4. **Stitching:** Compile images into an H.265 `.mp4` video file

**Output:** A portable MemVid archive file. Stale nodes in Graphiti replaced with lightweight "Tombstone Pointers" (e.g., `See Archive W42`).

---

## Quad Encoding Details
> Source: `gemini-prd.md` (Appendix I, lines 295–368)

| Resolution | What it Encodes | Agent Query Type |
|-----------|----------------|-----------------|
| Word (Token) | Keywords & Entities | "Find the exact definition of variable `MAX_RETRIES`." |
| Sentence | Discrete Facts | "What is the return type of `auth.login()`?" |
| Paragraph | Local Context | "How does the login flow handle 2FA failures?" |
| Boundary | Relationships & Flow | "What happens _after_ the login module finishes?" |

**Architecture:** "Heavy Index, Light Payload"
- The vector index (FAISS) grows 4x due to quad encoding
- The payload (H.265 compressed video) shrinks ~100x
- Optimal for local AI: trading disk space (cheap) for intelligence density (high)

**Integration with Sleep-Time Compute:**
- Live (ByteRover): Uses simple paragraph chunks (fast, good enough for active work)
- Sleep Time (Daemon): Explodes content into 4 layers, embeds all, encodes to MemVid
- Result: Next day, agent has "Super-Resolution" access to yesterday's work

---

## MP4 RAG Encoder Implementation
> Source: `gemini-prd.md` (Appendix II, lines 396–500)

Key classes:
- `MP4RAGEncoder` — Converts text chunks to video frames, encodes with H.265
  - `text_to_frame(text, chunk_id)` — Renders text as image frame with chunk ID overlay
  - `encode_chunks_to_mp4(chunks, embeddings, metadata, output_path)` — Full encoding pipeline
  - `decode_frame(mp4_path, frame_number)` — Seek to specific frame and extract text

Storage structure per MP4:
- `{name}.mp4` — The video file with QR/text frames
- `{name}_index.json` — Chunk index mapping frame numbers to chunk IDs
- `{name}_embeddings.npy` — NumPy array of embedding vectors

---

## Agent Retrieval Logic (The "Zoom" Pattern)
> Source: `gemini-prd.md` (lines 353–364)

Cascading lookup across quad-encoded layers:
1. **Scout (Paragraph Layer):** Find general concepts related to query (broad context)
2. **Snipe (Sentence Layer):** Check specific facts in the region (precise lines)
3. **Stitch (Boundary Layer):** If chunk ends abruptly, retrieve boundary vector to see what connects next

---

## Implementation Tasks

1. Create `src/memvid/encoder.py` — MP4RAGEncoder class (text-to-frame, H.265 encoding)
2. Create `src/memvid/decoder.py` — Frame seeking and text extraction
3. Create `src/memvid/quad_encoder.py` — Quad-encoding logic (4 vector layers per content)
4. Create `src/memvid/index_manager.py` — Sidecar JSON index management
5. Create `src/memvid/archiver.py` — Weekly archival job (Warm → Cold transition)
6. Create `src/memvid/retriever.py` — Cascading "Zoom" pattern retrieval
7. Integrate with FAISS for HNSW index support
8. Implement similarity-sorted frame ordering for better H.265 compression

---

## Conflicts & Ambiguities

1. **QR codes vs text rendering:** `gemini-prd.md` describes QR code frames, while the Appendix II code example renders text directly onto frames. The QR approach is more robust for data integrity; the text rendering is simpler. Need to decide which approach to use (or both — QR for data, text for debugging).

2. **4 FAISS indices vs 1:** Quad encoding implies 4 separate FAISS indices per video file (`gemini-prd.md` line 262: "Quad-Index (4 separate FAISS indices per video)"). This is a significant architectural decision affecting storage and query routing.

3. **MemVid frame rate:** `opus-prd2-v3.md` encoder config uses `gop: 30` suggesting 30fps, but the Appendix II code uses `30.0` fps. The embedding model's `default_fps: 1.0` is for video input processing, not MemVid encoding. These are separate concerns.

4. **ChromaDB vs FAISS:** `docs/UNIFIED_PRD.md` (line 270) lists "ChromaDB/FAISS" as vector store options. `gemini-prd.md` specifically uses FAISS. Need to pick one or support both.
