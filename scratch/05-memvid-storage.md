# Topic: MemVid Storage (Video-Encoded Vector Storage)

## Summary
MemVid is the Cold Memory storage layer that encodes chunks and embeddings into H.265 compressed video files (QR frames) for massive compression. Includes encoder configuration, quad-encoding strategy, and file organization.

---

## What is MemVid?

> **Source:** [gemini-prd.md](../gemini-prd.md), [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md)

MemVid is a deep archive (90d+) storage system that uses H.265 compressed video (QR frames) to store massive datasets. It provides:
- 50-100x compression over raw storage
- Quad-encoded vectors for high-fidelity retrieval
- Portable archive files (.mp4)

---

## Encoder Configuration

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md)

```yaml
memvid:
  encoder:
    codec: "hevc"          # H.265
    crf: 18                # Constant Rate Factor (quality)
    gop: 30                # Group of Pictures
    preset: "medium"       # Encoding speed/quality tradeoff
  
  vector_config:
    input_dimensions: 4096   # Native Qwen3-VL output
    storage_dimensions: 1024 # MRL-truncated for efficiency
    similarity_sort: true    # Sort vectors for better compression
  
  features:
    parallel_segments: true
    smart_recall: true
    text_search: true
    hnsw_index: true
```

---

## File Organization

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md)

Three separate MemVid files by domain:

| File | Domain | Content |
|------|--------|---------|
| `codebase.mp4` | Codebase | Multi-repository source code and configs |
| `research.mp4` | Research | Research papers, documentation, diagrams |
| `prompts.mp4` | Prompts | User inputs and prompts to LLMs |

---

## Encoding Process (The "Freeze" Transition)

> **Source:** [gemini-prd.md](../gemini-prd.md)

The Warm → Cold transition ("Freeze") follows this process:

1. **Deconstruction:** Serialize Graphiti nodes into JSON
2. **Rendering:** Generate QR Code images (PNGs) of the JSON data
   - QR Code Version 40, High Error Correction
3. **Quad-Encoding:** Generate 4 vector layers per content block
4. **Stitching:** Compile images into an H.265 `.mp4` video file

### QR Code Specifications
> **Source:** [gemini-prd.md](../gemini-prd.md)

- **Version:** 40 (maximum capacity)
- **Error Correction:** High
- **Format:** PNG images compiled into video frames

---

## Quad-Encoding Strategy

> **Source:** [gemini-prd.md](../gemini-prd.md) (Appendix I)

Each content block is encoded at four resolutions with separate FAISS indices:

| Layer | Resolution | What it Encodes | Use Case |
|-------|-----------|----------------|----------|
| 1 | Word/Token | Keywords & Entities | Exact definitions, variable names |
| 2 | Sentence/Fact | Discrete Facts | Return types, specific values |
| 3 | Paragraph/Context | Local Context | How flows handle edge cases |
| 4 | Boundary | Relationships & Flow | Cross-section connections |

### Why Quad-Encoding?
- **Needle in a Haystack Fix:** Word/Sentence vectors allow precise fact retrieval without paragraph dilution
- **Context Drift Fix:** Boundary vectors encode concept edges, preventing information loss at chunk boundaries

### Storage Architecture: "Heavy Index, Light Payload"
- **Index (4x larger):** 4 FAISS indices per video file
- **Payload (100x smaller):** H.265 compressed video stores actual content
- **Result:** Trading cheap disk space for high intelligence density

---

## MP4 RAG Encoder Implementation

> **Source:** [gemini-prd.md](../gemini-prd.md) (Appendix II)

```python
class MP4RAGEncoder:
    def __init__(self, frame_width=1920, frame_height=1080):
        self.frame_width = frame_width
        self.frame_height = frame_height
    
    def text_to_frame(self, text, chunk_id):
        """Convert text chunk to image frame"""
        # Render text with word wrap onto image
        # Add chunk_id as QR code or metadata overlay
    
    def encode_chunks_to_mp4(self, chunks, embeddings, metadata, output_path):
        """Encode all chunks into MP4 with H.265 compression"""
        # Write video with H.265 (HEVC) codec
        # Store chunk index and embeddings as sidecar files
    
    def decode_frame(self, mp4_path, frame_number):
        """Quickly seek to specific frame and extract text"""
        # OCR the frame to get text back
```

### Sidecar Files
Each `.mp4` file has companion files:
- `*_index.json` — Maps frame numbers to chunk IDs and metadata
- `*_embeddings.npy` — NumPy array of embedding vectors

---

## MemVid Index Table (SQLite)

> **Source:** [docs/SCHEMA_REFERENCE.md](../docs/SCHEMA_REFERENCE.md)

```sql
CREATE TABLE memvid_indices (
    memvid_file TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    chunk_id TEXT NOT NULL,
    embedding_id TEXT NOT NULL,
    corpus_id TEXT NOT NULL,
    PRIMARY KEY (memvid_file, frame_index),
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id),
    FOREIGN KEY (embedding_id) REFERENCES embeddings(embedding_id)
);
```

---

## Integration with Sleep-Time Compute

> **Source:** [gemini-prd.md](../gemini-prd.md)

Quad-encoding is computationally expensive (4x embedding time) and cannot be done in real-time. The workflow:

1. **Live (ByteRover):** Simple paragraph chunks (fast, good enough)
2. **Sleep Time (Daemon):** Explodes content into 4 layers, embeds all, encodes to MemVid
3. **Next Day:** Agent has "super-resolution" access to yesterday's work

---

## Retrieval from MemVid: The "Zoom" Pattern

> **Source:** [gemini-prd.md](../gemini-prd.md)

Cascading lookup strategy (not all 4 layers at once):

1. **Scout (Paragraph Layer):** Find general concepts — broad context
2. **Snipe (Sentence Layer):** Check specific facts in the region — precise lines
3. **Stitch (Boundary Layer):** Retrieve boundary vectors to see what connects next

---

## Implementation Requirements

1. Implement MP4RAGEncoder class with H.265 encoding
2. Implement QR code generation for JSON serialization
3. Build quad-encoding pipeline (4 FAISS indices per video)
4. Create sidecar file management (index.json, embeddings.npy)
5. Implement frame seeking and OCR-based decoding
6. Build the "Zoom" pattern retrieval logic
7. Integrate with sleep-time daemon for batch encoding

---

## Dependencies

- FFmpeg (H.265/HEVC encoding)
- OpenCV (cv2) for video I/O
- FAISS for vector indexing
- Pillow for image generation
- QR code library (qrcode or similar)
- Ghostscript (for QR rendering)

---

## Conflicts / Ambiguities

- **⚠️ QR vs text rendering:** gemini-prd.md describes QR code frames, but the MP4RAGEncoder code in Appendix II renders text directly onto frames. These are two different approaches — QR is more robust for data integrity, text rendering is simpler. The QR approach (from Section 3.4) appears to be the intended production approach.
- **⚠️ Vector storage location:** opus-prd2-v3.md mentions `hnsw_index: true` as a MemVid feature, but gemini-prd.md describes separate FAISS indices. These may be complementary (HNSW within FAISS).
- **⚠️ Sidecar vs embedded:** The code example uses sidecar files for index/embeddings, but the SQLite memvid_indices table provides a database-backed alternative. Both approaches may coexist.
