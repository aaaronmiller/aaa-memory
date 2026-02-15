# Topic 01: Embedding Model Stack

## Summary
Configure and initialize the multimodal embedding models (Qwen3-VL-Embedding-8B primary, Qwen3-VL-Reranker-8B, boundary detection model, and text-only fallback) that power the entire RAG pipeline.

---

## Requirements

### Primary Model: Qwen3-VL-Embedding-8B
> Sources: `chatgpt5.2-prd.md` (lines 59–100), `opus-prd1-v3.md`, `opus-prd2-v3.md` (lines 14–71), `docs/UNIFIED_PRD.md` (lines 92–142)

- **Model ID:** `Qwen/Qwen3-VL-Embedding-8B`
- **Released:** January 7-8, 2026 (arXiv:2601.04720)
- **Parameters:** 8.14B, 36 layers
- **Architecture:** Dual-Tower (query and document encoded independently)
- **Context Length:** 32,768 tokens (default 8,192)
- **Native Embedding Dimensions:** 4096
- **MRL (Matryoshka Representation Learning):** Supported — options: [256, 512, 1024, 2048, 4096]
  - Storage dimension: 1024 (truncated for MemVid efficiency)
  - Retrieval dimension: 2048 (higher precision for queries)
- **Quantization:** bf16 (recommended), fp16, int8, int4
- **Instruction-Aware:** Yes (task-specific instructions prepended to input)
- **Inference Config:**
  - `torch_dtype`: bfloat16
  - `attn_implementation`: flash_attention_2
  - `device_map`: auto

**Supported Input Modalities:**
- Pure text
- Pure image
- Pure video
- Text + image (mixed)
- Text + video (mixed)
- Image + video (mixed)
- Text + image + video (mixed)
- Screenshots (treated as images with OCR awareness)

**Vision Config:**
- `min_pixels`: 4096
- `max_pixels`: 1,843,200 (1280×1440)
- `total_video_pixels`: 7,864,320
- `default_fps`: 1.0
- `default_frames`: 64
- `max_frames`: 64

**Benchmarks:**
- MMEB-V2: 77.8 (Rank #1)
- MMTEB: 67.88
- Image Retrieval: 80.0
- Video Retrieval: 67.1
- VisDoc Retrieval: 82.4

### Reranker: Qwen3-VL-Reranker-8B
> Sources: `chatgpt5.2-prd.md` (lines 107–123), `opus-prd2-v3.md` (lines 82–103)

- **Model ID:** `Qwen/Qwen3-VL-Reranker-8B`
- **Parameters:** 8.14B, 36 layers
- **Architecture:** Single-Tower (cross-attention)
- **Input:** (Query, Document) pairs — both can be mixed-modal
- **Output:** Relevance score (via yes/no token generation probability)
- **Supported Modalities:** text, image, video, mixed
- **Inference:** bfloat16, flash_attention_2

### Boundary Detection Model: Qwen3-Embedding-0.6B
> Sources: `opus-prd2-v3.md` (lines 73–80), `docs/UNIFIED_PRD.md` (line 148)

- **Model ID:** `Qwen/Qwen3-Embedding-0.6B`
- **Type:** Text-only
- **Parameters:** 595.8M
- **Native Dimensions:** 1024
- **Purpose:** Cheap, fast similarity detection for semantic chunking boundary detection

### Fallback: Qwen3-Embedding-8B (Text-Only)
> Sources: `opus-prd2-v3.md` (lines 104–113)

- **Model ID:** `Qwen/Qwen3-Embedding-8B`
- **Type:** Text-only
- **Parameters:** 7.57B
- **Native Dimensions:** 4096
- **MTEB Score:** 70.58 (Rank #1) — higher than VL model on pure text
- **Use Case:** Fallback when multimodal embedding fails

### Alternative Consideration: Gemini Text-Embedding-001
> Source: `chatgpt5.2-prd.md` (lines 3–5)

- Mentioned as upcoming replacement for text-embedding-004
- Expected release: January 16, 2026
- **Status:** Not selected as primary; consider as future alternative

---

## Model Initialization Code
> Source: `chatgpt5.2-prd.md` (lines 147–171)

```python
import torch
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
from src.models.qwen3_vl_reranker import Qwen3VLReranker

# Primary Embedding Model
embedder = Qwen3VLEmbedder(
    model_name_or_path="Qwen/Qwen3-VL-Embedding-8B",
    max_length=8192,
    min_pixels=4096,
    max_pixels=1843200,
    total_pixels=7864320,
    fps=1.0,
    num_frames=64,
    max_frames=64,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

# Precision Reranker
reranker = Qwen3VLReranker(
    model_name_or_path="Qwen/Qwen3-VL-Reranker-8B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
```

---

## Implementation Tasks

1. Create `src/models/qwen3_vl_embedding.py` — wrapper class for Qwen3-VL-Embedding-8B
2. Create `src/models/qwen3_vl_reranker.py` — wrapper class for Qwen3-VL-Reranker-8B
3. Create `src/models/boundary_detector.py` — wrapper for Qwen3-Embedding-0.6B
4. Create `src/models/text_fallback.py` — wrapper for Qwen3-Embedding-8B text-only fallback
5. Implement MRL dimension truncation logic (4096 → 1024 for storage, 4096 → 2048 for retrieval)
6. Implement instruction-aware embedding (prepend task instructions to input)
7. Add multimodal input preprocessing (image resizing, video frame sampling)

---

## Cost Analysis
> Source: `docs/UNIFIED_PRD.md` (lines 240–249), `chatgpt5.2-prd.md` (lines 10–15)

| Component | Model | Estimated Cost |
|-----------|-------|---------------|
| Embedding | Qwen3-VL-Embedding-8B | ~$0.03/1M tokens |
| Reranking | Qwen3-VL-Reranker-8B | ~$0.05/1M tokens |
| One-time ingestion (35MB) | — | ~$0.10 |
| Queries (10K/day, annual) | — | ~$5.00 |

*Note: Not yet on OpenRouter at time of writing; may require self-hosting.*

---

## Conflicts & Ambiguities

1. **VL vs Text-Only tradeoff:** The text-only Qwen3-Embedding-8B scores higher on MTEB (70.58 vs 67.88) for pure text. The VL model is chosen for multimodal capability, but pure-text corpora may benefit from the text-only model. Decision documented in `chatgpt5.2-prd.md` lines 124–142.
2. **Gemini alternative:** `chatgpt5.2-prd.md` mentions considering Gemini Text-Embedding-001 as an alternative. No other document references this. Status unclear.
3. **Embedding dimension for storage:** `opus-prd2-v3.md` specifies 1024 for storage and 2048 for retrieval. `chatgpt5.2-prd.md` discusses "somewhere between 1 and 3K" without being specific. The YAML config is authoritative.
