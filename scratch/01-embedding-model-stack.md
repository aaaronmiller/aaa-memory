# Topic: Embedding Model Stack

## Summary
Configuration, initialization, and integration of the Qwen3-VL multimodal embedding models and reranker for the RAG v3.0 system.

---

## Primary Model: Qwen3-VL-Embedding-8B

> **Source:** [opus-prd1-v3.md](../opus-prd1-v3.md), [opus-prd2-v3.md](../opus-prd2-v3.md), [opus-prd3-v3.md](../opus-prd3-v3.md), [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md)

- **Model ID:** `Qwen/Qwen3-VL-Embedding-8B`
- **Released:** January 7-8, 2026 (arXiv:2601.04720)
- **Parameters:** 8.14B
- **Layers:** 36
- **Architecture:** Dual-Tower (qwen3_vl)
- **Context Length:** 32,768 tokens (default 8,192)
- **Native Embedding Dimensions:** 4096
- **MRL Support:** Yes — options: [256, 512, 1024, 2048, 4096]
  - Storage dimension: 1024 (truncated for MemVid efficiency)
  - Retrieval dimension: 2048 (higher precision for queries)
- **Quantization:** bf16 (recommended), fp16, int8, int4
- **Instruction-Aware:** Yes

### Benchmarks
| Benchmark | Score |
|-----------|-------|
| MMEB-V2 | 77.8 (Rank #1) |
| MMTEB | 67.88 |
| Image Retrieval | 80.0 |
| Video Retrieval | 67.1 |
| VisDoc Retrieval | 82.4 |

### Supported Input Modalities
- Pure text
- Pure image
- Pure video
- Text + image (mixed)
- Text + video (mixed)
- Image + video (mixed)
- Text + image + video (mixed)
- Screenshots (treated as images with OCR awareness)

### Vision Configuration
> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md)

- `min_pixels`: 4096
- `max_pixels`: 1,843,200 (1280×1440)
- `total_video_pixels`: 7,864,320
- `default_fps`: 1.0
- `default_frames`: 64
- `max_frames`: 64

### Inference Configuration
> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md)

- `torch_dtype`: bfloat16
- `attn_implementation`: flash_attention_2
- `device_map`: auto

### Architecture Details
> **Source:** [opus-prd3-v3.md](../opus-prd3-v3.md)

- Extracts `[EOS]` token hidden state from last layer as final representation
- Cross-modal pretraining with unified modality projection
- Integrates supervised tasks, masked modeling, and multimodal alignment objectives
- Enables efficient independent encoding for large-scale retrieval

---

## Boundary Detection Model: Qwen3-Embedding-0.6B

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md), [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md)

- **Model ID:** `Qwen/Qwen3-Embedding-0.6B`
- **Type:** Text-only
- **Parameters:** 595.8M
- **Native Dimensions:** 1024
- **Purpose:** Cheap/fast similarity detection for semantic chunking boundary detection

---

## Reranker: Qwen3-VL-Reranker-8B

> **Source:** [opus-prd1-v3.md](../opus-prd1-v3.md), [opus-prd2-v3.md](../opus-prd2-v3.md)

- **Model ID:** `Qwen/Qwen3-VL-Reranker-8B`
- **Parameters:** 8.14B
- **Layers:** 36
- **Architecture:** Single-Tower with Cross-Attention
- **Input:** (Query, Document) pairs — both can be mixed-modal
- **Output:** Relevance score (via yes/no token generation probability)
- **Supported Modalities:** text, image, video, mixed
- **Inference:** bfloat16, flash_attention_2

### Smaller Variant: Qwen3-VL-Reranker-2B
- **Parameters:** 2.13B
- Same architecture (Single-Tower)

---

## Fallback Model: Qwen3-Embedding-8B (Text-Only)

> **Source:** [opus-prd2-v3.md](../opus-prd2-v3.md)

- **Model ID:** `Qwen/Qwen3-Embedding-8B`
- **Type:** Text-only
- **Parameters:** 7.57B
- **Native Dimensions:** 4096
- **MTEB Score:** 70.58 (Rank #1)
- **Note:** Higher MTEB score than VL model (70.58 vs 67.88) but lacks multimodal capabilities

---

## Model Initialization Code

> **Source:** [chatgpt5.2-prd.md](../chatgpt5.2-prd.md) (Phase 1), [opus-prd1-v3.md](../opus-prd1-v3.md)

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

## Alternative Models Considered

> **Source:** [chatgpt5.2-prd.md](../chatgpt5.2-prd.md)

- **Gemini Text-Embedding-001:** Upcoming model (replacing text-embedding-004), expected January 16, 2026. Considered as alternative/complement.
- **Qwen3-VL-Embedding-2B:** Lightweight variant (2.13B params, 2048 dims, MMEB-V2: 73.2)

---

## Cost Analysis

> **Source:** [docs/UNIFIED_PRD.md](../docs/UNIFIED_PRD.md), [chatgpt5.2-prd.md](../chatgpt5.2-prd.md)

| Component | Model | Cost |
|-----------|-------|------|
| Embedding | Qwen3-VL-Embedding-8B | ~$0.03/1M tokens* |
| Reranking | Qwen3-VL-Reranker-8B | ~$0.05/1M tokens* |
| Ingestion (35MB) | One-time | ~$0.10 |
| Queries (10K/day, annual) | - | ~$5.00 |

*Estimated — not yet on OpenRouter, requires self-hosting or wait for API availability.

---

## Implementation Requirements

1. Set up Qwen3-VL-Embedding-8B environment with flash_attention_2
2. Implement model wrapper classes (`Qwen3VLEmbedder`, `Qwen3VLReranker`)
3. Support MRL dimension truncation for storage vs retrieval
4. Implement multimodal input preprocessing (text, image, video, mixed)
5. Add fallback to text-only model on multimodal failure
6. Integrate with OpenRouter for remote inference

---

## Conflicts / Ambiguities

- **⚠️ Dimension mismatch:** chatgpt5.2-prd.md mentions "1526 or 3746 or 3182" as possible embedding sizes — these don't match the actual Qwen3-VL dimensions (4096 native, MRL options: 256/512/1024/2048/4096). The opus PRDs provide the correct values.
- **⚠️ Gemini alternative:** chatgpt5.2-prd.md suggests potentially using both Qwen and Gemini embeddings. No other document addresses dual-embedding strategy.
- **⚠️ Hosting:** chatgpt5.2-prd.md assumes OpenRouter availability; cost estimates are speculative since the model may require self-hosting.
