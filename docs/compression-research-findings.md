# Compression Research Findings — H.265, MemVid, 4D/5D, Gaussian Splatting
**Date**: 2026-04-08  
**Session**: Memory Architecture — Compression Deep Dive

---

## 1. MemVid V2 (Keep as Cold Tier)

**What it is**: Rust rewrite of Python v1. Stores AI memory as "Smart Frames" in single `.mv2` file with WAL mode.

**Mechanism**:
- Each frame packages: original content + vector embedding + timestamp + relationship metadata
- V1 used QR codes → H.264/H.265 video encoding → MP4 file
- V2 uses unified `.mv2` format (not literal video encoding, but same compression principle)

**Metrics**:
- 90% size reduction — ~50,000 documents in ~200MB
- Sub-5ms to sub-17ms retrieval for 50k documents
- >60% higher accuracy than traditional RAG pipelines
- Hybrid search: semantic + keywords + tags + time/context

**Keep it.** No reason not to use it.

---

## 2. LLM.265 Paper — "Video Codecs are Secretly Tensor Codecs"

**Source**: HKUST, ACM Multimedia 2025  
**URL**: https://dl.acm.org/doi/10.1145/3725843.3756078

**The insight**: Video codecs (H.264/H.265/VP9) are NOT video-specific. They are general-purpose tensor compressors that exploit spatio-temporal redundancy.

**How it works**:
1. **I-frame**: Full reference tensor (save once)
2. **P-frames**: Only store the delta (change) from previous frame
3. **Motion vectors**: Predict where tensor values "moved" between frames
4. **DCT + quantization**: Compress the residual

**For our memory system**:
- Session embeddings are highly temporally redundant
- Same project across 10 sessions = vectors change incrementally
- H.265 treating them as "video frames" = 10-50x compression vs raw FP32

**This is our novel contribution**: Nobody has applied temporal video compression to semantic embedding sequences for memory systems.

---

## 3. N4MC — Neural 4D Mesh Compression

**Source**: arXiv Feb 2026 (2602.20312)  
**URL**: https://arxiv.org/html/2602.20312v1

**What it is**: First neural framework for compressing time-varying 4D mesh sequences.

**Compression ratios**:
- 89.56x for static meshes
- 4-6x better than existing methods for 4D sequences
- ~2.0-4.5 Mbps for high-fidelity 4D

**Architecture**:
- Volumetric TSDF-Def tensors (4D: 3D space + 1D time)
- 3D ConvNeXt autoencoder → latent compression
- Motion-guided latent codes via as-rigid-as-possible tracking
- 3D interpolation transformer for temporal super-resolution

**NOT applicable to text embeddings**:
- Intrinsically 3D-geometric
- Uses 3D convolutions, 3D PixelShuffle, Marching Cubes
- Volume tracking and geometry-specific metrics
- Would require replacing entire architecture for non-3D data

---

## 4. Gaussian Splatting Compression (3DGS)

**What it is**: Real-time 3D scene rendering using Gaussian primitives.

**Relevant compression techniques**:
- **PCGS** (Progressive Compression of 3DGS, NeurIPS 2025): Progressive quantization → applicable to embedding precision
- **3DGS.zip** (Eurographics 2026 survey): Vector quantization with codebooks → applicable to embedding storage
- **Distribution Regularization + Probabilistic Pruning** (AAAI 2026): Remove low-contribution elements → applicable to pruning stale embeddings

**NOT directly applicable**:
- 3DGS works because scenes have geometric locality (nearby Gaussians → nearby pixels)
- Text embeddings have no spatial structure — they're points in abstract semantic space
- The gs-embedding repo (ICLR 2026) converts 3DGS INTO embeddings, not the other way around

**What we steal**: Quantization techniques and codebook approaches. Not the splatting itself.

---

## 5. Google TurboQuant

**Source**: Google Research, ICLR 2026 (March 2026)

**What it is**: KV cache compression for LLM inference (6x reduction via PolarQuant + QJL)

**NOT applicable**: Strictly for inference-time working memory. Does not address training memory, vector embeddings, or persistent storage.

---

## 6. The 4D/5D/6D Question

Standard video codecs (H.264/H.265) handle **3D data** natively: 2D spatial (width × height) + 1D temporal (frames).

For **4D+** (e.g., embedding dimension as a 4th axis), two options:

### Option A: Slice the tensor (what MemVid does)
- Reshape 1024-dim embeddings as 32×32 "images"
- Treat sessions as "frames"
- H.265 sees 32×32×N as a video
- **This works** — it's what MemVid does with QR codes, but LLM.265 shows you can skip QR step and pack floats directly

### Option B: Neural compression (N4MC approach)
- Learn a latent space via autoencoder
- Entropy-code the latents
- Works for any dimensionality but requires training a model per data domain
- **Too complex** for our use case

**The practical answer**: Option A. Reshape embeddings as 2D arrays, use H.265. It's what MemVid already does. LLM.265 validates that video codecs are state-of-the-art tensor compressors.

---

## 7. Our Frozen Tier (Tier 4) Design

```
Monthly daemon (during sleep-time compute):
1. Scan Graphiti for nodes with zero query hits in 180+ days
2. Extract embeddings from those nodes
3. Group by project + temporal proximity
4. Reshape each embedding: 1024-dim → 32×32 float array
5. Quantize to uint8 (minimal precision loss, verify via recall tests)
6. Stack temporally: sessions on same project become "frames"
7. Encode with H.265 (FFmpeg, libx265, CRF 28)
8. Store: archive.mp4 + index file (frame→turn_id mapping, JSON)
9. Delete raw embeddings from SQLite (reclaim space)

On-demand retrieval:
1. Read index file → find relevant frames
2. ffmpeg -i archive.mp4 -f rawvideo -pix_fmt grayf32le -
3. Reshape back to 1024-dim vectors
4. Use for retrieval
```

**Compression estimate**:
- Raw FP32 embeddings: 1024 × 4 bytes = 4KB per embedding
- 100 sessions × 4KB = 400KB raw
- H.265 with temporal deltas: ~40-80KB (10-50x depending on similarity)
- Savings compound over time — more sessions on same project = more redundancy = better compression

---

## 8. Key Papers/Projects Referenced

| Source | What We Use | Status |
|--------|------------|--------|
| **MemVid V2** (olow304/memvid) | Cold tier storage format | ✅ Keep |
| **LLM.265** (HKUST, ACM MM 2025) | Validates video codecs as tensor compressors | ✅ Apply to embedding archive |
| **PCGS** (NeurIPS 2025) | Progressive quantization for embedding precision | ✅ Steal technique |
| **3DGS.zip** (Eurographics 2026) | Vector quantization codebooks for embedding compression | ✅ Steal technique |
| **N4MC** (arXiv Feb 2026) | Inspiration for neural latent compression | ❌ Not applicable (3D-geometric) |
| **gs-embedding** (ICLR 2026) | Gaussian splat → embedding pipeline | ❌ Wrong direction |
| **Google TurboQuant** (ICLR 2026) | KV cache compression | ❌ Irrelevant (inference-only) |

---

## 9. Research Gap (Our Novel Contribution)

**Applying temporal video compression to semantic embedding sequences.**

MemVid compresses QR-encoded documents. LLM.265 compresses ML tensors. But nobody has built a system that:
1. Embeds text into vectors
2. Orders them temporally by project/topic
3. Reshapes as 2D arrays
4. Compresses with H.265 exploiting semantic-temporal redundancy
5. Decodes on-demand for retrieval

That's our novel contribution. The pieces all exist — they just haven't been assembled for this use case.
