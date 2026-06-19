#!/usr/bin/env python3
"""
Smoke-test the embedding pipeline.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from aaa_memory.embedding import get_embedder, embed_to_base64, embed_from_base64


def main():
    print("=== Embedding Smoke Test ===")
    try:
        embedder = get_embedder("gemma")  # force sentence-transformers
        print(f"✓ Embedder loaded: {embedder.__class__.__name__}")
        print(f"  Dimension: {embedder.dimension}")

        text = "SQLite is a file-portable SQL database with zero operational overhead."
        emb = embedder.embed(text)
        print(f"✓ Embedded '{text[:50]}...'")
        print(f"  Vector shape: {emb.vector.shape}")
        print(f"  Provider: {emb.provider}")

        b64 = embed_to_base64(emb)
        print(f"✓ Serialized to base64 ({len(b64)} chars)")

        restored = embed_from_base64(b64, emb.model, emb.provider)
        print(
            f"✓ Deserialized back — vector norm: {np.linalg.norm(restored.vector):.4f}"
        )

        # Verify round-trip equality
        assert np.allclose(emb.vector, restored.vector, atol=1e-5)
        print("✓ Round-trip verification passed")

        print("\n✅ Embedding pipeline functional")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import numpy as np

    sys.exit(main())
