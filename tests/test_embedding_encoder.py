import numpy as np


def test_openrouter_embedder_uses_openrouter_api(monkeypatch):
    from aaa_memory.embedding.encoder import OpenRouterEmbedder
    import aaa_memory.embedding.encoder as encoder

    class FakeEmbeddings:
        def create(self, model, input, encoding_format):
            assert model == "qwen/qwen3-embedding-8b"
            assert input == "hello world"
            assert encoding_format == "float"
            return type("Resp", (), {
                "data": [type("Item", (), {"embedding": [0.1, 0.2, 0.3]})()]
            })()

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.embeddings = FakeEmbeddings()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(encoder.openai, "OpenAI", FakeClient)

    emb = OpenRouterEmbedder().embed("hello world")

    assert emb.provider == "openrouter"
    assert emb.model == "qwen/qwen3-embedding-8b"
    assert np.allclose(emb.vector, np.array([0.1, 0.2, 0.3], dtype=np.float32))


def test_auto_prefers_openrouter_when_available(monkeypatch):
    import aaa_memory.embedding.encoder as encoder

    class FakeOpenRouter:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(encoder.openai, "OpenAI", FakeOpenRouter)

    embedder = encoder.get_embedder("auto")
    assert embedder.__class__.__name__ == "OpenRouterEmbedder"

