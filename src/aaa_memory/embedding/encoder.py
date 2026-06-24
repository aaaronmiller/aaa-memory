"""
Embedding encoder — multi-provider with priority chain.

Priority order:
1. OpenRouter Qwen3-Embedding-8B (cloud, best cost/quality balance)
2. Qwen3-Embedding-8B local (GPU, vLLM or transformers)
3. EmbeddingGemma-300M local (GPU preferred, CPU only if unavoidable)
4. Jina v3 API (cloud fallback)

Embeddings stored in:
- Markdown frontmatter (base64-encoded, git-trackable)
- SQLite vec table (fast retrieval)
- Element metadata JSON (provenance)
"""

import base64
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
import openai

# ── Types ──────────────────────────────────────────────────────────────────────


@dataclass
class Embedding:
    """Container for an embedding vector with provenance."""

    model: str
    vector: np.ndarray  # shape (dim,)
    tokens: int
    provider: str  # 'openrouter', 'local-qwen3', 'local-gemma', 'cloud-jina'


# ── Abstract Base ─────────────────────────────────────────────────────────────


class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> Embedding: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...


# ── Provider 1: Qwen3-Embedding-8B (local) ────────────────────────────────────


class Qwen3Embedder(Embedder):
    """Local 8B embedding model via vLLM / Ollama / HuggingFace transformers."""

    def __init__(self, model_path: Optional[str] = None):
        # Try vLLM first, then Ollama, then transformers
        self.model_path = model_path or "Qwen/Qwen3-Embedding-8B"
        self.client = None  # lazy init

        # Prefer vLLM OpenAI-compatible endpoint
        self.use_vllm = os.getenv("VLLM_ENDPOINT") is not None
        if self.use_vllm:
            import openai

            self.client = openai.OpenAI(
                base_url=os.getenv("VLLM_ENDPOINT"), api_key="no-key"
            )
        else:
            # Fallback: transformers (slow, for Surface)
            from transformers import AutoTokenizer, AutoModel
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(
                self.model_path, device_map="auto", load_in_4bit=True
            )
            self.model.eval()

    def embed(self, text: str) -> Embedding:
        if self.use_vllm:
            # OpenAI-style embeddings endpoint
            resp = self.client.embeddings.create(model=self.model_path, input=text)
            vec = np.array(resp.data[0].embedding, dtype=np.float32)
            return Embedding(
                model=self.model_path,
                vector=vec,
                tokens=len(text.split()),
                provider="local-qwen3-vllm",
            )
        else:
            # Transformers mean-pooling
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Mean pooling over token dimension
            vec = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            return Embedding(
                model=self.model_path,
                vector=vec.astype(np.float32),
                tokens=inputs.input_ids.shape[1],
                provider="local-qwen3-tf",
            )

    @property
    def dimension(self) -> int:
        return 4096  # Qwen3-Embedding-8B output dim


class OpenRouterEmbedder(Embedder):
    """OpenRouter-hosted embeddings via OpenAI-compatible API."""

    DEFAULT_MODEL = "qwen/qwen3-embedding-8b"
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        http_referer: Optional[str] = None,
        app_title: Optional[str] = None,
    ):
        import openai

        self.model = model or os.getenv("OPENROUTER_EMBED_MODEL", self.DEFAULT_MODEL)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY required for OpenRouter embeddings")
        headers = {}
        if http_referer or os.getenv("OPENROUTER_HTTP_REFERER"):
            headers["HTTP-Referer"] = http_referer or os.getenv("OPENROUTER_HTTP_REFERER")
        if app_title or os.getenv("OPENROUTER_APP_TITLE"):
            headers["X-OpenRouter-Title"] = app_title or os.getenv("OPENROUTER_APP_TITLE")
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL,
            default_headers=headers or None,
        )
        self._dim = 4096

    def embed(self, text: str) -> Embedding:
        resp = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float",
        )
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        self._dim = len(vec)
        return Embedding(
            model=self.model,
            vector=vec,
            tokens=len(text.split()),
            provider="openrouter",
        )

    @property
    def dimension(self) -> int:
        return self._dim


# ── Provider 2: EmbeddingGemma-300M (local lightweight) ───────────────────────


class Gemma300MEmbedder(Embedder):
    """Lightweight embedding model using sentence-transformers.

    Tries to load embedding-gemma-300m if available; falls back to
    all-MiniLM-L6-v2 (384 dim) which is universally available and fast.
    """

    def __init__(self, model_name: str = "google/embedding-gemma-300m"):
        self.model_name = model_name
        from sentence_transformers import SentenceTransformer
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self._dim = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            print(
                f"[embedding] {model_name} unavailable ({e}), falling back to all-MiniLM-L6-v2"
            )
            self.model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
            self._dim = self.model.get_sentence_embedding_dimension()
        if self.device == "cuda":
            # Half precision for memory savings on GPU.
            self.model.half()

    def embed(self, text: str) -> Embedding:
        vec = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return Embedding(
            model=self.model_name,
            vector=vec.astype(np.float32),
            tokens=len(text.split()),
            provider="local-gemma",
        )

    @property
    def dimension(self) -> int:
        return self._dim


# ── Provider 3: Jina v3 API (cloud fallback) ──────────────────────────────────


class JinaEmbedder(Embedder):
    """Jina AI embeddings API (free tier ~1k/month)."""

    API_URL = "https://api.jina.ai/v1/embeddings"
    MODEL = "jina-embeddings-v3-base"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise RuntimeError("JINA_API_KEY required for cloud fallback")
        import httpx

        self.client = httpx.Client(timeout=30.0)

    def embed(self, text: str) -> Embedding:
        resp = self.client.post(
            self.API_URL,
            json={"model": self.MODEL, "input": text, "encoding_format": "float"},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()
        vec = np.array(data["data"][0]["embedding"], dtype=np.float32)
        return Embedding(
            model=self.MODEL,
            vector=vec,
            tokens=len(text.split()),
            provider="cloud-jina",
        )

    @property
    def dimension(self) -> int:
        return 768  # Jina v3 base dim


# ── Priority Chain ─────────────────────────────────────────────────────────────


def get_embedder(priority: str = "auto") -> Embedder:
    """
    Get the best available embedder following priority chain.

    priority: 'auto' | 'openrouter' | 'qwen3' | 'gemma' | 'jina'

    Notes:
    - OpenRouter is preferred when OPENROUTER_API_KEY is present
    - Qwen3-Embedding-8B requires vLLM endpoint or sufficient VRAM
    - Gemma uses sentence-transformers with CUDA when available
    - Jina requires API key
    """
    if priority == "auto":
        # 1. Cloud first, because it is the best cost/quality tradeoff and keeps
        #    us out of the CPU fallback path on this box.
        try:
            return OpenRouterEmbedder()
        except Exception as e:
            print(f"[embedding] OpenRouter unavailable: {e}, trying local GPU...")

        # 2. Try local GPU-backed Qwen3 when a vLLM endpoint is configured.
        if os.getenv("VLLM_ENDPOINT"):
            try:
                return Qwen3Embedder()
            except Exception as e:
                print(f"[embedding] Qwen3 local failed: {e}, trying Gemma...")

        # 3. Local lightweight fallback. Uses CUDA if present; CPU only if we
        #    have no GPU path and no cloud key.
        try:
            return Gemma300MEmbedder()
        except Exception as e:
            print(f"[embedding] Gemma failed: {e}, trying Jina...")
            pass

        # 2. Cloud fallback
        try:
            return JinaEmbedder()
        except Exception:
            pass

        raise RuntimeError(
            "No embedder available. Set OPENROUTER_API_KEY or JINA_API_KEY, or configure VLLM_ENDPOINT."
        )
    elif priority == "openrouter":
        return OpenRouterEmbedder()
    elif priority == "qwen3":
        return Qwen3Embedder()
    elif priority == "gemma":
        return Gemma300MEmbedder()
    elif priority == "jina":
        return JinaEmbedder()
    else:
        raise ValueError(f"Unknown priority: {priority}")


# ── Utilities ──────────────────────────────────────────────────────────────────


def embed_to_base64(embedding: Embedding) -> str:
    """Serialize embedding vector to base64 for markdown frontmatter."""
    raw = embedding.vector.tobytes()
    return base64.b64encode(raw).decode("ascii")


def embed_from_base64(b64: str, model: str, provider: str) -> Embedding:
    """Deserialize embedding from base64."""
    raw = base64.b64decode(b64)
    vec = np.frombuffer(raw, dtype=np.float32)
    return Embedding(model=model, vector=vec, tokens=0, provider=provider)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    text = sys.argv[1] if len(sys.argv) > 1 else "test sentence for embedding"
    embedder = get_embedder("auto")
    emb = embedder.embed(text)
    print(f"Model: {emb.model}")
    print(f"Provider: {emb.provider}")
    print(f"Dimension: {len(emb.vector)}")
    print(f"Tokens: {emb.tokens}")
    print(f"Base64: {embed_to_base64(emb)[:60]}...")
