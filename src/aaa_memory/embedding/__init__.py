"""Embedding package."""

from .encoder import (
    Embedder,
    OpenRouterEmbedder,
    Qwen3Embedder,
    Gemma300MEmbedder,
    JinaEmbedder,
    get_embedder,
    embed_to_base64,
    embed_from_base64,
)

__all__ = [
    "Embedder",
    "OpenRouterEmbedder",
    "Qwen3Embedder",
    "Gemma300MEmbedder",
    "JinaEmbedder",
    "get_embedder",
    "embed_to_base64",
    "embed_from_base64",
]
