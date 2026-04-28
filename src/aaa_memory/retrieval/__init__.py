"""Retrieval package."""

from .hot import search as hot_search
from .warm import search_relationship as warm_search
from .cold import search_archive as cold_search
from .fusion import (
    rrf_fusion,
    rerank_top_k,
    enforce_token_budget,
    strip_echo_cycles,
    detect_echo,
)

__all__ = [
    "hot_search",
    "warm_search",
    "cold_search",
    "rrf_fusion",
    "rerank_top_k",
    "enforce_token_budget",
    "strip_echo_cycles",
    "detect_echo",
]
