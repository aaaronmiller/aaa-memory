"""aaa-memory — Personal AI Interaction Archive.

A multi-tier memory system for AI agent sessions.
"""

__version__ = "0.1.0"
__author__ = "Clawdi"

from .models import Turn, Element, WikiPage, GraphEpisode
from .classifier import classify as classify_document
from .extractor import extract, Element as ExtractedElement
from .embedding import get_embedder
from .retrieval import hot_search, rrf_fusion
from .router import classify_intent
from .audit import discover_sessions, assemble_timeline
from .cli import main as cli_main

__all__ = [
    "Turn",
    "Element",
    "WikiPage",
    "GraphEpisode",
    "classify_document",
    "extract",
    "ExtractedElement",
    "get_embedder",
    "hot_search",
    "rrf_fusion",
    "classify_intent",
    "discover_sessions",
    "assemble_timeline",
    "cli_main",
]
