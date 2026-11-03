"""AAA-Memory public API, lazily loaded to keep capture independent of LLM dependencies.

Legacy public exports remain available via PEP 562 module __getattr__.
"""
from importlib import import_module

__version__ = "0.1.0"
__author__ = "Clawdi"

_EXPORTS = {
    "Turn": (".models", "Turn"),
    "Element": (".models", "Element"),
    "WikiPage": (".models", "WikiPage"),
    "GraphEpisode": (".models", "GraphEpisode"),
    "classify_document": (".classifier", "classify"),
    "extract": (".extractor", "extract"),
    "ExtractedElement": (".extractor", "Element"),
    "get_embedder": (".embedding", "get_embedder"),
    "hot_search": (".retrieval", "hot_search"),
    "rrf_fusion": (".retrieval", "rrf_fusion"),
    "classify_intent": (".router", "classify_intent"),
    "discover_sessions": (".audit", "discover_sessions"),
    "assemble_timeline": (".audit", "assemble_timeline"),
    "cli_main": (".cli", "main"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
