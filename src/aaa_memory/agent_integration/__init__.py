"""Agent integration package.

Each subpackage provides aaa-memory integration for a specific AI agent CLI tool.
Reference implementation: hermes (in-process Python plugin with lazy imports).
"""

from .hermes.provider import create_plugin as hermes_plugin
from .qwen.context import refresh_context as qwen_refresh
from .opencode import (
    parse_opencode_sessions,
    search as opencode_search,
    store_turn as opencode_store,
)
from .codex import parse_codex_rollouts, store_turn as codex_store
from .pi import search as pi_search, store_turn as pi_store

__all__ = [
    "hermes_plugin",
    "qwen_refresh",
    "parse_opencode_sessions",
    "opencode_search",
    "opencode_store",
    "parse_codex_rollouts",
    "codex_store",
    "pi_search",
    "pi_store",
]
