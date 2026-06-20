"""Hermes agent integration provider — wraps the real HermesMemoryProvider."""

from aaa_memory.hermes.provider import HermesMemoryProvider


def create_plugin(config: dict = None) -> HermesMemoryProvider:
    """Create and return a HermesMemoryProvider instance."""
    return HermesMemoryProvider()
