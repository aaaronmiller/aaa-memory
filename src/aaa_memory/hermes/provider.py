"""
Hermes integration — MemoryProvider ABC plugin.

Hermes agents can load custom MemoryProvider plugins to persist context.
This plugin routes all session storage to aaa-memory.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from aaa_memory.models import Turn
from aaa_memory.cli import main as aaa_memory_store


class HermesMemoryProvider(ABC):
    """Minimal Hermes MemoryProvider ABC subset."""

    @abstractmethod
    def store(self, turn: Turn) -> None:
        """Persist a single turn."""
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[Turn]:
        """Semantic search over stored turns."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        pass


# ── Concrete implementation (Hermes → aaa-memory) ────────────────────────────


class AaaMemoryProvider(HermesMemoryProvider):
    """
    Bridge: Hermes calls → aaa-memory CLI / direct SQLite writes.

    For production, use direct DB access; CLI used for prototyping.
    """

    def __init__(
        self, vault_path: Path = Path("/home/misscheta/.cache/clawmem/index.sqlite")
    ):
        self.vault = vault_path

    def store(self, turn: Turn) -> None:
        # Append to vault turns table via SQLite
        import sqlite3

        conn = sqlite3.connect(str(self.vault))
        conn.execute(
            """
            INSERT OR IGNORE INTO turns
            (turn_id, agent, session_id, turn_index, turn_type, raw_text, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                turn.turn_id,
                turn.agent,
                turn.session_id,
                turn.turn_index,
                turn.turn_type,
                turn.raw_text,
                turn.created_at,
                turn.metadata,
            ),
        )
        conn.commit()
        conn.close()

    def search(self, query: str, limit: int = 10) -> List[Turn]:
        # Defer to retrieval.hot search (placeholder)
        # Return mock
        return []

    def health_check(self) -> bool:
        return self.vault.exists()


# ── Hermes plugin entry point ──────────────────────────────────────────────────


def create_plugin():
    """Hermes loads this function to instantiate the provider."""
    return AaaMemoryProvider()


if __name__ == "__main__":
    # Quick sanity check
    provider = create_plugin()
    print("Health:", provider.health_check())
    # Create dummy turn
    dummy = Turn(
        turn_id="test-turn",
        agent="hermes-test",
        session_id="demo",
        turn_index=0,
        turn_type="user",
        raw_text="Hello from Hermes test",
        created_at="2026-04-27T18:00:00Z",
        metadata="{}",
    )
    provider.store(dummy)
    print("Stored test turn OK")
