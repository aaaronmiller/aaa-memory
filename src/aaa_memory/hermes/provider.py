"""Hermes MemoryProvider for aaa-memory."""
from aaa_memory.retrieval.pipeline import search as memory_search
from aaa_memory.claude.hooks import store_turn

class HermesMemoryProvider:
    """Implements Hermes MemoryProvider ABC."""
    
    def search(self, query: str, limit: int = 10) -> list:
        return memory_search(query, limit=limit)
    
    def store(self, turn: dict):
        store_turn("hermes", turn.get("prompt", ""), turn.get("response", ""), turn.get("session_id"))
    
    def health_check(self) -> dict:
        return {"status": "ok", "provider": "aaa-memory"}
