"""OpenClaw ContextEngine plugin for aaa-memory."""
from aaa_memory.claude.hooks import store_turn, VAULT
from aaa_memory.retrieval.pipeline import search

class AaaMemoryPlugin:
    """OpenClaw plugin: before_prompt_build injects context, afterTurn stores."""
    
    def before_prompt_build(self, context: dict) -> dict:
        query = context.get("user_message", "")[:200]
        results = search(query, limit=5) if query else []
        context["memory_context"] = results
        return context
    
    def after_turn(self, turn: dict):
        store_turn("openclaw", turn.get("user_message", ""), turn.get("assistant_response", ""), turn.get("session_id"))
    
    def compact(self, session_id: str):
        """Summarize old context."""
        import sqlite3
        conn = sqlite3.connect(str(VAULT))
        conn.execute("UPDATE turns SET raw_text = substr(raw_text, 1, 500) WHERE session_id = ? AND timestamp < datetime('now', '-7 days')", (session_id,))
        conn.commit()
        conn.close()
