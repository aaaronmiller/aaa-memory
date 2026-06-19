"""Integration tests — real API calls, real vault, all agents simulated."""
import os
import sys
import json
import sqlite3
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from aaa_memory import config as test_config
from aaa_memory.claude.hooks import store_turn
from aaa_memory.retrieval.pipeline import search
from aaa_memory.audit.embed_sessions import summarize_session
from aaa_memory.classifier.tuned import classify
from aaa_memory.extractor.llm_extractor import extract_fallback


@pytest.fixture(autouse=True)
def clean_vault():
    """Use a temp vault for each test so state doesn't leak."""
    tmp = tempfile.mkdtemp()
    old_vault = os.environ.get("AAA_MEMORY_VAULT")
    os.environ["AAA_MEMORY_VAULT"] = str(Path(tmp) / "test_vault.sqlite")
    import importlib
    import aaa_memory.config
    importlib.reload(aaa_memory.config)
    yield
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    if old_vault:
        os.environ["AAA_MEMORY_VAULT"] = old_vault


class TestClaudeCodeIntegration:
    def test_store_turn_creates_vault(self):
        tid = store_turn("claude-code", "What is the capital?",
                        "Paris.", session_id="test-session")
        vault = test_config.get_vault()
        assert vault.exists()
        conn = sqlite3.connect(str(vault))
        count = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        conn.close()
        assert count == 1
        print(f"  Vault: {count} turn(s)")

    def test_multiple_turns_same_session(self):
        sid = "session-multi-test"
        for i in range(3):
            store_turn("claude-code", f"Q{i}", f"A{i}", session_id=sid)
        vault = test_config.get_vault()
        conn = sqlite3.connect(str(vault))
        count = conn.execute(
            "SELECT COUNT(*) FROM turns WHERE session_id = ?", (sid,)
        ).fetchone()[0]
        conn.close()
        assert count == 3

    def test_turn_content_preserved(self):
        tid = store_turn("claude-code", "Explain quantum computing.",
                        "Qubits can be 0 and 1 simultaneously.",
                        session_id="content-test")
        vault = test_config.get_vault()
        conn = sqlite3.connect(str(vault))
        raw = conn.execute(
            "SELECT raw_text FROM turns WHERE turn_id = ?", (tid,)
        ).fetchone()[0]
        conn.close()
        assert "qubits" in raw.lower()

    def test_search_finds_stored_turns(self):
        store_turn("claude-code", "Design a REST API for a todo app.",
                   "Use GET /todos, POST /todos, DELETE /todos/:id",
                   session_id="search-test")
        from aaa_memory.retrieval.hot import search as hot_search
        results = hot_search("REST API", limit=5)
        assert len(results) > 0, f"Expected results, got {len(results)}"
        print(f"  Search: {len(results)} result(s)")


class TestPipeline:
    def test_classify_extract_chain(self):
        text = "Human: Let's design caching.\n\nAssistant: Decision: Use Redis."
        result = classify("test.md", content=text, llm_fallback=False)
        assert result.category in ("transcript", "prd", "research_paper", "knowledge_extract")
        elements = extract_fallback(text)
        assert len(elements) >= 1

    def test_end_to_end_roundtrip(self):
        import uuid
        sid = str(uuid.uuid4())[:12]
        store_turn("claude-code", "How do I implement JWT?",
                   "Decision: Use JWT with refresh tokens.",
                   session_id=sid)
        from aaa_memory.retrieval.hot import search as hot_search
        # Use single-word query for FTS5 (multi-word OR has edge cases)
        results = hot_search("JWT", limit=5)
        assert len(results) > 0, f"Expected JWT results, got 0"
        summary = summarize_session(sid)
        assert summary.get("turns", 0) >= 1, f"Expected >=1 turn, got {summary}"


class TestSessionAudit:
    def test_embed_sessions(self):
        for i in range(2):
            store_turn("claude-code", f"Q{i}", f"Decision: Use approach {i}.",
                      session_id="audit-session")
        summary = summarize_session("audit-session")
        assert summary.get("turns", 0) >= 2, f"Expected >=2 turns, got {summary}"

    def test_session_discovery(self):
        store_turn("claude-code", "test", "data", session_id="discover-me")
        vault = test_config.get_vault()
        conn = sqlite3.connect(str(vault))
        count = conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM turns"
        ).fetchone()[0]
        conn.close()
        assert count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
