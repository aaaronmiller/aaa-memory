"""End-to-end integration tests for aaa-memory pipeline."""
import pytest, tempfile, json, os
from pathlib import Path

@pytest.fixture
def sample_transcript():
    return """
Human: Hey, let's design the auth system.

Assistant: I think we should use JWT with refresh tokens. The access token can be short-lived (15 min) and the refresh token can last 7 days.

Decision: Use JWT with refresh tokens.

Human: What about the database schema?

Assistant: Decision: We'll use PostgreSQL with a users table and a sessions table. The sessions table will store refresh tokens with expiry.
"""

def test_classifier_rules():
    from aaa_memory.classifier.rules import classify
    result = classify("test.md", "")
    assert result is not None

def test_extractor_basic():
    from aaa_memory.extractor.llm_extractor import extract_fallback
    elements = extract_fallback("Human: We decided on option A.\n\nAssistant: That was the right call. Decision: Use option A for the architecture.")
    assert len(elements) > 0

def test_decision_extractor():
    from aaa_memory.audit.extract_decisions import extract_decisions
    decisions = extract_decisions("We decided to use JWT. Rejected the OAuth approach. Decision: JWT with refresh.")
    assert len(decisions) >= 1
    assert any("JWT" in d.title for d in decisions)

def test_cold_storage():
    from aaa_memory.retrieval.cold import _ensure_schema, search_archive, COLD_VAULT
    _ensure_schema()
    assert COLD_VAULT.exists() or COLD_VAULT.parent.exists()

def test_intent_classifier():
    from aaa_memory.router.intent import classify_intent, Intent
    result = classify_intent("what did we discuss yesterday")
    assert result.intent in [e.value for e in Intent]
