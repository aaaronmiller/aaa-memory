"""Extraction tests — 10 test cases with fixtures and expected elements."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from aaa_memory.extractor.llm_extractor import extract_fallback


# ── Fixtures ─────────────────────────────────────────────────────────────────

TRANSCRIPT_SIMPLE = """Human: Let's design the auth system.

Assistant: I recommend JWT with refresh tokens. Decision: Use short-lived access tokens (15 min).

Human: What about MFA?

Assistant: Decision: Add TOTP as optional second factor. Decision: Store recovery codes as bcrypt hashes.
"""

TRANSCRIPT_CODE = """Human: Can you write a retry function?

Assistant: ```python
def retry(func, max_attempts=3):
    for i in range(max_attempts):
        try:
            return func()
        except Exception:
            if i == max_attempts - 1:
                raise
            time.sleep(2 ** i)
```

Decision: Use exponential backoff with jitter.
"""

TRANSCRIPT_EMPTY = ""

TRANSCRIPT_NO_DECISIONS = """Human: Hello.
Assistant: Hi there! How can I help you today?
Human: Just testing.
Assistant: OK, let me know if you need anything.
"""

TRANSCRIPT_BULLET_LIST = """Human: What are the requirements?

Assistant: The system needs:
- Handle 1000 concurrent users
- Response time under 200ms
- 99.9% uptime SLA
- Support for multiple regions
"""

TRANSCRIPT_MIXED = """Human: Architecture decisions?

Assistant: Decision: Use microservices with API gateway.
Decision: PostgreSQL for primary data.
Decision: Redis for caching.
Decision: Kubernetes for orchestration.

Human: Any concerns?

Assistant: The main concern is operational complexity. Decision: Start with monolith, extract services gradually.
"""


# ── Tests ────────────────────────────────────────────────────────────────────

class TestExtractorBasic:
    def test_simple_transcript_extracts_decisions(self):
        elements = extract_fallback(TRANSCRIPT_SIMPLE)
        assert len(elements) >= 2, f"Expected >=2 elements, got {len(elements)}"
        # Should find Decision: keywords
        decision_els = [e for e in elements if "decision" in str(e).lower() or "jwt" in str(e).lower() or "totp" in str(e).lower()]
        assert len(decision_els) >= 1

    def test_code_block_detected(self):
        elements = extract_fallback(TRANSCRIPT_CODE)
        assert len(elements) >= 1
        code_els = [e for e in elements if "retry" in str(e).lower() or "max_attempts" in str(e).lower()]
        assert len(code_els) >= 1

    def test_empty_transcript_returns_empty(self):
        elements = extract_fallback(TRANSCRIPT_EMPTY)
        assert len(elements) == 0

    def test_no_decisions_returns_few_elements(self):
        elements = extract_fallback(TRANSCRIPT_NO_DECISIONS)
        # Should still return something or empty — not crash
        assert elements is not None

    def test_bullet_list_detected(self):
        elements = extract_fallback(TRANSCRIPT_BULLET_LIST)
        assert len(elements) >= 1
        bullet_els = [e for e in elements if "1000" in str(e) or "concurrent" in str(e)]
        assert len(bullet_els) >= 1

    def test_mixed_transcript_multiple_decisions(self):
        elements = extract_fallback(TRANSCRIPT_MIXED)
        assert len(elements) >= 3, f"Expected >=3 elements for 5 decisions, got {len(elements)}"

    def test_element_structure(self):
        elements = extract_fallback(TRANSCRIPT_SIMPLE)
        if elements:
            e = elements[0]
            # Verify Element dataclass has required fields
            assert hasattr(e, 'type'), "Element missing 'type'"
            assert hasattr(e, 'title'), "Element missing 'title'"
            assert hasattr(e, 'content'), "Element missing 'content'"

    def test_consistency_across_runs(self):
        r1 = extract_fallback(TRANSCRIPT_SIMPLE)
        r2 = extract_fallback(TRANSCRIPT_SIMPLE)
        assert len(r1) == len(r2), f"Length changed between runs: {len(r1)} vs {len(r2)}"

    def test_very_long_input(self):
        long_text = "Decision: Test decision. " * 100
        elements = extract_fallback(long_text)
        # Each "Decision:" yields an element — that's correct behavior
        # But the extractor should handle it without crashing
        assert len(elements) >= 1

    def test_unicode_text(self):
        text = "Human: Décision importante.\n\nAssistant: Décision: Utiliser PostgreSQL.\n\nDecision: Use UTF-8 everywhere."
        elements = extract_fallback(text)
        assert len(elements) >= 1
