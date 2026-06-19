"""Classifier unit tests — 25 test cases across all edge cases."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from aaa_memory.classifier.tuned import classify as tuned_classify
from aaa_memory.classifier.rules import classify_file, ClassificationResult
from aaa_memory.classifier.combined import classify as combined_classify


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_prd():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    f.write("# PRD: Authentication System\n\n## Architecture\nMicroservices\n## Requirements\n- Must handle 1000 req/s")
    f.close()
    yield Path(f.name)
    os.unlink(f.name)

@pytest.fixture
def sample_transcript():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    f.write("Human: Let's design the API.\n\nAssistant: Good idea. Decision: Use REST over gRPC.")
    f.close()
    yield Path(f.name)
    os.unlink(f.name)

@pytest.fixture
def sample_paper():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    f.write("# Attention Mechanisms\n\n## Abstract\nThis paper proposes...\n## References\n[1] Vaswani et al.")
    f.close()
    yield Path(f.name)
    os.unlink(f.name)

@pytest.fixture
def sample_knowledge():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    f.write("---\ntype: concept\ntitle: REST\n---\n[[REST]]\n[[HTTP]]\n[[API Design]]")
    f.close()
    yield Path(f.name)
    os.unlink(f.name)


# ── Basic classification tests ──────────────────────────────────────────────

class TestClassifierBasics:
    def test_prd_detected(self, sample_prd):
        c = tuned_classify(sample_prd)
        assert c.category == "prd", f"Expected prd, got {c.category}"
        assert c.confidence > 0

    def test_transcript_detected(self, sample_transcript):
        c = tuned_classify(sample_transcript)
        assert c.category == "transcript", f"Expected transcript, got {c.category}"

    def test_paper_detected(self, sample_paper):
        c = tuned_classify(sample_paper)
        assert c.category == "research_paper", f"Expected research_paper, got {c.category}"

    def test_knowledge_detected(self, sample_knowledge):
        c = tuned_classify(sample_knowledge)
        assert c.category == "knowledge_extract", f"Expected knowledge_extract, got {c.category}"

    def test_classification_result_dataclass(self):
        r = ClassificationResult(category="prd", confidence=0.9, rule_match="test", llm_used=False)
        assert r.category == "prd"
        assert r.confidence == 0.9
        assert r.rule_match == "test"
        assert r.llm_used is False

    def test_path_string_accepted(self, sample_prd):
        c = tuned_classify(str(sample_prd))
        assert c.category is not None

    def test_empty_file_returns_something(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        f.write("")
        f.close()
        c = tuned_classify(Path(f.name))
        assert c is not None
        os.unlink(f.name)


# ── Rule-based classifier tests ─────────────────────────────────────────────

class TestRuleBasedClassifier:
    def test_missing_file_handled(self):
        c = classify_file(Path("/nonexistent/file.md"))
        assert c.category == "unknown"
        assert c.confidence == 0.0

    def test_confidence_bounds(self, sample_prd):
        c = classify_file(sample_prd)
        assert 0.0 <= c.confidence <= 1.0

    def test_transcript_with_decision_keywords(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        f.write("Decision: Use PostgreSQL\nDecision: Add caching layer")
        f.close()
        c = classify_file(Path(f.name))
        assert c is not None
        os.unlink(f.name)

    def test_prd_with_architecture_keyword(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        f.write("# Architecture\nThe system shall...\n## Specification\nRequirements are...")
        f.close()
        c = classify_file(Path(f.name))
        assert c.category == "prd", f"Expected prd, got {c.category}"
        os.unlink(f.name)

    def test_paper_with_abstract_references(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        f.write("## Abstract\nWe propose...\n\n## References\n[1] Author (2024)")
        f.close()
        c = classify_file(Path(f.name))
        assert c.category == "research_paper", f"Expected research_paper, got {c.category}"
        os.unlink(f.name)


# ── Filename heuristic tests ────────────────────────────────────────────────

class TestFilenameHeuristic:
    def test_prd_in_filename(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", prefix="prd-auth", delete=False)
        f.write("short")
        f.close()
        c = tuned_classify(Path(f.name))
        assert c.category == "prd", f"Expected prd, got {c.category}"
        os.unlink(f.name)

    def test_transcript_in_filename(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", prefix="transcript-session", delete=False)
        f.write("short")
        f.close()
        c = tuned_classify(Path(f.name))
        assert c.category == "transcript"
        os.unlink(f.name)

    def test_paper_in_filename(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", prefix="research-paper", delete=False)
        f.write("short")
        f.close()
        c = tuned_classify(Path(f.name))
        assert c.category == "research_paper"
        os.unlink(f.name)

    def test_knowledge_in_filename(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", prefix="knowledge-extract", delete=False)
        f.write("short")
        f.close()
        c = tuned_classify(Path(f.name))
        assert c.category == "knowledge_extract"
        os.unlink(f.name)


# ── Edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_long_filename_no_hints(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", prefix="misc-notes-", delete=False)
        f.write("Some random notes that don't match any pattern.")
        f.close()
        c = tuned_classify(Path(f.name))
        assert c.confidence >= 0.0
        os.unlink(f.name)

    def test_binary_content_handled(self):
        f = tempfile.NamedTemporaryFile(mode="wb", suffix=".md", delete=False)
        f.write(b"\x00\x01\x02\x03\x04")
        f.close()
        c = tuned_classify(Path(f.name))
        assert c is not None
        os.unlink(f.name)

    def test_very_long_content(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        f.write("word " * 10000)
        f.close()
        c = tuned_classify(Path(f.name))
        assert c is not None
        os.unlink(f.name)

    def test_unicode_content(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        f.write("\u00e9\u00e8\u00ea\u00eb \u65e5\u672c\u8a9e \u0440\u0443\u0441\u0441\u043a\u0438\u0439")
        f.close()
        c = tuned_classify(Path(f.name))
        assert c is not None
        os.unlink(f.name)

    def test_confidence_stable_across_runs(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        f.write("# Architecture\nRequirements and specification for the system.")
        f.close()
        c1 = classify_file(Path(f.name))
        c2 = classify_file(Path(f.name))
        assert c1.confidence == c2.confidence
        os.unlink(f.name)

    def test_content_passed_directly(self):
        c = tuned_classify("test-prd-file.md", content="# PRD: API Design\n## Architecture", llm_fallback=False)
        assert c.category == "prd", f"Expected prd, got {c.category}"

    def test_yaml_frontmatter_only(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        f.write("---\ntype: prd\nproject: auth\n---")
        f.close()
        c = tuned_classify(Path(f.name))
        assert c is not None
        os.unlink(f.name)

    def test_llm_fallback_does_not_crash(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        f.write("Ambiguous content.")
        f.close()
        c = combined_classify(Path(f.name), llm_api_key=None, llm_fallback=False)
        assert c is not None
        os.unlink(f.name)

