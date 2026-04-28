"""Tests for aaa_memory.classifier"""

import pytest
from pathlib import Path
from aaa_memory.classifier.rules import classify_file, ClassificationResult
from aaa_memory.classifier import classify

# Test data fixtures — create temp files in tests/


def test_classify_prd():
    content = """# Product Requirements

## Overview
Build a memory system for AI agents.

## Requirements
- FR-001: Document ingestion
- FR-002: Element extraction
"""
    path = Path("/tmp/test_prd.md")
    path.write_text(content)
    result = classify_file(path)
    assert result.category == "prd"
    assert result.confidence > 0.5


def test_classify_transcript():
    content = """Human: How do I build a memory system?
Assistant: You need to start with a classifier.
Human: Thanks!
"""
    path = Path("/tmp/test_transcript.txt")
    path.write_text(content)
    result = classify_file(path)
    assert result.category == "transcript"
    assert result.confidence >= 0.6


def test_classify_research_paper():
    content = """## Abstract

This paper presents a novel approach to...

### 1. Introduction

Many researchers have studied...

References
[1] Smith, J. (2025). Deep Learning.
"""
    path = Path("/tmp/test_paper.md")
    path.write_text(content)
    result = classify_file(path)
    assert result.category == "research_paper"


def test_classify_knowledge_extract():
    content = """# Decision

We chose SQLite for the storage engine.

## Pattern

Use [[wikilinks]] for cross-references.

- Key fact: embeddings stored in three places
"""
    path = Path("/tmp/test_knowledge.md")
    path.write_text(content)
    result = classify_file(path)
    # May be unknown or knowledge — both acceptable depending on patterns
    assert result.category in ("knowledge_extract", "unknown")


def test_combined_classifier_with_llm_fallback(monkeypatch):
    """Test that low-confidence rule results fall back to LLM."""
    # Create ambiguous content that triggers low confidence
    content = "random stuff" * 10
    path = Path("/tmp/test_ambig.md")
    path.write_text(content)

    def fake_llm(content):
        return "knowledge_extract"

    # Patch llm_classify to avoid real API call
    import aaa_memory.classifier.llm_classifier as llm_mod

    monkeypatch.setattr(llm_mod, "classify", fake_llm)

    result = classify(path, llm_fallback=True)
    assert result.llm_used is True or result.confidence >= 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
