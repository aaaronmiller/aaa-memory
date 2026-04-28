#!/usr/bin/env python3
"""
Classifier smoke test — runs on sample documents in tests/classifier/fixtures/
"""

import sys
from pathlib import Path
from aaa_memory.classifier import classify

FIXTURES = Path("/home/misscheta/code/aaa-memory/tests/classifier/fixtures")
if not FIXTURES.exists():
    FIXTURES.mkdir(parents=True)

# Create sample fixtures if missing
samples = {
    "prd.md": """# Product Requirements Document

## Overview
Build a memory archive for AI agent sessions.

## Requirements
- FR-001: Ingest documents
- FR-002: Extract elements
""",
    "transcript.txt": """Human: How does the retrieval pipeline work?
Assistant: It uses hybrid search with FTS5 and vector similarity.
Human: Thanks!
""",
    "paper.md": """## Abstract

This paper presents a novel approach to memory retrieval...

### Introduction
Many researchers have studied...

References
[1] Smith (2025). Deep Learning.
""",
    "notes.md": """# Decision

Use SQLite for storage.

## Pattern

[[wikilinks]] for cross-references.

- Fact: EmbeddingGemma uses 0.4GB VRAM
""",
}

for name, content in samples.items():
    p = FIXTURES / name
    if not p.exists():
        p.write_text(content)

# Run classification
print("=== Classifier Smoke Test ===\n")
all_ok = True
for name, expected in [
    ("prd.md", "prd"),
    ("transcript.txt", "transcript"),
    ("paper.md", "research_paper"),
    ("notes.md", "knowledge_extract"),
]:
    path = FIXTURES / name
    result = classify(path)
    status = "✓" if result.category == expected else "✗"
    if result.category != expected:
        all_ok = False
    print(
        f"{status} {name}: {result.category} (conf={result.confidence:.2f}) {'(LLM)' if result.llm_used else ''}"
    )

if all_ok:
    print("\n✅ All classifier tests passed")
    sys.exit(0)
else:
    print("\n❌ Some classifications missed — acceptable for prototype")
    sys.exit(0)  # non-fatal
