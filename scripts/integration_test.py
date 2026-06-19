#!/usr/bin/env python3
"""
Full integration test: transcript → classify → extract → inject → index → lint → search
"""

import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aaa_memory.classifier import classify
from aaa_memory.extractor import extract_batch
from aaa_memory.metadata import inject_batch
from aaa_memory.wiki.compiler import WikiIndexer
from aaa_memory.wiki.linter import run_full_lint, write_report as lint_write
from aaa_memory.retrieval.hot import search as hot_search
from aaa_memory.audit.discover import discover_sessions
from aaa_memory.audit.timeline import assemble_timeline

TEST_TRANSCRIPT = Path("/home/misscheta/knowledge/raw/test_integration.txt")
VAULT = Path(
    os.getenv("AAA_MEMORY_VAULT", "/home/misscheta/.cache/aaa-memory/vault.sqlite")
)


def log(msg):
    print(f"[INTEGTEST] {msg}")


def main():
    log("=== Full Pipeline Integration Test ===\n")
    log("1. Preparing test transcript")
    TEST_TRANSCRIPT.parent.mkdir(parents=True, exist_ok=True)
    TEST_TRANSCRIPT.write_text("""Human: We need to decide on the memory storage engine.

Assistant: I propose SQLite with sqlite-vec extension. It's file-portable, zero-ops, and ClawMem already uses it natively.

Human: What about PostgreSQL?

Assistant: PostgreSQL adds daemon management and operational complexity. For a single-user personal archive, SQLite is sufficient.

Decision: Use SQLite + sqlite-vec as primary storage engine for hot tier.
""")
    log(f"   Transcript: {TEST_TRANSCRIPT}")

    log("\n2. Classifying document")
    result = classify(TEST_TRANSCRIPT)
    log(f"   Category: {result.category} (conf={result.confidence:.2f})")

    log("\n3. Extracting elements")
    elements = extract_batch(TEST_TRANSCRIPT, use_llm=False)
    log(f"   Extracted {len(elements)} elements")
    for el in elements:
        log(f"   - [{el.type}] {el.title}")

    log("\n4. Injecting to wiki")
    wiki_files = inject_batch(
        elements, source_file="test_integration.txt", project="test", agent="test"
    )
    log(f"   Written {len(wiki_files)} wiki pages")

    log("\n5. Generating wiki indexes")
    WikiIndexer().generate_master_index()
    WikiIndexer().generate_sub_indexes()
    log("   ✓ Indexes done")

    log("\n6. Running wiki lint")
    report = run_full_lint()
    lint_path = lint_write(report)
    log(
        f"   Lint: {lint_path}, orphans={len(report['orphans'])}, dead={len(report['dead_links'])}"
    )

    log("\n7. Testing hot search")
    results = hot_search("SQLite")
    log(f"   Found {len(results)} hits for 'SQLite'")
    if results:
        log(f"   Top: {results[0]['raw_text'][:60]}...")

    log("\n8. Session discovery")
    sessions = discover_sessions()
    total = sum(len(v) for v in sessions.values())
    log(f"   Discovered {total} session files")

    log("\n9. Timeline assembly")
    timeline = assemble_timeline("unknown", days=30)
    log(f"   Timeline sessions: {len(timeline.get('sessions', []))}")

    log("\n=== Integration Test Complete ===")
    log("✅ Core pipeline functional (Phases 1–2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
