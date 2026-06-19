#!/usr/bin/env python3
"""
Batch extract elements from classified transcripts.

Reads classification_report.json, filters to 'transcript' category,
runs LLM extractor on each, writes extracted elements to wiki/ via injector.

Supports checkpointing via --resume-from flag (reads already-processed files).
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from aaa_memory.extractor.llm_extractor import extract_batch
from aaa_memory.metadata.injector import inject_batch

RAW_BASE = Path("/home/misscheta/knowledge/raw")
REPORT_PATH = Path("/home/misscheta/knowledge/classification_report.json")
CHECKPOINT_FILE = Path("/home/misscheta/knowledge/.extraction_checkpoint.json")


def load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        data = json.loads(CHECKPOINT_FILE.read_text())
        return set(data.get("processed", []))
    return set()


def save_checkpoint(processed: list):
    CHECKPOINT_FILE.write_text(
        json.dumps(
            {"processed": list(processed), "last_update": datetime.now().isoformat()}
        )
    )


def main(resume: bool = False):
    report = json.loads(REPORT_PATH.read_text())
    transcripts = [f for f in report["files"] if f["category"] == "transcript"]

    processed = load_checkpoint() if resume else set()
    total = len(transcripts)
    newly_processed = []

    print(f"Found {total} transcript files; {len(processed)} already processed")
    batch_size = 20

    # Map category → subdir name (same as in vault_route.py)
    subdir_map = {
        "prd": "prds",
        "transcript": "transcripts",
        "research_paper": "research",
        "knowledge_extract": "knowledge",
    }

    for i in range(0, total, batch_size):
        batch = transcripts[i : i + batch_size]
        for entry in batch:
            rel = entry["path"]
            if rel in processed:
                continue

            # Determine new location: raw/<category_subdir>/<filename>
            category = entry["category"]
            subdir = subdir_map.get(category)
            if not subdir:
                print(f"  ✋ Skipping {rel} (unknown category {category})")
                continue

            filename = Path(rel).name
            filepath = RAW_BASE / subdir / filename

            if not filepath.exists():
                print(f"  ⚠️  Missing: {filepath} (from {rel})")
                continue

            try:
                elements = extract_batch(
                    filepath, use_llm=False
                )  # LLM may fail, use fallback
                injected = inject_batch(
                    elements,
                    source_file=str(filepath.relative_to(RAW_BASE)),
                    project="vault",
                    agent="batch-extractor",
                )
                print(f"  ✓ {filepath.name} → {len(injected)} elements")
                newly_processed.append(rel)
                processed.add(rel)
                save_checkpoint(list(processed))
            except Exception as e:
                print(f"  ✗ {filepath.name}: {e}")

        print(
            f"Batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} done"
        )

    print(f"\nExtraction complete. {len(newly_processed)} new files processed.")
    print(f"Total processed: {len(processed)}/{total}")


if __name__ == "__main__":
    resume = "--resume" in sys.argv
    main(resume)
