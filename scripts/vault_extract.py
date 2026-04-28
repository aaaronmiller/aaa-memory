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
    for i in range(0, total, batch_size):
        batch = transcripts[i : i + batch_size]
        for entry in batch:
            rel = entry["path"]
            if rel in processed:
                continue
            try:
                filepath = RAW_BASE / rel
                elements = extract_batch(filepath, use_llm=True)
                # Inject to wiki
                injected = inject_batch(
                    elements, source_file=rel, project="vault", agent="batch-extractor"
                )
                print(f"  ✓ {rel} → {len(injected)} elements")
                newly_processed.append(rel)
                # Flush checkpoint every file
                processed.add(rel)
                save_checkpoint(list(processed))
            except Exception as e:
                print(f"  ✗ {rel}: {e}")

        print(
            f"Batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} done"
        )

    print(f"\nExtraction complete. {len(newly_processed)} new files processed.")
    print(f"Total processed: {len(processed)}/{total}")


if __name__ == "__main__":
    resume = "--resume" in sys.argv
    main(resume)
