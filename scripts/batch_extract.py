#!/usr/bin/env python3
"""Batch element extraction with checkpoint support."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aaa_memory.extractor.llm_extractor import extract

BATCH_SIZE = 20
CHECKPOINT_FILE = "extraction_checkpoint.json"

def process_batch(files, batch_num):
    results = []
    for fpath in files:
        text = Path(fpath).read_text(encoding="utf-8", errors="replace")
        elements = extract(text)
        results.append({"file": str(fpath), "elements": [e.__dict__ if hasattr(e, '__dict__') else e for e in elements]})
    # Save checkpoint
    checkpoint = {"batch": batch_num, "files_processed": len(files), "results": results}
    with open(f"checkpoint_{batch_num:04d}.json", "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)
    # Update cumulative
    cumulative = []
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            cumulative = json.load(f)
    cumulative.extend(results)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cumulative, f, indent=2, default=str)
    return results

if __name__ == "__main__":
    import glob
    files = sorted(glob.glob(sys.argv[1]) if len(sys.argv) > 1 else glob.glob("raw/transcripts/*.jsonl"))
    for i in range(0, len(files), BATCH_SIZE):
        batch = files[i:i+BATCH_SIZE]
        process_batch(batch, i // BATCH_SIZE + 1)
        print(f"Checkpoint {i // BATCH_SIZE + 1}: {len(batch)} files")
