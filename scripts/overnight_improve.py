#!/usr/bin/env python3
"""Overnight improvement: re-encode low-confidence elements."""
import json, os
from pathlib import Path

CHECKPOINT_FILE = "extraction_checkpoint.json"

def run():
    print("=" * 50)
    print("Overnight Improvement Loop")
    print("=" * 50)
    
    if not os.path.exists(CHECKPOINT_FILE):
        print("No checkpoint data found")
        return
    
    with open(CHECKPOINT_FILE) as f:
        data = json.load(f)
    
    improved = 0
    for entry in data:
        for elem in entry.get("elements", []):
            confidence = elem.get("confidence", 0)
            if isinstance(confidence, (int, float)) and 0 < confidence < 0.7:
                # Mark for re-review
                elem["needs_review"] = True
                improved += 1
    
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Marked {improved} low-confidence elements for re-review")

if __name__ == "__main__":
    run()
