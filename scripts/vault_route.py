#!/usr/bin/env python3
"""
Route raw files into category subdirectories based on classification_report.json.

Reads classification_report.json, moves each file into:
  raw/transcripts/  → for transcript category
  raw/prds/         → for prd
  raw/research/     → for research_paper
  raw/knowledge/    → for knowledge_extract
"""

import json
from pathlib import Path

RAW_BASE = Path("/home/misscheta/knowledge/raw")
REPORT_PATH = Path("/home/misscheta/knowledge/classification_report.json")


def main():
    if not REPORT_PATH.exists():
        print(f"ERROR: Classification report not found at {REPORT_PATH}")
        print("Run vault_classify.py first.")
        return 1

    report = json.loads(REPORT_PATH.read_text())
    moved = 0
    errors = 0

    for entry in report["files"]:
        src = RAW_BASE / entry["path"]
        if not src.exists():
            continue

        category = entry["category"]
        if category == "unknown":
            continue

        # Map category → subdir name
        subdir_map = {
            "prd": "prds",
            "transcript": "transcripts",
            "research_paper": "research",
            "knowledge_extract": "knowledge",
        }
        subdir = subdir_map.get(category)
        if not subdir:
            continue

        dst_dir = RAW_BASE / subdir
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name

        try:
            src.rename(dst)
            print(f"Moved: {src.name} → {subdir}/")
            moved += 1
        except Exception as e:
            print(f"Error moving {src}: {e}")
            errors += 1

    print(f"\n✅ Moved {moved} files, {errors} errors")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
