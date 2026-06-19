#!/usr/bin/env python3
"""Route vault files into raw/<type>/ subdirectories based on classification."""
import json, shutil, sys
from pathlib import Path

CLASSIFICATION_REPORT = Path("classification_report.json")
RAW_DIR = Path.home() / "knowledge/raw"

def route(classification_report: str):
    with open(classification_report) as f:
        entries = json.load(f)
    for entry in entries:
        fpath = Path(entry["file"])
        ftype = entry.get("classification", "unknown")
        target = RAW_DIR / ftype
        target.mkdir(parents=True, exist_ok=True)
        if fpath.exists():
            shutil.move(str(fpath), str(target / fpath.name))
            print(f"  {fpath.name} -> {ftype}/")

if __name__ == "__main__":
    route(sys.argv[1] if len(sys.argv) > 1 else str(CLASSIFICATION_REPORT))
