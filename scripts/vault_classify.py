#!/usr/bin/env python3
"""
Batch classify vault files — writes classification_report.json.

Walks ~/knowledge/raw/ recursively, classifies each file, outputs
counts per category and per-agent breakdown.
"""

import json
from pathlib import Path
from datetime import datetime
from aaa_memory.classifier import classify

RAW_BASE = Path("/home/misscheta/knowledge/raw")
REPORT_PATH = Path("/home/misscheta/knowledge/classification_report.json")


def main():
    results = []
    stats = {"total": 0, "by_category": {}, "by_agent": {}, "errors": []}

    for filepath in RAW_BASE.rglob("*"):
        if not filepath.is_file():
            continue
        try:
            result = classify(filepath)
            results.append(
                {
                    "path": str(filepath.relative_to(RAW_BASE)),
                    "category": result.category,
                    "confidence": result.confidence,
                    "rule_match": result.rule_match,
                    "llm_used": result.llm_used,
                }
            )
            stats["total"] += 1
            cat = result.category
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

            # agent is derived from parent dir name of raw/
            # e.g., raw/transcripts/file.txt → agent = 'transcripts'
            agent = filepath.parent.name
            stats["by_agent"][agent] = stats["by_agent"].get(agent, 0) + 1
        except Exception as e:
            stats["errors"].append({"file": str(filepath), "error": str(e)})

    # Write report
    report = {
        "generated_at": datetime.now().isoformat(),
        "stats": stats,
        "files": results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Classified {stats['total']} files → {REPORT_PATH}")
    print("Breakdown:", stats["by_category"])


if __name__ == "__main__":
    main()
