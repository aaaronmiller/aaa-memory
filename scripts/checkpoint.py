#!/usr/bin/env python3
"""
Vault refinement checkpoints — analyze extraction quality at milestones.

Checkpoints:
  15%: Validate extraction quality on first batch; adjust prompt if approval < 70%
  30%: Cross-reference elements; merge duplicate concepts; wikilink graph validation
  50%: Schema evolution; identify new metadata fields; detect new topic clusters
  75%: Graph density analysis; orphan detection; suggest wikilinks
  100%: Full wiki lint; re-embed all if model improved; generate timelines; final report
"""

import json
from pathlib import Path
from datetime import datetime
from aaa_memory.wiki.linter import run_full_lint

CHECKPOINT_DIR = Path("/home/misscheta/knowledge/checkpoints")
REPORT_PATH = Path("/home/misscheta/knowledge/classification_report.json")


def load_report():
    if REPORT_PATH.exists():
        return json.loads(REPORT_PATH.read_text())
    return {"files": []}


def count_total_transcripts():
    report = load_report()
    return len([f for f in report["files"] if f["category"] == "transcript"])


def count_processed_wiki():
    wiki = Path("/home/misscheta/knowledge/wiki")
    return sum(1 for _ in wiki.rglob("*.md"))


def estimate_progress():
    total = count_total_transcripts()
    processed = count_processed_wiki()
    if total == 0:
        return 0
    return int((processed / total) * 100)


def run_checkpoint(percentage: int):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    out = CHECKPOINT_DIR / f"checkpoint_{percentage:03d}.md"

    lines = [
        f"# Vault Migration Checkpoint — {percentage}% Complete",
        f"Generated: {datetime.now().isoformat()}\n",
        "## Summary\n",
    ]

    # Stats
    total = count_total_transcripts()
    processed = count_processed_wiki()
    lines.append(f"- **Transcripts total**: {total}\n")
    lines.append(f"- **Wiki pages**: {processed}\n")
    if total > 0:
        lines.append(f"- **Progress**: {processed}/{total} (~{percentage}%)\n")

    # Lint report
    lint = run_full_lint()
    lines.append("## Lint Results\n")
    lines.append(f"- Orphans: {len(lint['orphans'])}")
    lines.append(f"- Dead links: {len(lint['dead_links'])}")
    lines.append(f"- Stale claims: {len(lint['stale_claims'])}\n")

    # Phase-specific checks
    lines.append("## Analysis\n")
    if percentage == 15:
        lines.append("**Action**: Validate extraction quality on first ~120 files.\n")
        lines.append("- Sample 20 extracted elements for approval rate.\n")
        lines.append("- If <70% useful: adjust extractor prompt, re-run batch.\n")
    elif percentage == 30:
        lines.append("**Action**: Cross-reference elements & merge duplicates.\n")
        lines.append("- Identify same concept across multiple pages.\n")
        lines.append("- Update [[wikilinks]] to point to canonical page.\n")
        lines.append(
            "- Validate embedding consistency (cosine similarity between duplicates).\n"
        )
    elif percentage == 50:
        lines.append("**Action**: Schema evolution check.\n")
        lines.append("- Identify new metadata fields in extracted elements.\n")
        lines.append("- Detect new topic clusters needing sub-indexes.\n")
        lines.append("- Re-run classifier on previously 'noise' elements.\n")
    elif percentage == 75:
        lines.append("**Action**: Graph density & orphan resolution.\n")
        lines.append("- Analyze wiki link graph density.\n")
        lines.append("- Propose new [[wikilinks]] to connect isolated clusters.\n")
        lines.append("- Validate retrieval against actual query patterns.\n")
    elif percentage == 100:
        lines.append("**Action**: Final audit & polish.\n")
        lines.append("- Full wiki lint (already above).\n")
        lines.append("- Re-embed all elements if embedding model improved.\n")
        lines.append("- Generate project timelines from session data.\n")
        lines.append("- Produce final quality report.\n")

    # Write
    out.write_text("".join(lines))
    print(f"Checkpoint report: {out}")
    return out


def main():
    import sys

    pct = int(sys.argv[1]) if len(sys.argv) > 1 else estimate_progress()
    print(f"Running checkpoint for {pct}%")
    run_checkpoint(pct)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
