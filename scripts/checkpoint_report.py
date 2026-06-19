#!/usr/bin/env python3
"""
Refinement checkpoint report — invoked at vault-migration milestones (15/30/50/75/100%).

Analyzes extraction quality, schema drift, graph density, and retrieval relevance.
Produces markdown report to ~/knowledge/checkpoint_<pct>%.md
"""

import json
from pathlib import Path
from datetime import datetime
from aaa_memory.wiki.linter import run_full_lint

CHECKPOINT_DIR = Path("/home/misscheta/knowledge/checkpoints")


def run_checkpoint(percentage: int):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    report = CHECKPOINT_DIR / f"checkpoint_{percentage:03d}.md"

    lines = [
        f"# Vault Migration Checkpoint — {percentage}% Complete",
        f"Generated: {datetime.now().isoformat()}\n",
        "## Summary\n",
    ]

    # Count wiki pages
    wiki_base = Path("/home/misscheta/knowledge/wiki")
    page_count = sum(1 for _ in wiki_base.rglob("*.md"))
    lines.append(f"- **Wiki pages**: {page_count}\n")

    # Lint report
    lint = run_full_lint()
    lines.append("## Lint Results\n")
    lines.append(f"- Orphans: {len(lint['orphans'])}")
    lines.append(f"- Dead links: {len(lint['dead_links'])}")
    lines.append(f"- Stale claims: {len(lint['stale_claims'])}\n")

    # TODO: embedding consistency check, retrieval QA sample

    report.write_text("\n".join(lines))
    print(f"Checkpoint report written to {report}")


if __name__ == "__main__":
    import sys

    pct = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    run_checkpoint(pct)
