#!/usr/bin/env python3
"""
Wiki auto-fix daemon — reads lint report, proposes safe fixes, requires approval.

Safe fixes: add missing [[wikilinks]] for orphan pages, update outdated years.
Complex fixes: create GitHub issues instead of auto-applying.
"""

import sys
from pathlib import Path
from datetime import datetime
from aaa_memory.wiki.linter import run_full_lint, write_report as lint_write
import subprocess

REPORT_PATH = Path("/home/misscheta/knowledge/wiki_lint_report.md")
FIX_LOG = Path("/home/misscheta/logs/wiki-fixes.log")


def auto_fix_orphans(orphan_list, auto_approve: bool = False):
    """Propose adding inbound links from related pages (simple keyword match)."""
    for orphan_path, _ in orphan_list:
        slug = orphan_path.stem
        # Find pages with similar tags or title words
        # Very simple: look for slug mention in any other page
        links_added = 0
        for other in orphan_path.parent.parent.glob("**/*.md"):
            if other == orphan_path or other.name == "index.md":
                continue
            content = other.read_text()
            if f"[[{slug}]]" in content:
                continue  # already linked
            # If other page contains any word from slug, suggest link
            if any(
                word in content.lower() for word in slug.split("-") if len(word) > 3
            ):
                suggestion = f"- In {other.relative_to(Path('/home/misscheta/knowledge'))}: add [[{slug}]] to connect to orphan"
                print(suggestion)
                if auto_approve:
                    # Insert minimal link at bottom
                    content += f"\nRelated: [[{slug}]]\n"
                    other.write_text(content)
                    links_added += 1
        if links_added:
            print(f"  Added {links_added} inbound links for orphan {slug}")


def auto_fix_dead_links(broken_list, auto_approve=False):
    """Dead links point to non-existent pages — either create stub or remove."""
    for path, broken_link in broken_list:
        target_slug = broken_link.split("/")[-1]
        if auto_approve:
            # Remove broken link, leave plain text
            text = path.read_text()
            text = text.replace(f"[[{broken_link}]]", broken_link)
            path.write_text(text)
            print(f"Unlinked: {broken_link} in {path.name}")
        else:
            print(f"Would unlink {broken_link} from {path.name} (target missing)")


def approve_fix(action_desc: str) -> bool:
    """Ask user to confirm fix."""
    resp = input(f"{action_desc} [Y/n] ").strip().lower()
    return resp in ("", "y", "yes")


def main(auto_approve: bool = False):
    print("[Wiki Auto-Fix] Running lint scan...")
    report = run_full_lint()
    changes = 0

    # 1. Orphans → add inbound links (safe-ish)
    if report["orphans"]:
        print(f"\nOrphans: {len(report['orphans'])} pages with no inbound links")
        auto_fix_orphans(report["orphans"], auto_approve=auto_approve)
        changes += len(report["orphans"])

    # 2. Dead links → unlink or create stubs
    if report["dead_links"]:
        print(f"\nDead links: {len(report['dead_links'])} broken [[wikilinks]]")
        auto_fix_dead_links(report["dead_links"], auto_approve=auto_approve)
        changes += len(report["dead_links"])

    # 3. Stale claims → open GitHub issue (manual)
    if report["stale_claims"]:
        print(f"\nStale claims: {len(report['stale_claims'])} pages need human review")
        print("Action: Create GitHub issue — 'Review stale claims in wiki'")

    # Re-run lint, show delta
    new_report = run_full_lint()
    new_lint = len(new_report["orphans"]) + len(new_report["dead_links"])
    old_lint = len(report["orphans"]) + len(report["dead_links"])
    print(f"\nLint score: {old_lint} → {new_lint} issues")
    print(f"Fixed: {old_lint - new_lint}")

    with open(FIX_LOG, "a") as f:
        f.write(f"{datetime.now().isoformat():{changes}} fixes applied\n")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--auto-approve", action="store_true", help="Apply fixes without confirmation"
    )
    args = ap.parse_args()
    main(auto_approve=args.auto_approve)
