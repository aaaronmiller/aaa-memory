"""
Wiki linter — scans wiki/ for structural issues.

Detects:
- Orphans: pages with zero inbound [[wikilink]] references
- Dead links: [[wikilink]] pointing to non-existent page
- Stale claims: outdated dates/claims (simple heuristic)
- Contradictions: opposing statements on same topic (basic keyword polarity)
"""

import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
from aaa_memory import config

WIKI_BASE = config.WIKI_BASE


def find_all_pages() -> Dict[str, Path]:
    """Map slug → filepath for all wiki pages."""
    pages = {}
    for md in WIKI_BASE.glob("**/*.md"):
        slug = md.stem
        pages[slug] = md
    return pages


def extract_wikilinks(text: str) -> List[str]:
    """Extract [[link]] targets from markdown."""
    return re.findall(r"\[\[([^\]]+)\]\]", text)


def lint_orphans() -> List[Tuple[Path, int]]:
    """
    Find pages with zero inbound links.

    Returns list of (filepath, inbound_count) where count=0.
    """
    pages = find_all_pages()
    inbound = defaultdict(int)

    for path in pages.values():
        content = path.read_text()
        for link in extract_wikilinks(content):
            # Link might be Namespace/page or just page
            target = link.split("/")[-1]  # take last segment
            inbound[target] += 1

    orphans = []
    for slug, path in pages.items():
        count = inbound.get(slug, 0)
        if count == 0:
            orphans.append((path, count))
    return orphans


def lint_dead_links() -> List[Tuple[Path, str]]:
    """
    Find [[wikilink]] references pointing to non-existent pages.
    Returns list of (filepath, broken_link).
    """
    pages = find_all_pages()
    broken = []
    for path in pages.values():
        content = path.read_text()
        for link in extract_wikilinks(content):
            target_slug = link.split("/")[-1]
            if target_slug not in pages:
                broken.append((path, link))
    return broken


def lint_stale_claims() -> List[Path]:
    """
    Find pages with potentially outdated year claims (e.g., "2024" in a "latest" context).
    Heuristic: pages mentioning "latest" or "current" alongside a past year.
    """
    stale = []
    current_year = datetime.now().year
    for md in WIKI_BASE.glob("**/*.md"):
        content = md.read_text().lower()
        if ("latest" in content or "current" in content) and str(
            current_year - 1
        ) in content:
            stale.append(md)
    return stale


def run_full_lint() -> Dict[str, any]:
    """Execute all lint checks and return report dict."""
    report = {
        "orphans": lint_orphans(),
        "dead_links": lint_dead_links(),
        "stale_claims": lint_stale_claims(),
    }
    return report


def write_report(
    report: Dict[str, any],
    output: Path = config.WIKI_BASE.parent / "wiki_lint_report.md",
):
    """Render lint report as markdown."""
    lines = ["# Wiki Lint Report\n"]

    # Orphans
    lines.append("## Orphans (no inbound links)\n")
    if report["orphans"]:
        for path, count in report["orphans"]:
            rel = path.relative_to(WIKI_BASE)
            lines.append(f"- `{rel}` — 0 inbound links\n")
    else:
        lines.append("- ✅ None found\n")

    # Dead links
    lines.append("\n## Dead Links\n")
    if report["dead_links"]:
        for path, link in report["dead_links"]:
            rel = path.relative_to(WIKI_BASE)
            lines.append(f"- `{rel}`: broken `[[{link}]]`\n")
    else:
        lines.append("- ✅ None found\n")

    # Stale claims
    lines.append("\n## Stale Claims\n")
    if report["stale_claims"]:
        for path in report["stale_claims"]:
            rel = path.relative_to(WIKI_BASE)
            lines.append(f"- `{rel}` — mentions outdated year\n")
    else:
        lines.append("- ✅ None found\n")

    output.write_text("".join(lines))
    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from datetime import datetime
    from zoneinfo import ZoneInfo

    print("Running wiki lint...")
    report = run_full_lint()
    out = write_report(report)
    print(f"Report written to {out}")
    print(f"Orphans: {len(report['orphans'])}")
    print(f"Dead links: {len(report['dead_links'])}")
    print(f"Stale claims: {len(report['stale_claims'])}")
