"""Wiki auto-fix agent — safe auto-fixes for lint issues."""
import re
from pathlib import Path
from typing import List, Dict

WIKI_DIR = Path.home() / "knowledge/wiki"

def fix_orphans(lint_report: str) -> List[Dict]:
    """Add orphaned pages to the master index."""
    fixes = []
    for line in lint_report.split("\n"):
        if "orphan" in line.lower() and ":" in line:
            page = line.split(":")[-1].strip()
            index_path = WIKI_DIR / "index.md"
            if index_path.exists():
                with open(index_path) as f:
                    content = f.read()
                if page not in content:
                    with open(index_path, "a") as f:
                        f.write(f"\n- [[{page}]]")
                    fixes.append({"page": page, "fix": "added to index.md"})
    return fixes

def fix_dead_links(lint_report: str) -> List[Dict]:
    """Remove broken wikilinks pointing to missing pages."""
    fixes = []
    for line in lint_report.split("\n"):
        if "dead link" in line.lower() and ":" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                fixes.append({"link": parts[-1].strip(), "note": "manual review needed"})
    return fixes
