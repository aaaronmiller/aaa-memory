"""
Wiki compiler — converts extracted Elements into markdown wiki pages.

Routes element types to directories:
  decision → wiki/decisions/
  code     → wiki/code/
  prompt   → wiki/prompts/
  concept  → wiki/concepts/
  pattern  → wiki/concepts/ (patterns are a subtype)
  fact     → compact into existing pages or decisions/

Each page gets YAML frontmatter and [[wikilink]] references.
"""

import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import re

from ..models import Element

WIKILINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")

WIKI_BASE = Path.home() / "knowledge/wiki"

# Type → directory mapping
TYPE_DIR = {
    "decision": "decisions",
    "code": "code",
    "prompt": "prompts",
    "concept": "concepts",
    "pattern": "concepts",  # patterns live under concepts/
    "fact": "decisions",  # facts embed into decisions
}


def compile_element(element: Element, output_base: Path = WIKI_BASE) -> Path:
    """
    Compile a single element into its wiki page.

    Returns path to written file.
    """
    subdir = TYPE_DIR.get(element.type, "uncategorized")
    out_dir = output_base / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filename from title slug
    safe = re.sub(r"[^a-z0-9-]", "-", element.title.lower())[:60]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"{ts}-{safe}.md"
    path = out_dir / filename

    # Frontmatter
    frontmatter = {
        "title": element.title,
        "type": element.type,
        "tags": element.tags,
        "confidence": round(element.confidence, 3),
        "source_file": element.source_file,
        "extracted_at": element.extracted_at,
        "project": element.project or "default",
        "agent": element.agent or "unknown",
    }

    # Body
    body = f"# {element.title}\n\n"
    body += element.content

    # Footnotes: source provenance
    body += f"\n\n---\n*Source: `{element.source_file}` | Agent: {element.agent}*"

    # Write
    with open(path, "w") as f:
        f.write("---\n")
        yaml.dump(frontmatter, f, default_flow_style=False, sort_keys=False)
        f.write("---\n")
        f.write(body)

    return path


def compile_batch(elements: List[Element], output_base: Path = WIKI_BASE) -> List[Path]:
    """Compile multiple elements."""
    return [compile_element(el, output_base) for el in elements]


# ── Wiki index builder (T033 — Karpathy pointer-based index) ─────────────────


class WikiIndexer:
    """Builds pointer-based index pages."""

    def __init__(self, wiki_base: Path = WIKI_BASE):
        self.wiki_base = wiki_base

    def generate_master_index(self) -> Path:
        """Create wiki/_master-index.md listing all sub-indexes."""
        master = self.wiki_base / "_master-index.md"
        sections = []
        for subdir in sorted(self.wiki_base.iterdir()):
            if subdir.is_dir() and not subdir.name.startswith("_"):
                sections.append(f"- [[{subdir.name}/index]]")

        content = "# Master Index\n\n" + "\n".join(sections) + "\n"
        master.write_text(content)
        return master

    def generate_sub_indexes(self) -> List[Path]:
        """Generate index.md for each category that lists all pages."""
        indexes = []
        for subdir in sorted(self.wiki_base.iterdir()):
            if subdir.is_dir() and not subdir.name.startswith("_"):
                index_path = subdir / "index.md"
                pages = []
                for md in sorted(subdir.glob("*.md")):
                    if md.name == "index.md":
                        continue
                    title = md.read_text().split("\n")[0].lstrip("# ").strip()
                    pages.append(f"[[{md.stem}]] — {title}")
                content = f"# {subdir.name.title()}\n\n" + "\n".join(pages) + "\n"
                index_path.write_text(content)
                indexes.append(index_path)
        return indexes


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from ..extractor.llm_extractor import Element

    # Demo: compile elements from stdin JSON
    if len(sys.argv) > 1 and sys.argv[1] == "index":
        idx = WikiIndexer()
        print("Generating master index...")
        idx.generate_master_index()
        print("Generating sub-indexes...")
        idx.generate_sub_indexes()
        print("Done")
    else:
        print("Usage: python -m aaa_memory.wiki.compiler index")
