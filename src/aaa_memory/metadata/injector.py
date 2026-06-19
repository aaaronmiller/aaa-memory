"""
Metadata injector — adds YAML frontmatter and provenance to extracted elements.

Writes structured markdown files with frontmatter containing:
- title, type, tags, confidence, source_file, extraction_ts, project, agent, session_id
- [[wikilinks]] auto-generated from content analysis
"""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import yaml
import re
from typing import Optional

from ..extractor.llm_extractor import Element

WIKILINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")


def inject_metadata(
    element: Element,
    source_file: str,
    project: str = "default",
    agent: str = "unknown",
    session_id: Optional[str] = None,
    output_dir: Path = Path.home() / "knowledge/wiki",
) -> Path:
    """
    Write element as markdown file with YAML frontmatter.

    Parameters
    ----------
    element : Element
        Extracted knowledge element
    source_file : str
        Origin file path
    project : str
        Project identifier
    agent : str
        Agent name (claude-code, openclaw, etc.)
    session_id : str | None
        Session identifier if known
    output_dir : Path
        Base output directory — subdirectories by element type used

    Returns
    -------
    Path
        Path to written markdown file
    """
    # Determine output subdirectory by type
    type_dir = output_dir / f"{element.type}s"  # decision → decisions, code → code
    type_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    safe_title = re.sub(r"[^a-z0-9-]", "-", element.title.lower())[:60]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"{ts}-{safe_title}.md"
    filepath = type_dir / filename

    # Extract wikilinks from content → add to tags
    wikilinks = WIKILINK_PATTERN.findall(element.content)
    all_tags = list(set(element.tags + [f"[[{link}]]" for link in wikilinks]))

    # Build frontmatter dict
    frontmatter = {
        "title": element.title,
        "type": element.type,
        "tags": all_tags,
        "confidence": round(element.confidence, 3),
        "source_file": source_file,
        "extraction_ts": datetime.now(timezone.utc).isoformat(),
        "project": project,
        "agent": agent,
        "session_id": session_id or "unknown",
    }
    if element.quote:
        frontmatter["quote"] = element.quote

    # Render markdown
    body = ""

    # Add [[wikilinks]] section if any
    if wikilinks:
        body += f"**Related**: {', '.join(f'[[{w}]]' for w in wikilinks)}\n\n"

    # Main content
    body += element.content

    # Provenance footer
    body += f"\n\n---\n*Extracted from `{source_file}` by {agent} on {frontmatter['extraction_ts']}*"

    # Write file with frontmatter
    with open(filepath, "w") as f:
        f.write("---\n")
        yaml.dump(frontmatter, f, default_flow_style=False, sort_keys=False)
        f.write("---\n")
        f.write(body)

    return filepath


def inject_batch(
    elements: list[Element],
    source_file: str,
    project: str = "default",
    agent: str = "unknown",
    output_dir: Path = Path.home() / "knowledge/wiki",
) -> list[Path]:
    """Inject multiple elements at once."""
    return [
        inject_metadata(el, source_file, project, agent, output_dir=output_dir)
        for el in elements
    ]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json
    from pathlib import Path

    # Quick test: read a JSON element, write markdown
    if len(sys.argv) < 3:
        print(
            "Usage: python -m aaa_memory.metadata.injector <element.json> <output_dir>"
        )
        sys.exit(1)

    elem_json = json.loads(Path(sys.argv[1]).read_text())
    element = Element(**elem_json)
    out = Path(sys.argv[2])

    path = inject_metadata(
        element, source_file="test", project="demo", agent="test", output_dir=out
    )
    print(f"Wrote: {path}")
