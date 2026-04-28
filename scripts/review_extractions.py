#!/usr/bin/env python3
"""
Interactive extraction review — human-in-the-loop approval UI.

Walks newly extracted elements from recent runs, displays them, and
accepts Approve/Reject/Edit decisions. Stores results in extraction_reviews.jsonl.
"""

import json
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.panel import Panel

console = Console()
REVIEW_LOG = Path("/home/misscheta/knowledge/extraction_reviews.jsonl")
WIKI_BASE = Path("/home/misscheta/knowledge/wiki")


def find_recent_elements(limit: int = 50) -> list[Path]:
    """Find recently modified markdown files in wiki/ (likely new extractions)."""
    candidates = list(WIKI_BASE.rglob("*.md"))
    # Sort by mtime
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[:limit]


def parse_element_from_md(path: Path) -> dict:
    """Parse YAML frontmatter + body from wiki page."""
    import yaml

    text = path.read_text()
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    frontmatter = yaml.safe_load(parts[1])
    body = parts[2].strip()
    return {**frontmatter, "body": body, "file": str(path)}


def review_element(element_data: dict) -> dict:
    """Interactive review session for one element."""
    console.clear()
    console.print(
        Panel.fit(
            f"[bold]{element_data.get('title', '?')}[/bold]\n"
            f"Type: {element_data.get('type')} | "
            f"Confidence: {element_data.get('confidence')}"
        )
    )

    # Body (code or text)
    body = element_data.get("body", "")
    if element_data.get("type") == "code":
        syntax = Syntax(body, "python", line_numbers=True, theme="monokai")
        console.print(syntax)
    else:
        console.print(body)

    # Actions
    console.print("\n[bold]Actions:[/bold]")
    console.print("  [green]A[/green] — Approve  (keep as-is)")
    console.print("  [red]R[/red] — Reject   (delete file)")
    console.print("  [yellow]E[/yellow] — Edit     (open in $EDITOR)")
    console.print("  [blue]S[/blue] — Skip     (no decision)")

    choice = Prompt.ask("Choice", choices=["a", "r", "e", "s"], default="a")
    decision = {"approved": False, "edited": False, "deleted": False}

    if choice == "a":
        decision["approved"] = True
    elif choice == "r":
        decision["deleted"] = True
        if Confirm.ask("Delete file?"):
            Path(element_data["file"]).unlink(missing_ok=True)
    elif choice == "e":
        # Open in EDITOR
        editor = os.getenv("EDITOR", "vim")
        os.system(f"{editor} {element_data['file']}")
        decision["edited"] = True
        decision["approved"] = Confirm.ask("Approve after edit?")
    # 's' → skip, no decision

    return {
        "file": element_data["file"],
        "title": element_data.get("title"),
        "type": element_data.get("type"),
        "decision": decision,
        "reviewed_at": datetime.now().isoformat(),
    }


def main():
    console.print("[bold]Clawdi Extraction Review[/bold]\n")
    elements = find_recent_elements(limit=50)
    console.print(f"Found {len(elements)} recent elements to review.\n")

    if not Confirm.ask("Begin review session?"):
        return

    reviews = []
    for path in elements:
        try:
            el = parse_element_from_md(path)
            if not el:
                continue
            result = review_element(el)
            reviews.append(result)

            # Log immediately
            with open(REVIEW_LOG, "a") as f:
                f.write(json.dumps(result) + "\n")

            if result.get("deleted"):
                console.print(f"  → Deleted {path.name}")
            elif result.get("approved"):
                console.print(f"  → Approved {path.name}")
            elif result.get("edited"):
                console.print(f"  → Edited {path.name}")
        except KeyboardInterrupt:
            console.print("\n[red]Interrupted — saving progress[/red]")
            break
        except Exception as e:
            console.print(f"[red]Error reviewing {path}: {e}[/red]")

    summary = {
        "session_end": datetime.now().isoformat(),
        "total_reviewed": len(reviews),
        "approved": sum(1 for r in reviews if r["decision"].get("approved")),
        "rejected": sum(1 for r in reviews if r["decision"].get("deleted")),
        "edited": sum(1 for r in reviews if r["decision"].get("edited")),
    }
    console.print("\n[bold]Session Summary:[/bold]")
    console.print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import os

    main()
