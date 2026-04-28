#!/usr/bin/env python3
"""
Graphiti ingestion — episodes from PRD/wiki files.

Reads `~/knowledge/wiki/decisions/` and `~/knowledge/wiki/concepts/`,
creates GraphEpisode entities and writes to warm tier.
Currently a placeholder — Graphiti integration TBD.
"""

from pathlib import Path
import json

WIKI_BASE = Path("/home/misscheta/knowledge/wiki")


def main():
    print("[Graphiti] Ingest stub — Graphiti Python SDK not yet installed")
    print("Would read PRD files and create temporal knowledge graph episodes")
    # Placeholder: collect all concept/decision pages, extract entities
    files = list((WIKI_BASE / "decisions").glob("*.md")) + list(
        (WIKI_BASE / "concepts").glob("*.md")
    )
    print(f"Found {len(files)} candidate pages for Graphiti episodes")
    print(
        "Action: Create Graphiti Python client; define Entity/Relation schemas; batch upsert"
    )


if __name__ == "__main__":
    main()
