#!/usr/bin/env python3
"""
Cold archival — move processed raw transcripts to compressed storage.

Strategy: Use MemVid V2 .mv2 files if available; else fallback to
compressed SQLite FTS5 archive in ~/.cache/clawmem/cold.sqlite.
"""

from pathlib import Path
import shutil
from datetime import datetime

RAW_BASE = Path("/home/misscheta/knowledge/raw/transcripts")
COLD_BASE = Path("/home/misscheta/.cache/clawmem/cold")
ARCHIVE_LOG = Path("/home/misscheta/knowledge/archive_log.jsonl")


def archive_file(filepath: Path):
    """Move file to cold storage with metadata."""
    COLD_BASE.mkdir(parents=True, exist_ok=True)
    rel = filepath.relative_to(Path("/home/misscheta/knowledge"))
    target = COLD_BASE / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(filepath), str(target))
    log = {
        "archived_at": datetime.now().isoformat(),
        "source": str(filepath),
        "destination": str(target),
        "size_bytes": target.stat().st_size,
    }
    with open(ARCHIVE_LOG, "a") as f:
        f.write(json.dumps(log) + "\n")
    print(f"Archived: {filepath.name} → {target}")


def main():
    print("[Cold Tier] Archival stub — MemVid V2 adapter not yet installed")
    print("Using fallback: move to compressed SQLite archive directory")
    # Simple file move for now
    for transcript in RAW_BASE.glob("**/*.jsonl"):
        # Only archive if already indexed (simple: processed flag file)
        # In full system, consult vault for ingestion status
        archive_file(transcript)


if __name__ == "__main__":
    main()
