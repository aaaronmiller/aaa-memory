"""
Session audit discovery scanner — walks known agent storage locations.

Finds session files for: Claude Code, OpenClaw, Hermes, Qwen, OpenCode, Codex, Web captures.
"""

import os
from pathlib import Path
from typing import List, Dict
import subprocess

# Known session roots per agent
AGENT_PATHS = {
    "claude-code": Path(os.getenv("HOME")) / ".claude/sessions",
    "openclaw": Path(os.getenv("HOME")) / ".openclaw/sessions",
    "hermes": Path(os.getenv("HOME")) / ".hermes/state.db",  # single SQLite
    "qwen": Path(os.getenv("HOME")) / ".qwen/context",
    "opencode": Path(os.getenv("HOME")) / ".opencode/sessions",
    "codex": Path(os.getenv("HOME")) / ".codex/rollouts",
    "web": Path(os.getenv("HOME")) / "knowledge/raw/web",
}


def discover_sessions() -> Dict[str, List[Path]]:
    """
    Scan all agent storage roots for session files.

    Returns dict mapping agent → list of file paths.
    """
    found: Dict[str, List[Path]] = {}

    for agent, root in AGENT_PATHS.items():
        files = []
        if root.is_file():
            files = [root]  # single-file DB
        elif root.is_dir():
            for pattern in ["**/*.jsonl", "**/*.json", "**/*.db", "**/*.txt"]:
                try:
                    files.extend(root.glob(pattern))
                except Exception:
                    pass
        found[agent] = files
        print(f"[audit] {agent}: {len(files)} session files found")

    return found


# Quick test
if __name__ == "__main__":
    result = discover_sessions()
    total = sum(len(v) for v in result.values())
    print(f"\nTotal session files discovered: {total}")
