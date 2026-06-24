"""Codex CLI agent integration.

Codex is a Rust CLI tool with a plugin system (`codex plugin`).
Integration uses the `clawmem` CLI for memory access.

For Codex plugin integration, the plugin would:
  1. Register pre/post turn hooks via Codex's plugin API
  2. Call `clawmem hook context-surfacing` for pre-turn context
  3. Call `clawmem diary write` for post-turn recording
  4. Call `clawmem search <query>` for explicit memory search

Usage from Codex plugin (shell script or plugin):
```bash
clawmem hook context-surfacing
clawmem diary write "$TURN_SUMMARY" -t agent-turn -a codex
clawmem search "$QUERY"
```
"""

import subprocess
from pathlib import Path
from typing import Optional


CLAWMEM_BIN = str(Path.home() / ".npm-global" / "bin" / "clawmem")


def parse_codex_rollouts(path: Optional[str] = None) -> list:
    """Parse Codex session rollouts from the filesystem and import them.

    Scans the Codex sessions directory for rollouts and parses them
    into a format compatible with aaa-memory's import pipeline.
    """
    import sqlite3
    from datetime import datetime

    codex_home = Path(path or Path.home() / ".codex")
    sessions_db = codex_home / "sessions.db"

    if not sessions_db.exists():
        return []

    results = []
    try:
        conn = sqlite3.connect(str(sessions_db))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # Codex stores sessions in a SQLite DB; adapt to our schema
        tables = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [r["name"] for r in tables]

        if "sessions" in table_names:
            cur.execute(
                "SELECT id, created_at, updated_at, summary FROM sessions "
                "ORDER BY updated_at DESC LIMIT 20"
            )
            for row in cur.fetchall():
                results.append({
                    "session_id": row["id"],
                    "agent": "codex",
                    "started_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "summary": row["summary"] or "",
                })
        conn.close()
    except (sqlite3.Error, FileNotFoundError):
        pass

    return results


def store_turn(prompt: str, response: str, session_id: Optional[str] = None) -> bool:
    """Store a turn via clawmem CLI."""
    try:
        entry = "\n\n".join(
            part for part in [
                f"session_id: {session_id}" if session_id else "",
                f"prompt:\n{prompt or ''}",
                f"response:\n{response or ''}",
            ] if part
        )
        result = subprocess.run(
            [CLAWMEM_BIN, "diary", "write", entry, "-t", "agent-turn", "-a", "codex"],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
