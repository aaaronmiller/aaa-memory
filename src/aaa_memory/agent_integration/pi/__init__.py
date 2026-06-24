"""Pi agent integration stub.

Pi is a Go-based CLI tool for model routing. It has no native plugin system.

Integration is done via Claude Code hooks (which Pi sessions share when
launched through the `xx` launcher) and the `clawmem` CLI.

Usage:
```bash
# Record context
clawmem diary write "$TURN_SUMMARY" -t agent-turn -a pi

# Search memory
clawmem search "what did we discuss"

# Surface context at session start
clawmem hook context-surfacing
```
"""

import subprocess
import json
from pathlib import Path
from typing import Optional


CLAWMEM_BIN = str(Path.home() / ".npm-global" / "bin" / "clawmem")


def search(query: str, limit: int = 10) -> list:
    """Search memory via clawmem CLI."""
    try:
        result = subprocess.run(
            [CLAWMEM_BIN, "search", query, "-n", str(limit), "--json"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
        return []
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        return []


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
            [CLAWMEM_BIN, "diary", "write", entry, "-t", "agent-turn", "-a", "pi"],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
