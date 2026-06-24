"""Qwen Code context integration.

Qwen Code is a TypeScript CLI tool that supports extensions and hooks.
Integration calls the `clawmem` CLI for memory operations.

For Qwen extension integration, the extension would:
  1. Use `qwen hooks` to register pre/post-turn callbacks
  2. Call `clawmem hook context-surfacing` for pre-turn context
  3. Call `clawmem diary write` for post-turn recording
  4. Call `clawmem search <query>` for explicit memory search

Usage from Qwen hook script:
```bash
clawmem hook context-surfacing
clawmem diary write "$TURN_SUMMARY" -t agent-turn -a qwen
```
"""

import subprocess
import json
from pathlib import Path
from typing import Optional


CLAWMEM_BIN = str(Path.home() / ".npm-global" / "bin" / "clawmem")


def refresh_context(session_id: Optional[str] = None) -> dict:
    """Refresh session context with relevant memories.

    Calls the clawmem context-surfacing hook and returns
    any surfaced context as a dict.
    """
    try:
        hook_input = json.dumps({"session_id": session_id or ""})
        result = subprocess.run(
            [CLAWMEM_BIN, "hook", "context-surfacing"],
            input=hook_input, capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
        return {"status": "ok", "context": None}
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
        return {"status": "error", "error": str(e)}
