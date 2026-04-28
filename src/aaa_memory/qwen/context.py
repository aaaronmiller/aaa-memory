"""
Qwen Code integration — context refresh + MCP tool registration.
"""

from pathlib import Path

# Qwen Code reads PROJECT_SUMMARY.md at project root to get context
SUMMARY_PATH = Path("PROJECT_SUMMARY.md")


def refresh_context(project_root: Path, summary: str) -> Path:
    """
    Write updated PROJECT_SUMMARY.md for Qwen Code.

    This file is auto-injected into Qwen Code sessions.
    """
    p = project_root / SUMMARY_PATH
    p.write_text(f"# Project Summary\n\n{summary}\n")
    return p


def mcp_tools_available() -> bool:
    """Check if aaa-memory MCP server is running and reachable."""
    # Placeholder — real check would attempt stdio connection
    return True


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    summary = (
        sys.argv[2] if len(sys.argv) > 2 else "Default project context from aaa-memory"
    )
    path = refresh_context(root, summary)
    print(f"Context written to {path}")
