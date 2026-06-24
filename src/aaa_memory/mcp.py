"""
MCP server for aaa-memory.

Exposes memory operations as MCP tools for AI agents.

Tools:
  - memory_search(query, limit) - hybrid retrieval across tiers
  - memory_sessions(project_id) - list sessions for a project
  - memory_timeline(project_id, days) - generate timeline markdown
  - memory_store(agent, turn_data, session_id) - store content in the vault
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from aaa_memory import config
from aaa_memory.audit.timeline import assemble_timeline
from aaa_memory.hot.mem_store import VaultMemoryStore
from aaa_memory.retrieval.pipeline import search as memory_search


def handle_search(query: str, limit: int = 20) -> List[Dict]:
    """Search across all memory tiers."""
    return memory_search(query, limit=limit)


def handle_sessions(project_id: Optional[str] = None) -> List[Dict]:
    """List sessions, optionally filtered by project. Uses the current vault schema."""
    if not config.VAULT.exists():
        return []

    conn = sqlite3.connect(str(config.VAULT))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        if project_id:
            cur.execute(
                "SELECT session_id, agent, MIN(created_at) as first_seen, "
                "MAX(created_at) as last_seen, COUNT(*) as turns "
                "FROM turns WHERE metadata LIKE ? "
                "GROUP BY session_id, agent ORDER BY last_seen DESC LIMIT 50",
                (f'%{project_id}%',),
            )
        else:
            cur.execute(
                "SELECT session_id, agent, MIN(created_at) as first_seen, "
                "MAX(created_at) as last_seen, COUNT(*) as turns "
                "FROM turns GROUP BY session_id, agent "
                "ORDER BY last_seen DESC LIMIT 50"
            )
        results = [dict(row) for row in cur.fetchall()]
    except sqlite3.OperationalError:
        results = []
    finally:
        conn.close()
    return results


def handle_timeline(project_id: str, days: int = 7) -> str:
    """Generate a markdown timeline for a project."""
    return assemble_timeline(project_id, days)


def _ensure_turns_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turn_id TEXT UNIQUE NOT NULL,
            agent TEXT NOT NULL,
            session_id TEXT,
            turn_index INTEGER,
            turn_type TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_agent ON turns(agent)")


def handle_store(agent: str, turn_data: str, session_id: Optional[str] = None) -> Dict:
    """Store content in both the turns table and hot memories."""
    config.VAULT.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    turn_id = str(uuid.uuid4())
    sid = session_id or turn_id
    agent_name = agent or "unknown"

    conn = sqlite3.connect(str(config.VAULT))
    try:
        _ensure_turns_table(conn)
        conn.execute(
            """
            INSERT OR IGNORE INTO turns
                (turn_id, session_id, agent, turn_index, turn_type, raw_text, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                turn_id,
                sid,
                agent_name,
                None,
                "memory_store",
                turn_data or "",
                now,
                json.dumps({"source": "mcp", "project": "default"}),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    memory = None
    if (turn_data or "").strip():
        memory = VaultMemoryStore().add(
            turn_data,
            tags=["mcp", "memory_store"],
            project="default",
            source=agent_name,
        )

    return {
        "turn_id": turn_id,
        "session_id": sid,
        "memory_id": memory.get("id") if memory else None,
        "status": "stored",
    }


def run_server(transport: str = "stdio", port: int = 7437, host: str = "127.0.0.1") -> None:
    """Run the MCP server.

    Two modes:
      - stdio (default): per-session stdio transport. DEPRECATED for production -
        use SSE daemon mode instead to avoid per-session process spawning.
      - sse: shared singleton daemon on host:port. All sessions connect to the same
        process via URL-based MCP config.

    Args:
        transport: "stdio" or "sse"
        port: HTTP port for SSE mode
        host: bind address for SSE mode
    """
    from mcp.server.fastmcp import FastMCP

    app = FastMCP("aaa-memory")

    @app.tool(name="memory_search")
    def memory_search_tool(query: str, limit: int = 20) -> str:
        """Search across all memory tiers using hybrid retrieval."""
        return json.dumps(handle_search(query, limit), indent=2)

    @app.tool(name="memory_sessions")
    def memory_sessions_tool(project_id: Optional[str] = None) -> str:
        """List sessions, optionally filtered by project."""
        return json.dumps(handle_sessions(project_id), indent=2)

    @app.tool(name="memory_timeline")
    def memory_timeline_tool(project_id: str, days: int = 7) -> str:
        """Generate a project timeline as markdown."""
        return handle_timeline(project_id, days)

    @app.tool(name="memory_store")
    def memory_store_tool(agent: str, turn_data: str, session_id: Optional[str] = None) -> str:
        """Store a turn or durable memory in the aaa-memory vault."""
        return json.dumps(handle_store(agent, turn_data, session_id), indent=2)

    if transport == "sse":
        # SSE daemon mode: shared singleton, all sessions connect via URL
        import uvicorn
        sse_app = app.sse_app()
        print(f"aaa-memory MCP SSE daemon listening on http://{host}:{port}/sse", flush=True)
        uvicorn.run(sse_app, host=host, port=port, log_level="warning")
    else:
        # Stdio mode: per-session (backward compat)
        app.run("stdio")


def main() -> None:
    """Entry point.

    Usage:
        python -m aaa_memory.mcp              # stdio (per-session, deprecated)
        python -m aaa_memory.mcp serve         # SSE daemon (shared singleton)
        python -m aaa_memory.mcp serve --port 7437 --host 127.0.0.1
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run aaa-memory MCP server")
    subparsers = parser.add_subparsers(dest="command")
    serve = subparsers.add_parser("serve", help="run a shared SSE daemon")
    serve.add_argument("--port", type=int, default=7437)
    serve.add_argument("--host", default="127.0.0.1")

    args = parser.parse_args()
    if args.command == "serve":
        run_server(transport="sse", port=args.port, host=args.host)
    else:
        run_server(transport="stdio")


if __name__ == "__main__":
    main()
