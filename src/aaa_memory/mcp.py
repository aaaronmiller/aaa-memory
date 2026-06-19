"""
MCP server for aaa-memory.

Exposes memory operations as MCP tools for AI agents (Claude Code, etc.).

Tools:
  - memory_search(query, limit) — hybrid retrieval across tiers
  - memory_sessions(project_id) — list sessions for a project
  - memory_timeline(project_id, days) — generate timeline markdown
  - memory_store(agent, turn_data) — store a turn into the hot vault
"""

import json
import os
import sys
from typing import List, Dict, Optional

from aaa_memory.retrieval.pipeline import search as memory_search
from aaa_memory.audit.timeline import assemble_timeline
from aaa_memory.retrieval.hot import VAULT as config.VAULT


def handle_search(query: str, limit: int = 20) -> List[Dict]:
    """Search across all memory tiers."""
    results = memory_search(query, limit=limit)
    return results


def handle_sessions(project_id: Optional[str] = None) -> List[Dict]:
    """List sessions, optionally filtered by project. Uses actual vault schema."""
    if not config.VAULT.exists():
        return []
    import sqlite3
    conn = sqlite3.connect(str(config.VAULT))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        if project_id:
            cur.execute(
                "SELECT session_id, agent, MIN(created_at) as first_seen, MAX(created_at) as last_seen, "
                "COUNT(*) as turns FROM turns WHERE metadata LIKE ? "
                "GROUP BY session_id ORDER BY last_seen DESC LIMIT 50",
                (f'%{project_id}%',),
            )
        else:
            cur.execute(
                "SELECT session_id, agent, MIN(created_at) as first_seen, MAX(created_at) as last_seen, "
                "COUNT(*) as turns FROM turns "
                "GROUP BY session_id ORDER BY last_seen DESC LIMIT 50",
            )
        results = [dict(row) for row in cur.fetchall()]
    except sqlite3.OperationalError:
        results = []
    conn.close()
    return results


def handle_timeline(project_id: str, days: int = 7) -> str:
    """Generate a markdown timeline for a project."""
    return assemble_timeline(project_id, days)


def handle_store(agent: str, turn_data: str, session_id: Optional[str] = None) -> Dict:
    """Store a turn into the hot vault."""
    import sqlite3
    import uuid
    from datetime import datetime

    config.VAULT.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.VAULT))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            turn_id TEXT PRIMARY KEY,
            session_id TEXT,
            agent TEXT,
            project TEXT,
            timestamp TEXT,
            raw_text TEXT
        )
    """)

    tid = str(uuid.uuid4())
    sid = session_id or tid
    ts = datetime.utcnow().isoformat()

    conn.execute(
        "INSERT OR IGNORE INTO turns (turn_id, session_id, agent, project, timestamp, raw_text) VALUES (?, ?, ?, ?, ?, ?)",
        (tid, sid, agent, "default", ts, turn_data),
    )
    conn.commit()
    conn.close()

    return {"turn_id": tid, "session_id": sid, "status": "stored"}


def run_server():
    """Run the MCP server on stdio (standard MCP protocol)."""
    import asyncio
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    app = Server("aaa-memory")

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> List[Dict]:
        if name == "memory_search":
            results = handle_search(arguments.get("query", ""), arguments.get("limit", 20))
            return [{"type": "text", "text": json.dumps(results, indent=2)}]
        elif name == "memory_sessions":
            results = handle_sessions(arguments.get("project_id"))
            return [{"type": "text", "text": json.dumps(results, indent=2)}]
        elif name == "memory_timeline":
            text = handle_timeline(arguments.get("project_id", ""), arguments.get("days", 7))
            return [{"type": "text", "text": text}]
        elif name == "memory_store":
            result = handle_store(arguments.get("agent", ""), arguments.get("turn_data", ""), arguments.get("session_id"))
            return [{"type": "text", "text": json.dumps(result)}]
        else:
            return [{"type": "text", "text": json.dumps({"error": f"unknown tool: {name}"})}]

    @app.list_tools()
    async def list_tools():
        return [
            {
                "name": "memory_search",
                "description": "Search across all memory tiers (hot/warm/cold) using hybrid retrieval",
                "inputSchema": {"type": "object", "properties": {
                    "query": {"type": "string", "description": "Natural language query"},
                    "limit": {"type": "number", "description": "Max results (default 20)"},
                }},
            },
            {
                "name": "memory_sessions",
                "description": "List sessions, optionally filtered by project",
                "inputSchema": {"type": "object", "properties": {
                    "project_id": {"type": "string", "description": "Filter by project (optional)"},
                }},
            },
            {
                "name": "memory_timeline",
                "description": "Generate project timeline as markdown",
                "inputSchema": {"type": "object", "properties": {
                    "project_id": {"type": "string"},
                    "days": {"type": "number", "description": "Lookback window (default 7)"},
                }},
            },
            {
                "name": "memory_store",
                "description": "Store a turn into the hot vault",
                "inputSchema": {"type": "object", "properties": {
                    "agent": {"type": "string"},
                    "turn_data": {"type": "string", "description": "The turn content"},
                    "session_id": {"type": "string", "description": "Optional session grouping ID"},
                }},
            },
        ]

    async def _run():
        async with stdio_server() as streams:
            await app.run(streams)

    asyncio.run(_run())


if __name__ == "__main__":
    run_server()
