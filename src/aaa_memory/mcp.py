#!/usr/bin/env python3
"""
MCP server for aaa-memory.

Tools:
  - memory_search(query, limit) — hybrid retrieval across tiers
  - memory_sessions(project_id) — list sessions for project
  - memory_timeline(project_id, days) — generate timeline markdown
"""

import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from aaa_memory.retrieval.pipeline import search as memory_search
from aaa_memory.audit.timeline import assemble_timeline
import asyncio

app = Server("aaa-memory")

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "memory_search":
        query = arguments.get("query", "")
        limit = arguments.get("limit", 10)
        results = memory_search(query, limit=limit)
        return {"results": results, "total": len(results)}
    elif name == "memory_sessions":
        project_id = arguments.get("project_id", "")
        timeline = assemble_timeline(project_id, days=30)
        return {"sessions": timeline.get("sessions", [])}
    elif name == "memory_timeline":
        project_id = arguments.get("project_id", "")
        days = arguments.get("days", 7)
        timeline = assemble_timeline(project_id, days=days)
        return {"timeline": timeline}
    else:
        raise ValueError(f"Unknown tool: {name}")

@app.list_tools()
async def list_tools():
    return [
        {
            "name": "memory_search",
            "description": "Search across hot (FTS5), warm (graph), cold (archive) tiers with intent-aware routing",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query"},
                    "limit": {"type": "number", "description": "Max results (default 10)"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "memory_sessions",
            "description": "List sessions for a project",
            "inputSchema": {
                "type": "object",
                "properties": {"project_id": {"type": "string"}},
                "required": ["project_id"]
            }
        },
        {
            "name": "memory_timeline",
            "description": "Generate project timeline markdown",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "days": {"type": "number", "description": "Lookback window (default 7)"}
                },
                "required": ["project_id"]
            }
        },
    ]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
