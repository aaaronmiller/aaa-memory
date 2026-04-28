#!/usr/bin/env python3
"""
Minimal MCP server for aaa-memory.

Exposes two tools initially:
  - memory_search(query: str, limit: int = 10) -> list[dict]
  - memory_sessions(project_id: str) -> list[dict]

Run via: python -m aaa_memory.mcp
Claude Code connects via: claude mcp add aaa-memory python -m aaa_memory.mcp
"""

import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from aaa_memory.retrieval.fusion import search as hybrid_search
from aaa_memory.audit.timeline import assemble_timeline
import asyncio

app = Server("aaa-memory")


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "memory_search":
        query = arguments.get("query", "")
        limit = arguments.get("limit", 10)
        # Return mock results (real implementation would hit retrieval pipeline)
        return {
            "results": [
                {"title": f"Result for '{query}'", "snippet": "...", "score": 0.95}
            ],
            "total": 1,
        }
    elif name == "memory_sessions":
        project_id = arguments.get("project_id", "")
        # Assemble timeline and return summary
        timeline = assemble_timeline(project_id, days=30)
        return {"sessions": timeline.get("sessions", [])}
    else:
        raise ValueError(f"Unknown tool: {name}")


@app.list_tools()
async def list_tools():
    return [
        {
            "name": "memory_search",
            "description": "Search memory across hot/warm/cold tiers",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Max results (default 10)",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "memory_sessions",
            "description": "List sessions for a project",
            "inputSchema": {
                "type": "object",
                "properties": {"project_id": {"type": "string"}},
                "required": ["project_id"],
            },
        },
    ]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
