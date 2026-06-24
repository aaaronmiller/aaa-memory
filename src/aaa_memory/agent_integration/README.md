# Agent Integrations - Patch/Plugin Pattern

Each agent tool integrates aaa-memory as a **library**, not as an MCP server.
This saves ~1-2GB per session by avoiding per-session process spawning.

## Pattern: In-Process Lazy Loading (Hermes reference)

```
Tool Plugin/Hook -> aaa-memory library (lazy import) -> SQLite vault
```

No IPC, no MCP server process. The library is loaded only when first called.

## Supported Integrations

| Tool | Plugin Format | Status | Integration |
|---|---|---|---|
| **Hermes** | Python plugin (`plugins/`) | Working | `hermes/provider.py` - `MemoryProvider` with lazy imports |
| **Claude Code** | Hooks (`settings.json`) | Working | `claude/hooks.py` - record/surface via hook CLI |
| **Claude Code MCP** | `--mcp-config` (SSE daemon) | Working | `mcp.py serve` - shared SSE daemon on `:7437` |
| **OpenCode** | npm plugin (`opencode plugin`) | Planned | Calls `clawmem` CLI for record/search |
| **Qwen Code** | Extension (`qwen extensions`) | Planned | Calls `clawmem` CLI for record/search |
| **Codex** | Plugin (`codex plugin`) | Planned | Calls `clawmem` CLI for record/search |
| **Pi** | CLI-only | Planned | Runs `clawmem hook` / `mem.py` for record/search |

## How to Add a New Integration

1. Create `<tool>/` directory in this folder
2. Implement the integration using one of:
   - **Python library import**: For Python-based tools (like Hermes)
   - **CLI subprocess**: For non-Python tools, call `clawmem` or `python3 -m aaa_memory`
3. Add a factory function to `__init__.py`
4. Document the tool-side plugin/hook registration

## Performance Comparison

| Approach | Latency | Memory | Process Count |
|---|---|---|---|
| **Per-session stdio MCP** | ~2-5s start + IPC | ~1-2GB per session | 1 per session |
| **SSE daemon MCP** | ~10ms connect | ~1-2GB total | 1 shared |
| **In-process library (lazy)** | ~0ms (zero until call) | ~0 until first call | 0 extra |
