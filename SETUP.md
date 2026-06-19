# aaa-memory Setup Guide

## Quick Start
```bash
pip install -e .
python3 scripts/search.py --help
```

## Vault Locations
- Hot tier: `~/.cache/aaa-memory/vault.sqlite`
- Cold tier: `~/.cache/aaa-memory/cold.sqlite`
- Logs: `~/.cache/aaa-memory/logs/`

## Commands
- `python3 scripts/search.py "query"` — Search all tiers
- `python3 scripts/search.py --sessions` — List sessions
- `python3 scripts/search.py --timeline PROJECT` — Project timeline
- `python3 scripts/transition_hot_warm.py` — Run hot→warm transition
- `python3 scripts/transition_warm_cold.py` — Run warm→cold transition
- `python3 scripts/batch_extract.py "raw/*.jsonl"` — Batch extraction
- `python3 scripts/overnight_improve.py` — Re-evaluate low-confidence elements

## Agent Integration
- **MCP**: `python3 -m aaa_memory.mcp` (stdio transport)
- **Hermes**: Configure `MemoryProvider = "aaa_memory.hermes.provider.HermesMemoryProvider"`
- **OpenClaw**: Add `aaa_memory.openclaw.plugin.AaaMemoryPlugin` to plugin chain
