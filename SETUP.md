# aaa-memory Setup Guide

## Quick Start
```bash
pip install -e .
python3 scripts/search.py --help
```

## Required Cass Setup

`aaa-memory` uses Cass as the raw coding-agent session evidence tier. A complete install now includes Cass plus a cron refresh so session search stays current.

Install Cass:

```bash
curl -fsSL "https://raw.githubusercontent.com/Dicklesworthstone/coding_agent_session_search/main/install.sh?$(date +%s)" \
  | bash -s -- --easy-mode --verify
```

Initialize or inspect Cass:

```bash
cass triage --json
cass index --json --no-progress-events --data-dir "$HOME/.local/share/coding-agent-search"
```

Install the four-times-daily Cass refresh cron job:

```bash
CASS_BIN="$(command -v cass)"
mkdir -p "$HOME/.cache/aaa-memory/logs"
( crontab -l 2>/dev/null | grep -v 'aaa-memory cass refresh'; \
  printf '0 */6 * * * %s index --json --no-progress-events --data-dir "$HOME/.local/share/coding-agent-search" >> "$HOME/.cache/aaa-memory/logs/cass-index.log" 2>&1 # aaa-memory cass refresh\n' "$CASS_BIN" ) | crontab -
```

Verify:

```bash
cass triage --json
crontab -l | grep 'aaa-memory cass refresh'
```

Semantic Cass models remain optional. Install one only when semantic refinement is needed:

```bash
cass models install --model all-minilm-l6-v2
```

## Vault Locations
- Hot tier: `~/.cache/aaa-memory/vault.sqlite`
- Cold tier: `~/.cache/aaa-memory/cold.sqlite`
- Logs: `~/.cache/aaa-memory/logs/`
- Cass index: `~/.local/share/coding-agent-search`

## Commands
- `python3 scripts/search.py "query"` — Search all tiers
- `cass triage --json` — Check Cass readiness and recommended next command
- `cass search "query" --robot --robot-meta` — Search raw coding-agent sessions
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
