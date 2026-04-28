# aaa-memory

**Personal AI Interaction Archive** — Multi-tier memory system for AI agent sessions.

> "Everything I've ever told any model, searchable everywhere."

## Architecture

```
Ingestion → Extraction → Wiki Compilation → Tiered Storage → Unified Retrieval
    │            │               │                    │                │
    │            │               │                    │                ├─ Hot (ClawMem: SQLite FTS5 + vec)
    │            │               │                    │                ├─ Warm (Graphiti: knowledge graph)
    │            │               │                    │                └─ Cold (MemVid V2: compressed archive)
    │            │               │
    ├─ Document classifier (rule + LLM)
    ├─ Element extractor (LLM + regex fallback)
    └─ Metadata injector (YAML frontmatter, [[wikilinks]])
```

## Storage

- **Vault**: `~/.cache/aaa-memory/vault.sqlite` (SQLite + sqlite-vec)
- **Wiki**: `~/knowledge/` (Karpathy-style markdown with frontmatter)
- **Raw captures**: `~/knowledge/raw/{transcripts,prds,web}/`

## Quickstart

```bash
# 1. Install
pip install -e .

# 2. Initialize knowledge base
mkdir -p ~/knowledge/{raw,wiki,index}
git init ~/knowledge

# 3. Install ClawMem hooks (captures Claude Code automatically)
clawmem init
clawmem setup hooks
clawmem setup mcp

# 4. Run daily capture service (cron)
crontab -e
# Add: 0 2 * * * /home/youruser/code/aaa-memory/scripts/daily-update.sh

# 5. Test ingestion on a transcript
python scripts/vault_classify.py
python scripts/vault_extract.py
python scripts/review_extractions.py  # optional human review
```

## Commands

```bash
# Classify all files in ~/knowledge/raw
python scripts/vault_classify.py

# Extract elements from transcripts (after classification)
python scripts/vault_extract.py [--resume]

# Interactive extraction review
python scripts/review_extractions.py

# Wiki lint & auto-fix
python src/aaa_memory/wiki/linter.py
python scripts/wiki_auto_fix.py [--auto-approve]

# Session audit
aaa-memory sessions              # list discovered agent storage
aaa-memory timeline <project>    # generate project timeline
aaa-memory audit --update        # full refresh

# Hot search
python -c "from aaa_memory.retrieval.hot import search; print(search('query'))"
```

## Configuration

| Env var | Purpose | Default |
|----------|---------|---------|
| `AAA_MEMORY_VAULT` | Path to SQLite vault | `~/.cache/aaa-memory/vault.sqlite` |
| `OPENROUTER_API_KEY` | LLM classifier & extractor (Nemotron 3 Super) | required for LLM features |
| `JINA_API_KEY` | Cloud embedding fallback | none |
| `VLLM_ENDPOINT` | vLLM OpenAI-compatible endpoint for Qwen3-Embedding | none |

## Tech Stack

- **Python 3.12+**
- **SQLite + sqlite-vec** — hot tier storage + FTS5
- **sentence-transformers** — embeddings (all-MiniLM-L6-v2 fallback, Gemma-300M when available)
- **OpenAI client** — OpenRouter for Nemotron 3 Super (free)
- **rich** — CLI UI

## Development Status

**Phase 1–2**: ✅ Complete  
- Project structure, ClawMem hooks, Tampermonkey scripts, daily service  
- Document classifier (rule + LLM), element extractor (LLM + fallback), metadata injector, embedding encoder  
- Wiki compiler + indexer, linter  
- Vault ingestion scripts, session audit discovery, CLI, MCP stub, hot retrieval (FTS5)  

**Phase 3+**: 🔄 In Progress  
- Full 800-file vault migration (needs actual vault data)  
- Graphiti warm tier integration (placeholder)  
- MemVid V2 cold tier (placeholder)  
- Agent plugins (OpenClaw, Hermes, Qwen, OpenCode, Codex) — stubs written  
- Tier transitions & overnight improvement — scripts in place, untested  

## License

MIT
