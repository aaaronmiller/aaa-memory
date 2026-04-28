# Quickstart: aaa-memory

## Prerequisites

- Python 3.12+
- Bun 1.1+ (for ClawMem)
- SQLite 3.35+ (with FTS5 and loadable extensions)
- Git

## Setup (Phase 0)

```bash
cd ~/code/aaa-memory

# Create knowledge base structure
mkdir -p ~/knowledge/{raw/{prds,youtube,papers,transcripts},wiki/{projects,research,concepts,prompts,code,decisions},references/{technical-writing,architecture-decisions,research-summaries,prd-templates},logs}

# Initialize git repo for knowledge base
cd ~/knowledge && git init && cd -

# Install Karpathy wiki schema
cd ~/knowledge
git clone https://github.com/joshpocock/karpathy-obsidian-vault wiki-schema
cp wiki-schema/CLAUDE.md ./CLAUDE.md
rm -rf wiki-schema

# Install ClawMem
cd ~/code
git clone https://github.com/yoloshii/ClawMem.git
cd ClawMem && bun install
clawmem init --vault ~/knowledge
clawmem setup hooks   # Claude Code hooks
clawmem setup mcp     # MCP tools

# Initialize aaa-memory Python package
cd ~/code/aaa-memory
pip install -e .

# Index existing wiki files (if any)
clawmem index --path ~/knowledge/wiki --rebuild
```

## Daily Use

```bash
# Drop a document into raw/
cp my_prd.md ~/knowledge/raw/prds/

# Tell any agent: "ingest the new PRD"
# Agent will: classify → extract → compile wiki → update index + log

# Search your knowledge
clawmem search "how did we handle auth in the proxy?"

# Lint the wiki for health
# Tell any agent: "lint the wiki for contradictions and orphans"

# Fix issues from lint report
# Tell any agent: "fix the issues reported in the last lint"
```

## Architecture Overview

```
~/knowledge/                    ← Karpathy Wiki (curated, git-tracked)
├── raw/                        ← Immutable source documents
├── wiki/                       ← LLM-compiled with [[wikilinks]]
│   ├── index.md               ← Master pointer table
│   └── {projects,research,concepts,prompts,code,decisions}/
└── CLAUDE.md                   ← Schema + agent workflow rules

~/.cache/clawmem/index.sqlite   ← Shared memory vault (SQLite + sqlite-vec)
~/code/aaa-memory/              ← Python orchestration package
```

## Verification

```bash
# Check ClawMem is indexed
clawmem stats

# Check hooks are installed
cat ~/.claude/settings.json | grep -i clawmem

# Test search
clawmem search "test"
```
