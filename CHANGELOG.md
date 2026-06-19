# Changelog

## [1.0.0] — 2026-06-17
### Added
- Complete multi-tier memory pipeline (hot/warm/cold)
- Intent-aware retrieval router with Nemotron classification
- Cross-encoder reranker (Qwen3-Reranker-0.6B / cosine fallback)
- MCP server with 4 tools (search, sessions, timeline, store)
- CLI search tool with --intent, --project, --timeline, --sessions, --json
- Session audit system (discover, parse, classify, timeline, decisions)
- Key decision extractor with confidence scoring
- Cold storage FTS5 with archive_turns()/search_archive()
- Agent integrations: Claude hooks, OpenClaw plugin, Hermes provider
- Agent parsers: Qwen, OpenCode, Codex
- Tier transitions: hot→warm (weekly), warm→cold (monthly)
- Overnight improvement loop for low-confidence elements
- Post-transition reporting
- Structured JSON logging with rotation
- E2E tests and performance benchmarks
- SETUP.md documentation
- Batch extraction with checkpoint support
- Wiki indexing into FTS5 vault
- All planning docs archived in docs/archive/

### Fixed
- mcp.py — completed stub with full tool implementations
- cold.py — replaced placeholder with real FTS5 cold storage
- Removed 6 empty stub module directories
- Moved local-inference/llama-cpp-turboquant to ~/git/
