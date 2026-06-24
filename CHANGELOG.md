# Changelog

## [Unreleased]
### Added
- Added shared SSE daemon mode for the aaa-memory MCP server via `aaa-memory-mcp serve`.
- Added CLI-backed Codex, OpenCode, Qwen, and Pi integration helpers for memory search/context access.
- Added tests for CLI-backed agent integration command construction.
- Added a Cass-backed Claude `UserPromptSubmit` fallback hook for bounded prior-session prompt history context.

### Fixed
- Restored the public `parse_opencode_sessions` export to avoid breaking existing imports.
- Fixed Codex/OpenCode/Pi turn recording helpers to use the real `clawmem diary write` command instead of a nonexistent `clawmem hook store-turn`.
- Fixed OpenCode/Pi memory search helpers to use `clawmem search -n ... --json` instead of unsupported `--limit` output parsing.
- Replaced ad hoc aaa-memory MCP argument parsing with `argparse` validation for bad daemon flags.
- Fixed low-confidence classifier fallback so `unknown` rule results can still use the LLM classifier.
- Fixed unified retrieval so hot/warm/wiki searches use the live SQLite FTS and ClawMem `/retrieve` schemas.
- Fixed hot-to-warm and warm-to-cold transition code to work with the current `turns.created_at`/`metadata` vault schema.
- Removed a misplaced pytest config entry that emitted an unknown-option warning on every test run.

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
