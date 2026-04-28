# Changelog

All notable changes to aaa-memory will be documented in this file.

## [Unreleased]

### Added
- Initial project scaffold — Spec-Kit tasks generated from `spec.md` + `plan.md`
- Core Python package `aaa_memory` with modules:
  - `classifier`: rule-based + LLM fallback document classification
  - `extractor`: LLM-based knowledge element extraction with regex fallback
  - `metadata`: YAML frontmatter injector with wikilink detection
  - `embedding`: multi-provider encoder (Gemma300M via sentence-transformers, Jina fallback)
  - `wiki`: markdown compiler + Karpathy pointer indexer + linter (orphans, dead links, stale claims)
  - `retrieval`: hot tier FTS5 search, RRF fusion stub, intent router stub
  - `audit`: cross-agent session discovery, parser (Claude/OpenClaw/Web stubs), classifier, timeline assembler
  - `cli`: `aaa-memory sessions`, `timeline`, `audit` commands
  - `mcp`: minimal MCP server stub with 2 tools
- Infrastructure:
  - Daily update service (`scripts/daily-update.py`) with SQLite WAL schema
  - Vault classification & extraction scripts (`vault_classify.py`, `vault_extract.py`)
  - Interactive extraction review UI (`review_extractions.py`)
  - Wiki linting + auto-fix daemon stub
  - Transition scripts (hot→warm, warm→cold, overnight improve) — placeholders
- Documentation:
  - `README.md` with quickstart, architecture, configuration
  - Specs with full PRD (`spec.md`), implementation plan (`plan.md`), data schema (`schema.sql`)

### Fixed
- n/a (initial commit)

### Changed
- n/a

### Removed
- n/a

## [0.1.0] — 2026-04-27

Initial prototype. Not yet production-ready.
