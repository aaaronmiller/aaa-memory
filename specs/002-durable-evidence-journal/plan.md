# Feature 002 — implementation plan

## Architecture

Agent hook or parser → synchronous evidence append → SQLite event row and FTS trigger → compact search/timeline → explicit full detail. No background observer is required on the capture path.

The separate journal avoids changing the existing vault schema while the October 2026 canonical schema migration remains pending. Capture transactions are short and each event uses a source-controlled stable ID. Duplicate events with the same content hash return their existing ID. Conflicting event IDs fail closed.

## Dependencies and migration

Python 3.12 standard library and SQLite FTS5. No new runtime dependency. The journal is additive. Existing vault tables and Cass/ClawMem indexes remain untouched. To recover a journal, restore the SQLite database from a trustworthy backup; `verify` detects only limited forms of corruption and does not replace backups.

## Acceptance

Run `python -m pytest -q tests/test_evidence_journal.py tests/test_capture_import.py tests/test_artifact_registry.py`. Assert normal search, timeline, repeat replay, rejection of rewritten source IDs, privacy exclusions, user-query safety, concurrent append and stored-content tamper detection. Later benchmark capture throughput, sqlite lock contention and p95 search latency on actual hardware.

## Release gates

- Unit suite passes on Python 3.12 with SQLite FTS5.
- Existing import paths and `aaa-memory` commands continue to operate.
- No installation scripts modify shell startup files, source archives, or active agent configurations.
- Review before merge; never automatically ingest credentials or untrusted source directories.

## Follow-on implementation

JSONL importer is explicit and per-file; parsing/ingestion reports bad line numbers, and a source-file/line ID conflict prevents silent rewrite. The artifact registry keeps user-selected generated files under a private SHA-256 blob store, with atomic publishing and checked recovery. Restore operations never overwrite unless explicitly requested. Future adapters must register artifacts at creation time, before temporary workspaces are removed.
