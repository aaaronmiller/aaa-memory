# Feature 002 — implementation plan

## Architecture

Agent hook or parser → synchronous evidence append → SQLite event row and FTS trigger → compact search/timeline → explicit full detail. No background observer is required on the capture path.

The separate journal avoids changing the existing vault schema while the October 2026 canonical schema migration remains pending. Capture transactions are short and each event uses a source-controlled stable ID. Duplicate events with the same content hash return their existing ID. Conflicting event IDs fail closed.

## Dependencies and migration

Python 3.12 standard library and SQLite FTS5. No new runtime dependency. The journal is additive. Existing vault tables and Cass/ClawMem indexes remain untouched. To recover a journal, restore the SQLite database from a trustworthy backup; `verify` detects only limited forms of corruption and does not replace backups.

## Acceptance

Run `python -m pytest -q tests/test_evidence_journal.py`. Assert normal search, timeline, repeat replay, rejection of rewritten source IDs, privacy exclusions, user-query safety, concurrent append and stored-content tamper detection. Later benchmark capture throughput, sqlite lock contention and p95 search latency on actual hardware.

## Release gates

- Unit suite passes on Python 3.12 with SQLite FTS5.
- Existing import paths and `aaa-memory` commands continue to operate.
- No installation scripts modify shell startup files, source archives, or active agent configurations.
- Review before merge; never automatically ingest credentials or untrusted source directories.
