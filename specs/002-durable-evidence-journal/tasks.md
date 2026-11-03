# Feature 002 — implementation tasks

- [x] T001 Create source-linked append-only event model with stable event identity.
- [x] T002 Add SQLite FTS5 compact search, get-by-ID and timeline.
- [x] T003 Add redaction/private-exclusion and content-integrity digests.
- [x] T004 Add CLI commands for append/search/get/timeline/verify.
- [x] T005 Repair missing `json` import and audit cache directory handling.
- [x] T006 Add regression tests covering idempotency, integrity, privacy and concurrency.
- [x] T007 Confirm focused CI passes and address concurrency/CLI compatibility failures.
- [x] T008a Implement explicit idempotent JSONL transcript import and malformed-line diagnostics.
- [ ] T008b Install actual native harness hooks, cursors, and continuous replay.
- [x] T009a Add manual content-addressed artifact registry with verification and safe restore.
- [ ] T009b Automatically capture produced files and integrate with verified task/work states.
- [ ] T010 Validate against the 2026-10-26 platform acceptance scenarios.
