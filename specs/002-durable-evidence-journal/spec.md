# Feature 002 — Durable evidence capture and progressive recall

**Status:** implemented first slice, pending CI verification
**Date:** 2026-11-02
**Base:** AAA-Memory rebuild v2.0 (2026-10-26), evidence/provenance, progressive disclosure and reliable capture requirements.

## User outcomes

1. An agent can append source-linked evidence without requiring an LLM or vector index.
2. Replaying the same source event is a no-op; a changed payload with the same source event ID raises a conflict rather than silently rewriting evidence.
3. Retrieval returns a compact index; full contents require an explicit evidence ID lookup.
4. A timeline reconstructs the ordered events from one session.
5. Private spans, injected prior context and common credential formats are redacted before persistence.
6. A verification command finds stored-text tampering; a missing raw source cannot be reconstructed from a digest.
7. Existing hot/warm/cold retrieval, MCP tools, and legacy scripts continue to function.

## Interface contracts

- `EvidenceJournal.append(source, external_id, session_id, content, ...)` returns `{id, duplicate, stored_sha256}`.
- `EvidenceJournal.search(query, limit, project)` returns small citations and 160-character excerpts.
- `EvidenceJournal.get(id)` returns a complete stored event or null.
- `EvidenceJournal.timeline(session_id, limit)` returns ordered session events.
- `EvidenceJournal.verify()` checks SHA-256 of each stored content body and compares FTS row count.
- `AAA_EVIDENCE_DB` overrides the journal path; default is `~/.cache/aaa-memory/evidence.sqlite`.
- `aaa-memory evidence append|search|get|timeline|verify` provides operator access.

## Safety boundaries

The journal never executes source content and does not store secrets knowingly. Redaction is best-effort, not a guarantee against every secret format. Keep access to the SQLite journal restricted to trusted users. Original source bytes remain in their owning application; the journal retains a digest only, plus the redacted stored text. Event source identifiers must be stable, otherwise retry deduplication cannot work.

## Explicitly deferred

- Automatic native hook installation and streaming parsers for every harness.
- Automatic artifact discovery and retention/garbage-collection policies; the manual content-addressed registry is implemented.
- Queue leases/dead-letter handling for derived AI processing.
- Cross-host replication, ACLs, encryption at rest.
- Quota engine, Dreamtime task orchestrator, learned MEMORILLA readout.
- User-friendly Web UI and the full analytics catalog.

This increment is not the complete 78-requirement platform rebuild.

## Implemented follow-on additions

- `aaa-memory evidence import-jsonl` opt-in transcript intake: stable source-file/line IDs, unchanged JSON payloads subject to redaction, replay deduplication, explicit invalid-line and conflict counts.
- `aaa-memory artifact register|list|get|verify|restore`: private content-addressed copies of user-selected local files, SHA-256 verification, explicit non-overwrite recovery and optional source session / evidence reference IDs.
- Artifact snapshots are not automatically validated against human acceptance criteria, and they do not recover files originally generated exclusively in an inaccessible cloud environment.
