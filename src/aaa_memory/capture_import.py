"""Replay-safe opt-in ingestion of JSONL agent session transcripts.

The importer does not watch private directories or auto-upload anything. For
each input line it retains the original JSON representation after redaction,
with a stable source-file+line identity for deduplication.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .evidence_journal import EvidenceConflict, EvidenceJournal


def import_jsonl(
    path: str | Path,
    *,
    source: str,
    session_id: str | None = None,
    project: str = "unclassified",
    journal: EvidenceJournal | None = None,
) -> dict[str, Any]:
    """Import newline-delimited JSON, reporting malformed records explicitly.

    Once a line number has been committed, editing that line is a conflict, not
    a silent replacement. Log rotation should use a new stable filename.
    """
    file = Path(path).expanduser().resolve(strict=True)
    if not file.is_file() or file.is_symlink():
        raise ValueError("source must be a regular file")
    if file.suffix.lower() != ".jsonl":
        raise ValueError("only .jsonl session files can be imported")
    if not source.strip():
        raise ValueError("source must be nonempty")
    store = journal or EvidenceJournal()
    identifier = hashlib.sha256(str(file).encode("utf-8")).hexdigest()[:24]
    session = session_id or file.stem
    report: dict[str, Any] = {
        "source": source,
        "session_id": session,
        "inserted": 0,
        "duplicates": 0,
        "invalid_lines": [],
        "conflict_lines": [],
    }
    with file.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > 4 * 1024 * 1024:
                report["invalid_lines"].append(number)
                continue
            try:
                message = json.loads(line)
            except (ValueError, UnicodeError):
                report["invalid_lines"].append(number)
                continue
            if not isinstance(message, dict):
                report["invalid_lines"].append(number)
                continue
            event_kind = message.get("type", message.get("role", "event"))
            if not isinstance(event_kind, str) or not event_kind:
                event_kind = "event"
            when = message.get("timestamp")
            if not isinstance(when, str):
                when = None
            meta = {"line": number, "source_filename": file.name, "format": "jsonl"}
            try:
                result = store.append(
                    source=source,
                    external_id=f"{identifier}:{number}",
                    session_id=session,
                    project=project,
                    kind=event_kind,
                    content=line.rstrip("\r\n"),
                    source_path=str(file),
                    observed_at=when,
                    metadata=meta,
                )
                report["duplicates" if result["duplicate"] else "inserted"] += 1
            except EvidenceConflict:
                report["conflict_lines"].append(number)
            except (ValueError, TypeError):
                report["invalid_lines"].append(number)
    return report
