"""JSONL transcript ingestion and replay tests."""
import json
from pathlib import Path

import pytest

from aaa_memory.capture_import import import_jsonl
from aaa_memory.evidence_journal import EvidenceJournal


def test_import_replay_and_changed_line_detection(tmp_path):
    path = tmp_path / "claude-session.jsonl"
    source = [
        {"type": "user", "timestamp": "2026-11-02T12:00:00Z", "message": {"content": "design source IDs"}},
        {"type": "assistant", "timestamp": "2026-11-02T12:00:01Z", "message": {"content": "Use stable IDs"}},
    ]
    path.write_text("\n".join(json.dumps(line) for line in source) + "\n", encoding="utf-8")
    journal = EvidenceJournal(tmp_path / "journal.sqlite")
    first = import_jsonl(path, source="claude-code", project="aaa", journal=journal)
    assert first["inserted"] == 2
    assert first["duplicates"] == 0
    assert journal.verify()["events"] == 2

    second = import_jsonl(path, source="claude-code", project="aaa", journal=journal)
    assert second["duplicates"] == 2
    assert second["inserted"] == 0
    assert second["conflict_lines"] == []

    source[0]["message"]["content"] = "substituted decision"
    path.write_text("\n".join(json.dumps(line) for line in source) + "\n", encoding="utf-8")
    conflict = import_jsonl(path, source="claude-code", project="aaa", journal=journal)
    assert conflict["conflict_lines"] == [1]
    assert conflict["duplicates"] == 1
    assert journal.verify()["events"] == 2
    assert "substituted" not in journal.get(1)["content"]


def test_malformed_lines_are_reported_and_never_silent(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text('{"type":"user","timestamp":"bad-date"}\nnot-json\n[]\n{"type":"assistant"}\n', encoding="utf-8")
    journal = EvidenceJournal(tmp_path / "journal.sqlite")
    result = import_jsonl(path, source="codex", journal=journal)
    assert result["inserted"] == 1
    assert result["invalid_lines"] == [1, 2, 3]
    assert journal.verify()["events"] == 1


def test_refuses_non_jsonl_or_missing_source(tmp_path):
    path = tmp_path / "credentials.env"
    path.write_text("TOP_SECRET=bad\n", encoding="utf-8")
    with pytest.raises(ValueError):
        import_jsonl(path, source="codex", journal=EvidenceJournal(tmp_path / "journal.sqlite"))
    path = tmp_path / "regular.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        import_jsonl(path, source="", journal=EvidenceJournal(tmp_path / "journal.sqlite"))
