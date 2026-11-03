"""Regression tests for evidence capture, replay, confidentiality and recall."""
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from aaa_memory.evidence_journal import EvidenceConflict, EvidenceJournal, redact


@pytest.fixture
def journal(tmp_path):
    return EvidenceJournal(tmp_path / "vault" / "evidence.sqlite")


def test_roundtrip_search_timeline_and_verify(journal):
    saved = journal.append(
        source="codex", external_id="event-001", session_id="session-A",
        project="aaa-memory", kind="decision", content="The retrieval engine uses exact provenance.",
        metadata={"turn": 1},
    )
    assert saved["duplicate"] is False
    row = journal.get(saved["id"])
    assert row["content"] == "The retrieval engine uses exact provenance."
    assert row["project"] == "aaa-memory"
    assert json.loads(row["metadata_json"]) == {"turn": 1}
    assert journal.search("retrieval", project="aaa-memory")[0]["id"] == saved["id"]
    assert journal.search("retrieval", project="other") == []
    assert journal.timeline("session-A")[0]["id"] == saved["id"]
    assert journal.verify() == {"events": 1, "indexed": 1, "corrupt_ids": [], "ok": True}


def test_replayed_event_is_idempotent_and_conflicts_are_rejected(journal):
    params = dict(source="claude", external_id="same", session_id="session-B",
                  content="Stable decision")
    first = journal.append(**params)
    second = journal.append(**params)
    assert first["id"] == second["id"]
    assert second["duplicate"]
    assert journal.verify()["events"] == 1
    with pytest.raises(EvidenceConflict):
        journal.append(**{**params, "content": "Rewritten decision"})


def test_redaction_and_context_exclusion(journal):
    secret = "ghp_" + "a" * 24
    text = (
        "Before <private>Do not save this</private> "
        "<!-- AAA_MEMORY_CONTEXT_START -->Injected text<!-- AAA_MEMORY_CONTEXT_END --> "
        "Authorization: Bearer amazingCredential "
        "--client-secret=hidden-value "
        f"{secret} after"
    )
    saved = journal.append(source="claude", external_id="redaction", session_id="S", content=text,
                           metadata={"auth": "Authorization: Bearer metadataSecret"})
    row = journal.get(saved["id"])
    for value in ("Do not save this", "Injected text", "amazingCredential",
                  "hidden-value", secret, "metadataSecret"):
        assert value not in row["content"] + row["metadata_json"]
    assert "[REDACTED]" in row["content"]
    assert row["raw_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert row["stored_sha256"] == hashlib.sha256(row["content"].encode()).hexdigest()


def test_fts_handles_untrusted_query_input(journal):
    journal.append(source="pi", external_id="1", session_id="X",
                   content="A tricky phrase about SQL databases")
    for query in ('" OR NOT *', 'SQL: !(bad)', '""', "12345"):
        assert isinstance(journal.search(query), list)
    assert journal.search("database OR SQL")


def test_4_mib_limit_and_input_validation(journal):
    with pytest.raises(ValueError):
        journal.append(source="", external_id="a", session_id="b", content="hi")
    with pytest.raises(ValueError):
        journal.append(source="x", external_id="a", session_id="b", content="z" * (4 * 1024 * 1024 + 1))


def test_parallel_identical_events_do_not_duplicate(journal):
    def write(_):
        return journal.append(source="opencode", external_id="event", session_id="S",
                              content="same")

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(write, range(4)))
    assert len({x["id"] for x in outputs}) == 1
    assert journal.verify()["events"] == 1


def test_verify_detects_content_tampering(journal):
    saved = journal.append(source="a", external_id="b", session_id="c", content="original")
    with sqlite3.connect(journal.path) as db:
        db.execute("UPDATE evidence_events SET content=? WHERE id=?", ("modified", saved["id"]))
    verified = journal.verify()
    assert not verified["ok"]
    assert verified["corrupt_ids"] == [saved["id"]]
