"""Artifact content-addressed preservation, safe restore, and corruption tests."""
import os

import pytest

from aaa_memory.artifact_registry import ArtifactRegistry


def test_register_deduplicate_and_restore(tmp_path):
    registry = ArtifactRegistry(tmp_path / "cas")
    work = tmp_path / "workspace"
    work.mkdir()
    path = work / "output.txt"
    path.write_text("durable generated work\n", encoding="utf-8")
    first = registry.register(path, project="aaa-memory", session_id="session-1", evidence_id=5)
    assert first["size"] == len(path.read_bytes())
    second = registry.register(path, project="aaa-memory", session_id="session-2")
    assert second["digest"] == first["digest"]
    assert first["id"] != second["id"]
    assert len(list((tmp_path / "cas" / "blobs").rglob(first["digest"]))) == 1
    assert registry.verify(first["id"])["ok"]
    assert len(registry.list(project="aaa-memory")) == 2

    path.unlink()
    restored = registry.restore(first["id"], work / "restored.txt")
    assert restored.read_text(encoding="utf-8") == "durable generated work\n"
    with pytest.raises(FileExistsError):
        registry.restore(first["id"], restored)
    assert registry.restore(first["id"], restored, overwrite=True) == restored


def test_artifact_corruption_blocks_restore(tmp_path):
    registry = ArtifactRegistry(tmp_path / "cas")
    source = tmp_path / "output.html"
    source.write_text("original report")
    row = registry.register(source)
    blob = registry._blob(row["digest"])
    blob.write_text("tampered content")
    assert not registry.verify(row["id"])["ok"]
    with pytest.raises(RuntimeError):
        registry.restore(row["id"], tmp_path / "restored.html")


def test_secret_and_symlink_rejection(tmp_path):
    registry = ArtifactRegistry(tmp_path / "cas")
    key = tmp_path / ".env"
    key.write_text("PASSWORD=secret")
    with pytest.raises(ValueError):
        registry.register(key)
    target = tmp_path / "data.txt"
    target.write_text("ordinary data")
    link = tmp_path / "shortcut.txt"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        registry.register(link)
    assert registry.list() == []


def test_restore_missing_id_and_missing_parent(tmp_path):
    registry = ArtifactRegistry(tmp_path / "cas")
    source = tmp_path / "source.txt"
    source.write_text("source")
    record = registry.register(source)
    with pytest.raises(KeyError):
        registry.restore("no-such-id", tmp_path / "missing.txt")
    with pytest.raises(FileNotFoundError):
        registry.restore(record["id"], tmp_path / "new-dir" / "restored.txt")


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode-bit assertion")
def test_private_permissions(tmp_path):
    registry = ArtifactRegistry(tmp_path / "cas")
    source = tmp_path / "report.md"
    source.write_text("private content")
    record = registry.register(source)
    assert registry._blob(record["digest"]).stat().st_mode & 0o777 == 0o600
    assert (registry.root / "catalog.sqlite").stat().st_mode & 0o777 == 0o600
