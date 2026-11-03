"""Content-addressed, opt-in artifact preservation for generated work.

Copy files into a private CAS before ephemeral workspaces disappear; register
source/evidence context in a SQLite catalog. Files are never removed or
overwritten implicitly. This does not capture remote ChatGPT cloud artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def default_root() -> Path:
    return Path(os.environ.get("AAA_ARTIFACT_ROOT", "~/.cache/aaa-memory/artifacts")).expanduser()


def _digest(path: Path) -> tuple[str, int]:
    hash_ = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hash_.update(block)
            size += len(block)
    return hash_.hexdigest(), size


class ArtifactRegistry:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root).expanduser() if root is not None else default_root()

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        dbpath = self.root / "catalog.sqlite"
        db = sqlite3.connect(dbpath, timeout=30)
        os.chmod(dbpath, 0o600)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA busy_timeout=30000")
            db.execute("""CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY,
                digest TEXT NOT NULL,
                size INTEGER NOT NULL,
                original_path TEXT NOT NULL,
                project TEXT NOT NULL,
                session_id TEXT,
                evidence_id INTEGER,
                registered_at TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS ix_artifact_digest ON artifacts(digest)")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _blob(self, digest: str) -> Path:
        return self.root / "blobs" / digest[:2] / digest

    def register(
        self, path: str | Path, *, project: str = "unclassified",
        session_id: str | None = None, evidence_id: int | None = None,
    ) -> dict:
        source = Path(path).expanduser().absolute()
        if source.is_symlink() or not source.is_file():
            raise ValueError("only regular, non-symlink files can be registered")
        if not stat.S_ISREG(source.stat().st_mode):
            raise ValueError("source must be a regular file")
        # Refuse common credential files by default. Callers must not submit
        # credential dumps or secrets disguised under ordinary filenames.
        forbidden = {".env", "id_rsa", "id_ed25519"}
        if source.name in forbidden or source.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
            raise ValueError("credential or private-key files cannot be registered")
        initial = source.stat()
        digest, size = _digest(source)
        blob = self._blob(digest)
        blob.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        if not blob.exists():
            fd, tmpname = tempfile.mkstemp(prefix=".capture-", dir=blob.parent)
            try:
                with os.fdopen(fd, "wb") as out, source.open("rb") as inp:
                    os.fchmod(out.fileno(), 0o600)
                    for block in iter(lambda: inp.read(1024 * 1024), b""):
                        out.write(block)
                    out.flush()
                    os.fsync(out.fileno())
                staged = Path(tmpname)
                staged_digest, staged_size = _digest(staged)
                if staged_digest != digest or staged_size != size:
                    raise RuntimeError("source changed while artifact was being copied")
                # link() publishes atomically without replacing an existing blob.
                try:
                    os.link(staged, blob)
                except FileExistsError:
                    pass
            finally:
                Path(tmpname).unlink(missing_ok=True)

        final = source.stat()
        if (initial.st_size != final.st_size or initial.st_mtime_ns != final.st_mtime_ns
                or _digest(blob) != (digest, size)):
            raise RuntimeError("source changed or blob failed validation")
        record = {
            "id": uuid.uuid4().hex,
            "digest": digest, "size": size,
            "original_path": str(source),
            "project": project,
            "session_id": session_id,
            "evidence_id": evidence_id,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._db() as db:
            db.execute("""INSERT INTO artifacts
                (id,digest,size,original_path,project,session_id,evidence_id,registered_at)
                VALUES (:id,:digest,:size,:original_path,:project,:session_id,:evidence_id,:registered_at)""", record)
        return record

    def get(self, artifact_id: str) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
            return dict(row) if row else None

    def list(self, project: str | None = None, limit: int = 30) -> list[dict]:
        with self._db() as db:
            sql = "SELECT * FROM artifacts"
            parameters: list = []
            if project is not None:
                sql += " WHERE project=?"
                parameters.append(project)
            sql += " ORDER BY registered_at DESC, id DESC LIMIT ?"
            parameters.append(max(1, min(int(limit), 1000)))
            return [dict(row) for row in db.execute(sql, parameters)]

    def verify(self, artifact_id: str) -> dict:
        record = self.get(artifact_id)
        if record is None:
            raise KeyError("unknown artifact ID")
        blob = self._blob(record["digest"])
        if not blob.is_file() or blob.is_symlink():
            return {"id": artifact_id, "ok": False, "reason": "blob missing"}
        digest, size = _digest(blob)
        return {"id": artifact_id, "ok": digest == record["digest"] and size == record["size"],
                "digest": digest, "size": size}

    def restore(self, artifact_id: str, destination: str | Path, *, overwrite: bool = False) -> Path:
        verification = self.verify(artifact_id)
        if not verification["ok"]:
            raise RuntimeError("artifact checksum validation failed")
        record = self.get(artifact_id)
        target = Path(destination).expanduser().absolute()
        if not target.parent.is_dir():
            raise FileNotFoundError("destination directory does not exist")
        if target.exists() and not overwrite:
            raise FileExistsError("destination already exists; no implicit replacement")
        fd, tmppath = tempfile.mkstemp(prefix=".aaa-restore-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as out, self._blob(record["digest"]).open("rb") as inp:
                os.fchmod(out.fileno(), 0o600)
                for block in iter(lambda: inp.read(1024 * 1024), b""):
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
            if _digest(Path(tmppath)) != (record["digest"], record["size"]):
                raise RuntimeError("restored contents did not match artifact digest")
            if overwrite:
                os.replace(tmppath, target)
            else:
                # Atomic exclusive publish when another process creates target.
                os.link(tmppath, target)
        finally:
            Path(tmppath).unlink(missing_ok=True)
        return target
