"""Durable local catalog. One connection per operation; SQLite owns write locking.

Block payloads are immutable content-addressed files outside SQLite. Files are
fsynced before references are committed. Crashes may leave unreferenced files,
but never intentionally publish references before the corresponding file exists.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import (
    BlockKind, BoundingBox, CoordinateSpace, IngestionJob, IngestionStage,
    IngestionStatus, PageSpan, Paper, PaperVersion, SourceBlock, SourceKind,
    VersionState,
)


class CatalogConflict(ValueError):
    """Identity conflict, invalid transition, deleted paper or stale worker."""


def _now():
    return datetime.now(timezone.utc)


def _decode(cls, raw):
    data = json.loads(raw)
    for key in ("created_at", "updated_at", "lease_expires_at"):
        if data.get(key) is not None:
            data[key] = datetime.fromisoformat(data[key])
    if cls is PaperVersion:
        data["state"] = VersionState(data["state"])
    if cls is IngestionJob:
        data["status"] = IngestionStatus(data["status"])
        data["stage"] = IngestionStage(data["stage"])
    if cls is SourceBlock:
        data["extraction_warnings"] = tuple(data.get("extraction_warnings", ()))
        data["kind"] = BlockKind(data["kind"])
        data["source_kind"] = SourceKind(data["source_kind"])
        data["section_path"] = tuple(data["section_path"])
        data["page_spans"] = tuple(PageSpan(**x) for x in data["page_spans"])
        data["bboxes"] = tuple(BoundingBox(**dict(
            x, coordinate_space=CoordinateSpace(x["coordinate_space"])
        )) for x in data["bboxes"])
    return cls(**data)


class SQLitePaperCatalog:
    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.block_dir = self.path.parent / (self.path.name + ".blocks")
        self.block_dir.mkdir(exist_ok=True)
        with self._connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("BEGIN IMMEDIATE")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, self.SCHEMA_VERSION):
                raise CatalogConflict(f"Unsupported schema version: {version}")
            if version == 0:
                statements = (
                    "CREATE TABLE papers (paper_id TEXT PRIMARY KEY, data TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0)",
                    "CREATE TABLE versions (paper_id TEXT NOT NULL REFERENCES papers(paper_id), version TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY(paper_id,version))",
                    "CREATE TABLE active_versions (paper_id TEXT PRIMARY KEY, version TEXT NOT NULL, FOREIGN KEY(paper_id,version) REFERENCES versions(paper_id,version))",
                    "CREATE TABLE blocks (paper_id TEXT NOT NULL, version TEXT NOT NULL, ordinal INTEGER NOT NULL, block_id TEXT NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(paper_id,version,block_id), UNIQUE(paper_id,version,ordinal), FOREIGN KEY(paper_id,version) REFERENCES versions(paper_id,version))",
                    "CREATE TABLE jobs (job_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, version TEXT NOT NULL, data TEXT NOT NULL, UNIQUE(paper_id,version), FOREIGN KEY(paper_id,version) REFERENCES versions(paper_id,version))",
                )
                for sql in statements:
                    db.execute(sql)
                db.execute("PRAGMA user_version=1")

    @contextmanager
    def _connection(self, *, write=False):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA synchronous=FULL")
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _live(self, db, paper_id):
        row = db.execute("SELECT deleted FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
        if row is None or row[0]:
            raise CatalogConflict("Paper missing or deleted")

    def create_paper(self, paper: Paper) -> None:
        with self._connection(write=True) as db:
            row = db.execute("SELECT data,deleted FROM papers WHERE paper_id=?", (paper.paper_id,)).fetchone()
            if row:
                old = _decode(Paper, row[0])
                if row[1] or replace(old, created_at=paper.created_at) != paper:
                    raise CatalogConflict("Paper ID already has different content")
                return
            db.execute("INSERT INTO papers(paper_id,data) VALUES (?,?)", (paper.paper_id, paper.to_json()))

    def get_paper(self, paper_id: str) -> Paper | None:
        with self._connection() as db:
            row = db.execute("SELECT data FROM papers WHERE paper_id=? AND deleted=0", (paper_id,)).fetchone()
            return _decode(Paper, row[0]) if row else None

    def _version(self, db, paper_id, version):
        row = db.execute("SELECT data FROM versions WHERE paper_id=? AND version=?", (paper_id, version)).fetchone()
        return _decode(PaperVersion, row[0]) if row else None

    def save_version(self, version: PaperVersion) -> None:
        """Create a staging version. Existing version identities are immutable."""
        with self._connection(write=True) as db:
            self._live(db, version.paper_id)
            old = self._version(db, version.paper_id, version.version)
            if old:
                if replace(old, created_at=version.created_at) != version:
                    raise CatalogConflict("Version is immutable; create a new version")
                return
            if version.state is not VersionState.STAGING:
                raise CatalogConflict("New version must be staging")
            db.execute("INSERT INTO versions VALUES (?,?,?)", (version.paper_id, version.version, version.to_json()))

    def get_version(self, paper_id: str, version: str) -> PaperVersion | None:
        with self._connection() as db:
            return self._version(db, paper_id, version)

    def get_active_version(self, paper_id: str) -> PaperVersion | None:
        with self._connection() as db:
            row = db.execute(
                "SELECT v.data FROM active_versions a JOIN versions v USING(paper_id,version) "
                "JOIN papers p USING(paper_id) WHERE a.paper_id=? AND p.deleted=0", (paper_id,)
            ).fetchone()
            return _decode(PaperVersion, row[0]) if row else None

    def list_active_versions(self) -> tuple[PaperVersion, ...]:
        """One SQLite snapshot for query scope; deleted papers are excluded."""
        with self._connection() as db:
            rows = db.execute(
                "SELECT v.data FROM active_versions a JOIN versions v USING(paper_id,version) "
                "JOIN papers p USING(paper_id) WHERE p.deleted=0 ORDER BY a.paper_id"
            ).fetchall()
            return tuple(version for (raw,) in rows
                         if (version := _decode(PaperVersion, raw)).state is VersionState.READY)

    def _activate(self, db, paper_id, version):
        self._live(db, paper_id)
        target = self._version(db, paper_id, version)
        if target is None or target.state is not VersionState.READY:
            raise CatalogConflict("Only a verified ready version can be activated")
        db.execute(
            "INSERT INTO active_versions VALUES (?,?) ON CONFLICT(paper_id) "
            "DO UPDATE SET version=excluded.version", (paper_id, version)
        )

    def activate_version(self, paper_id: str, version: str) -> None:
        with self._connection(write=True) as db:
            self._activate(db, paper_id, version)

    def save_blocks(self, blocks: Sequence[SourceBlock]) -> None:
        """Save a complete ordered snapshot, not an incremental batch."""
        if not blocks:
            raise ValueError("Empty block snapshot")
        key = (blocks[0].paper_id, blocks[0].version)
        ids = {b.block_id for b in blocks}
        if len(ids) != len(blocks) or any((b.paper_id, b.version) != key for b in blocks):
            raise CatalogConflict("Blocks must belong to one version with unique IDs")
        for b in blocks:
            if any(ref is not None and ref not in ids for ref in (
                b.parent_block_id, b.previous_block_id, b.next_block_id
            )):
                raise CatalogConflict("Dangling block relation")
        refs = []
        for b in blocks:
            payload = b.to_json().encode("utf-8")
            digest = hashlib.sha256(payload).hexdigest()
            target = self.block_dir / (digest + ".json")
            fd, name = tempfile.mkstemp(dir=self.block_dir, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(name, target)
            finally:
                Path(name).unlink(missing_ok=True)
            refs.append((b.block_id, digest))
        with self._connection(write=True) as db:
            self._live(db, key[0])
            version = self._version(db, *key)
            existing = db.execute(
                "SELECT block_id,digest FROM blocks WHERE paper_id=? AND version=? ORDER BY ordinal", key
            ).fetchall()
            if existing == refs:
                return
            if existing or version is None or version.state is not VersionState.STAGING:
                raise CatalogConflict("Block snapshot is immutable or version missing")
            db.executemany("INSERT INTO blocks VALUES (?,?,?,?,?)", [
                (*key, i, block_id, digest) for i, (block_id, digest) in enumerate(refs)
            ])

    def get_blocks(self, paper_id: str, version: str) -> tuple[SourceBlock, ...]:
        with self._connection() as db:
            rows = db.execute(
                "SELECT digest FROM blocks WHERE paper_id=? AND version=? ORDER BY ordinal", (paper_id, version)
            ).fetchall()
        result = []
        for (digest,) in rows:
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise CatalogConflict("Invalid block file reference")
            raw = (self.block_dir / (digest + ".json")).read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise CatalogConflict("Block file checksum mismatch")
            result.append(_decode(SourceBlock, raw))
        return tuple(result)

    def _job(self, db, job_id):
        row = db.execute("SELECT data FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return _decode(IngestionJob, row[0]) if row else None

    def _put_job(self, db, job):
        db.execute("UPDATE jobs SET data=? WHERE job_id=?", (job.to_json(), job.job_id))

    def save_job(self, job: IngestionJob) -> None:
        """Create only. Use claim/advance/fail/retry for state transitions."""
        with self._connection(write=True) as db:
            self._live(db, job.paper_id)
            old = self._job(db, job.job_id)
            if old == job:
                return
            if old is not None:
                raise CatalogConflict("Job exists; use state transition methods")
            version = self._version(db, job.paper_id, job.target_version)
            if version is None or version.source_hash != job.source_hash or version.state is not VersionState.STAGING:
                raise CatalogConflict("Job source must match its staging version")
            if (job.status is not IngestionStatus.QUEUED or job.stage is not IngestionStage.WAITING
                    or job.attempts or job.lease_owner or job.lease_expires_at):
                raise CatalogConflict("New job must be unclaimed and queued")
            try:
                db.execute("INSERT INTO jobs VALUES (?,?,?,?)",
                           (job.job_id, job.paper_id, job.target_version, job.to_json()))
            except sqlite3.IntegrityError as exc:
                raise CatalogConflict("A job already exists for this version") from exc

    def get_job(self, job_id: str) -> IngestionJob | None:
        with self._connection() as db:
            return self._job(db, job_id)

    def list_unfinished_jobs(self) -> tuple[IngestionJob, ...]:
        with self._connection() as db:
            jobs = [_decode(IngestionJob, r[0]) for r in db.execute(
                "SELECT j.data FROM jobs j JOIN papers p USING(paper_id) WHERE p.deleted=0 ORDER BY j.rowid"
            )]
            return tuple(j for j in jobs if j.status in (IngestionStatus.QUEUED, IngestionStatus.RUNNING))

    def claim_job(self, worker_id: str, lease_seconds: int = 300, *, job_id: str | None = None) -> IngestionJob | None:
        if not worker_id.strip() or lease_seconds <= 0:
            raise ValueError("Worker and positive lease duration required")
        with self._connection(write=True) as db:
            for (raw,) in db.execute(
                "SELECT j.data FROM jobs j JOIN papers p USING(paper_id) WHERE p.deleted=0 ORDER BY j.rowid"
            ):
                job = _decode(IngestionJob, raw)
                if job.status is IngestionStatus.QUEUED and (job_id is None or job.job_id == job_id):
                    now = _now()
                    job = replace(job, status=IngestionStatus.RUNNING, stage=IngestionStage.PARSING,
                                  attempts=job.attempts + 1, lease_owner=worker_id,
                                  lease_expires_at=now + timedelta(seconds=lease_seconds), updated_at=now)
                    self._put_job(db, job)
                    return job
        return None

    def _owned(self, db, expected):
        self._live(db, expected.paper_id)
        if (self._job(db, expected.job_id) != expected or expected.status is not IngestionStatus.RUNNING
                or expected.lease_expires_at is None or expected.lease_expires_at <= _now()):
            raise CatalogConflict("Stale job snapshot or expired lease")

    def heartbeat(self, expected: IngestionJob, lease_seconds: int = 300) -> IngestionJob:
        if lease_seconds <= 0:
            raise ValueError("Positive lease duration required")
        with self._connection(write=True) as db:
            self._owned(db, expected)
            now = _now()
            job = replace(expected, updated_at=now, lease_expires_at=now + timedelta(seconds=lease_seconds))
            self._put_job(db, job)
            return job

    def advance_job(self, expected: IngestionJob, stage: IngestionStage) -> IngestionJob:
        stages = list(IngestionStage)
        if stage is IngestionStage.COMPLETED or stages.index(stage) != stages.index(expected.stage) + 1:
            raise CatalogConflict("Stages must advance one step; use complete_job to publish")
        with self._connection(write=True) as db:
            self._owned(db, expected)
            job = replace(expected, stage=stage, progress_current=0, progress_total=None, updated_at=_now())
            self._put_job(db, job)
            return job

    def update_progress(self, expected: IngestionJob, current: int, total: int | None) -> IngestionJob:
        with self._connection(write=True) as db:
            self._owned(db, expected)
            if current < expected.progress_current:
                raise CatalogConflict("Progress cannot move backwards within a stage")
            job = replace(expected, progress_current=current, progress_total=total, updated_at=_now())
            self._put_job(db, job)
            return job

    def fail_job(self, expected: IngestionJob, error: str) -> IngestionJob:
        with self._connection(write=True) as db:
            self._owned(db, expected)
            job = replace(expected, status=IngestionStatus.FAILED, error_message=error,
                          lease_owner=None, lease_expires_at=None, updated_at=_now())
            self._put_job(db, job)
            return job

    def cancel_job(self, job_id: str) -> IngestionJob:
        """Invalidate a worker snapshot atomically; already published jobs stay ready."""
        with self._connection(write=True) as db:
            job = self._job(db, job_id)
            if job is None or job.status is IngestionStatus.READY:
                raise CatalogConflict("Missing job or already published")
            job = replace(job, status=IngestionStatus.CANCELLED,
                          lease_owner=None, lease_expires_at=None, updated_at=_now())
            self._put_job(db, job)
            return job

    def retry_job(self, job_id: str) -> IngestionJob:
        with self._connection(write=True) as db:
            job = self._job(db, job_id)
            if job is None or job.status is not IngestionStatus.FAILED:
                raise CatalogConflict("Only failed jobs can be retried")
            self._live(db, job.paper_id)
            job = replace(job, status=IngestionStatus.QUEUED, stage=IngestionStage.WAITING,
                          error_message=None, progress_current=0, progress_total=None, updated_at=_now())
            self._put_job(db, job)
            return job

    def recover_expired_jobs(self) -> int:
        """Requeue abandoned jobs; live leases survive another process starting."""
        count = 0
        with self._connection(write=True) as db:
            now = _now()
            for (raw,) in db.execute("SELECT data FROM jobs").fetchall():
                job = _decode(IngestionJob, raw)
                if job.status is IngestionStatus.RUNNING and (job.lease_expires_at is None or job.lease_expires_at <= now):
                    self._put_job(db, replace(job, status=IngestionStatus.QUEUED,
                        stage=IngestionStage.WAITING, lease_owner=None, lease_expires_at=None,
                        progress_current=0, progress_total=None, updated_at=now))
                    count += 1
        return count

    def complete_job(self, expected: IngestionJob, *, indexed_chunk_count: int,
                     expected_chunk_count: int, index_searchable: bool) -> IngestionJob:
        """Caller verifies ES bulk/refresh; SQLite cannot verify remote ES.

        Mark ready, switch active version and finish the job in ONE transaction.
        """
        if (expected.stage is not IngestionStage.PUBLISHING or not index_searchable
                or expected_chunk_count <= 0 or indexed_chunk_count != expected_chunk_count):
            raise CatalogConflict("Publication requires complete searchable index")
        self.get_blocks(expected.paper_id, expected.target_version)
        with self._connection(write=True) as db:
            self._owned(db, expected)
            key = (expected.paper_id, expected.target_version)
            if not db.execute("SELECT 1 FROM blocks WHERE paper_id=? AND version=?", key).fetchone():
                raise CatalogConflict("Cannot publish without source blocks")
            version = self._version(db, *key)
            version = replace(version, state=VersionState.READY, chunk_count=indexed_chunk_count)
            db.execute("UPDATE versions SET data=? WHERE paper_id=? AND version=?", (version.to_json(), *key))
            self._activate(db, *key)
            job = replace(expected, status=IngestionStatus.READY, stage=IngestionStage.COMPLETED,
                          lease_owner=None, lease_expires_at=None, updated_at=_now())
            self._put_job(db, job)
            return job

    def soft_delete_paper(self, paper_id: str) -> None:
        with self._connection(write=True) as db:
            db.execute("UPDATE papers SET deleted=1 WHERE paper_id=?", (paper_id,))
            db.execute("DELETE FROM active_versions WHERE paper_id=?", (paper_id,))
            for (raw,) in db.execute("SELECT data FROM jobs WHERE paper_id=?", (paper_id,)).fetchall():
                job = _decode(IngestionJob, raw)
                if job.status in (IngestionStatus.RUNNING, IngestionStatus.QUEUED, IngestionStatus.FAILED):
                    self._put_job(db, replace(job, status=IngestionStatus.CANCELLED,
                        lease_owner=None, lease_expires_at=None, updated_at=_now()))

    def enqueue_import(
        self, source_uri: str, source_hash: str, *, parser_version: str,
        chunker_version: str, embedding_model: str | None = None,
        embedding_revision: str | None = None, title: str | None = None,
    ) -> IngestionJob:
        """Atomically create/reuse paper, version and job.

        source_hash is the caller's SHA-256 of immutable source bytes. Config
        revisions must include all output-affecting options, including tokenizer.
        Equal bytes at another path reuse the first source reference.
        """
        identity = json.dumps([source_hash, parser_version, chunker_version,
                               embedding_model, embedding_revision], ensure_ascii=False)
        paper_id = "paper-" + hashlib.sha256(source_hash.encode()).hexdigest()
        version_id = "version-" + hashlib.sha256(identity.encode()).hexdigest()
        job_id = "job-" + version_id.removeprefix("version-")
        paper = Paper(paper_id, source_uri, source_hash, title)
        version = PaperVersion(paper_id, version_id, source_hash, parser_version,
                               chunker_version, embedding_model, embedding_revision)
        with self._connection(write=True) as db:
            row = db.execute("SELECT data,deleted FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
            if row:
                if row[1]:
                    raise CatalogConflict("Paper is deleted; explicit restore is required")
                paper = _decode(Paper, row[0])
            else:
                db.execute("INSERT INTO papers(paper_id,data) VALUES (?,?)", (paper_id, paper.to_json()))
            old = self._job(db, job_id)
            if old:
                return old
            existing = self._version(db, paper_id, version_id)
            if existing and existing.state is not VersionState.STAGING:
                raise CatalogConflict("Version already published without this job")
            if not existing:
                db.execute("INSERT INTO versions VALUES (?,?,?)",
                           (paper_id, version_id, version.to_json()))
            job = IngestionJob(job_id, paper_id, version_id, paper.source_uri, source_hash)
            db.execute("INSERT INTO jobs VALUES (?,?,?,?)", (job_id, paper_id, version_id, job.to_json()))
            return job
