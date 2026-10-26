"""SQLite memory outbox and lease fencing; no network or background worker."""
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

from .contracts import MemoryMessage, MemoryScope


class JobState(str, Enum):
    PENDING = "pending"
    APPENDING = "appending"
    APPENDED = "appended"
    COMMITTING = "committing"
    POLLING = "polling"
    COMPLETED = "completed"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"
    CANCELLED = "cancelled"


class BatchConflict(ValueError):
    """Same logical batch was supplied with a different immutable payload."""


class StaleJobError(RuntimeError):
    """Another writer already changed this record."""


@dataclass(frozen=True)
class MemoryJob:
    job_id: str
    batch_key: str
    target_id: str
    scope: MemoryScope
    session_id: str
    start: int
    end: int
    policy_version: str
    messages: tuple[MemoryMessage, ...]
    payload_sha256: str
    state: JobState
    revision: int
    remote_session_id: str | None
    remote_task_id: str | None
    error_code: str | None
    error_message: str | None
    created_at: float
    updated_at: float
    lease_owner: str | None
    lease_generation: int
    lease_expires_at: float | None
    retry_count: int
    next_run_at: float
    retry_state: str | None


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class MemoryJobStore:
    """One connection per instance. Caller owns close and thread confinement.

    target_id is a stable, non-secret identifier for the configured backend and
    identity mapping. Never put credentials in it. All SQL transactions are local.
    """
    def __init__(self, path: str | Path, *, clock=None):
        self._clock = clock if clock is not None else time.time
        self._db = sqlite3.connect(str(path), timeout=5)
        self._db.row_factory = sqlite3.Row
        try:
            self._db.execute("PRAGMA synchronous=FULL")
            states = ",".join("'" + state.value + "'" for state in JobState)
            with self._db:
                # Serialize migration as well as concurrent first opens.
                self._db.execute("BEGIN IMMEDIATE")
                version = self._db.execute("PRAGMA user_version").fetchone()[0]
                if version not in (0, 1, 2, 3, 4, 5):
                    raise ValueError("Unsupported memory job schema version")
                self._db.execute(f"""CREATE TABLE IF NOT EXISTS memory_jobs (
                    job_id TEXT PRIMARY KEY, batch_key TEXT NOT NULL UNIQUE,
                    target_id TEXT NOT NULL, user_id TEXT NOT NULL, project_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, event_start INTEGER NOT NULL CHECK(event_start>=0),
                    event_end INTEGER NOT NULL CHECK(event_end>event_start),
                    policy_version TEXT NOT NULL, payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN ({states})),
                    revision INTEGER NOT NULL DEFAULT 0 CHECK(revision>=0),
                    remote_session_id TEXT, remote_task_id TEXT, error_code TEXT, error_message TEXT,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL)""")
                self._db.execute("CREATE INDEX IF NOT EXISTS memory_jobs_state ON memory_jobs(state,created_at)")
                self._db.execute("""CREATE TABLE IF NOT EXISTS memory_scan_cursors (
                    scan_key TEXT PRIMARY KEY, event_end INTEGER NOT NULL CHECK(event_end>=0),
                    prefix_sha256 TEXT NOT NULL, last_reason TEXT NOT NULL, updated_at REAL NOT NULL)""")
                columns = {row[1] for row in self._db.execute("PRAGMA table_info(memory_jobs)")}
                for name, definition in (
                    ("lease_owner", "TEXT"),
                    ("lease_generation", "INTEGER NOT NULL DEFAULT 0 CHECK(lease_generation>=0)"),
                    ("lease_expires_at", "REAL"),
                    ("retry_count", "INTEGER NOT NULL DEFAULT 0"),
                    ("next_run_at", "REAL NOT NULL DEFAULT 0"),
                    ("retry_state", "TEXT"),
                ):
                    if name not in columns:
                        self._db.execute(f"ALTER TABLE memory_jobs ADD COLUMN {name} {definition}")
                self._db.execute("CREATE INDEX IF NOT EXISTS memory_jobs_lease ON memory_jobs(lease_expires_at)")
                self._db.execute("""CREATE TABLE IF NOT EXISTS memory_reconciliations (
                    job_id TEXT NOT NULL, revision INTEGER NOT NULL,
                    decision TEXT NOT NULL, evidence_ref TEXT NOT NULL,
                    actor TEXT NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY(job_id,revision))""")
                self._db.execute("""CREATE TABLE IF NOT EXISTS memory_retry_events (
                    job_id TEXT NOT NULL, revision INTEGER NOT NULL, kind TEXT NOT NULL,
                    error_code TEXT, actor TEXT, created_at REAL NOT NULL,
                    PRIMARY KEY(job_id,revision))""")
                self._db.execute("PRAGMA user_version=5")
        except BaseException:
            self._db.close()
            raise

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def _decode(row):
        if row is None:
            return None
        return MemoryJob(
            row["job_id"], row["batch_key"], row["target_id"],
            MemoryScope(row["user_id"], row["project_id"]), row["session_id"],
            row["event_start"], row["event_end"], row["policy_version"],
            tuple(MemoryMessage(**m) for m in json.loads(row["payload_json"])),
            row["payload_sha256"], JobState(row["state"]), row["revision"],
            row["remote_session_id"], row["remote_task_id"], row["error_code"],
            row["error_message"], row["created_at"], row["updated_at"],
            row["lease_owner"], row["lease_generation"], row["lease_expires_at"],
            row["retry_count"], row["next_run_at"], row["retry_state"])

    def get(self, job_id: str) -> MemoryJob | None:
        return self._decode(self._db.execute("SELECT * FROM memory_jobs WHERE job_id=?", (job_id,)).fetchone())

    def create(self, *, target_id: str, scope: MemoryScope, session_id: str,
               start: int, end: int, policy_version: str,
               messages: tuple[MemoryMessage, ...]) -> MemoryJob:
        for value in (target_id, session_id, policy_version):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Nonempty identity and policy required")
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError("Invalid event range")
        messages = tuple(messages)
        if not 1 <= len(messages) <= 100:
            raise ValueError("Expected 1 to 100 messages")
        ids = [m.source_message_id for m in messages]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate source message ID")
        payload = _json([asdict(m) for m in messages])
        key = _hash(_json([target_id, asdict(scope), session_id, start, end, policy_version]))
        now = time.time()
        with self._db:
            self._db.execute("""INSERT INTO memory_jobs
                (job_id,batch_key,target_id,user_id,project_id,session_id,event_start,event_end,
                 policy_version,payload_json,payload_sha256,state,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(batch_key) DO NOTHING""",
                (uuid4().hex,key,target_id,scope.user_id,scope.project_id,session_id,start,end,
                 policy_version,payload,_hash(payload),JobState.PENDING.value,now,now))
            row = self._db.execute("SELECT * FROM memory_jobs WHERE batch_key=?",(key,)).fetchone()
            if row["payload_json"] != payload:
                raise BatchConflict("Existing batch has a different payload")
        return self._decode(row)

    def save_progress(self, job_id: str, *, expected_revision: int, state: JobState,
                      remote_session_id: str | None = None, remote_task_id: str | None = None,
                      error_code: str | None = None, error_message: str | None = None,
                      lease: MemoryJob | None = None) -> MemoryJob:
        """Persist a snapshot; claimed jobs require a current, unexpired lease.

        The P3-02 unleased API remains available ONLY before the first claim
        (generation zero). It cannot bypass fencing, even after release/recovery.
        Errors default to cleared; omitted remote IDs retain their values.
        """
        state = JobState(state)
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("Invalid revision")
        if error_message is not None and len(error_message) > 2000:
            raise ValueError("Error message exceeds 2000 characters")
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            now = self._now()
            if lease is not None:
                if lease.job_id != job_id or lease.revision != expected_revision:
                    raise StaleJobError("Lease snapshot does not match progress revision")
                self._owned(lease, now)
            else:
                current = self.get(job_id)
                if (current is None or current.revision != expected_revision
                        or current.lease_generation != 0 or current.lease_owner is not None):
                    raise StaleJobError("Current lease required for claimed jobs")
            terminal = state in {JobState.COMPLETED, JobState.FAILED,
                                 JobState.CANCELLED, JobState.AMBIGUOUS}
            self._db.execute("""UPDATE memory_jobs SET state=?,revision=revision+1,
                remote_session_id=COALESCE(?,remote_session_id),remote_task_id=COALESCE(?,remote_task_id),
                error_code=?,error_message=?,updated_at=?,
                lease_owner=CASE WHEN ? THEN NULL ELSE lease_owner END,
                lease_expires_at=CASE WHEN ? THEN NULL ELSE lease_expires_at END
                WHERE job_id=? AND revision=?""",
                (state.value,remote_session_id,remote_task_id,error_code,error_message,now,
                 terminal,terminal,job_id,expected_revision))
            result = self.get(job_id)
        return result

    def _now(self):
        now = self._clock()
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ValueError("Finite clock value required")
        return now

    @staticmethod
    def _lease_duration(seconds):
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("Positive finite lease duration required")

    def _owned(self, expected: MemoryJob, now: float):
        current = self.get(expected.job_id)
        if (current is None or not expected.lease_owner
                or current.lease_owner != expected.lease_owner
                or current.lease_generation != expected.lease_generation
                or current.revision != expected.revision
                or current.lease_expires_at is None or current.lease_expires_at <= now):
            raise StaleJobError("Stale snapshot or expired lease")
        return current

    def claim_next(self, worker_id: str, *, target_id: str, scope: MemoryScope,
                   lease_seconds: float = 300) -> MemoryJob | None:
        """Atomically claim one unowned safe-stage job for this configured scope.

        Run recover_expired explicitly before claiming expired work. Merely
        opening a store never invalidates another process's live lease.
        """
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("Nonempty worker ID required")
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError("Nonempty target required")
        self._lease_duration(lease_seconds)
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            now = self._now()
            deadline = now + lease_seconds
            if not math.isfinite(deadline):
                raise ValueError("Lease deadline overflow")
            row = self._db.execute("""SELECT job_id FROM memory_jobs
                WHERE target_id=? AND user_id=? AND project_id=? AND lease_owner IS NULL
                  AND state IN ('pending','appended','polling') AND next_run_at<=?
                ORDER BY created_at,rowid LIMIT 1""",
                (target_id,scope.user_id,scope.project_id,now)).fetchone()
            if row is None:
                return None
            self._db.execute("""UPDATE memory_jobs SET lease_owner=?,
                lease_generation=lease_generation+1,lease_expires_at=?,
                revision=revision+1,updated_at=? WHERE job_id=?""",
                (worker_id,deadline,now,row["job_id"]))
            result = self.get(row["job_id"])
        return result

    def heartbeat(self, expected: MemoryJob, *, lease_seconds: float = 300) -> MemoryJob:
        """Renew a live lease; callers must use the returned revision afterwards."""
        self._lease_duration(lease_seconds)
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            now = self._now()
            current = self._owned(expected,now)
            deadline = max(current.lease_expires_at, now + lease_seconds)
            if not math.isfinite(deadline):
                raise ValueError("Lease deadline overflow")
            self._db.execute("""UPDATE memory_jobs SET lease_expires_at=?,
                revision=revision+1,updated_at=? WHERE job_id=?""",
                (deadline,now,expected.job_id))
            result = self.get(expected.job_id)
        return result

    def _clear_lease(self, job: MemoryJob, now: float):
        uncertain = job.state in {JobState.APPENDING, JobState.COMMITTING}
        self._db.execute("""UPDATE memory_jobs SET lease_owner=NULL,lease_expires_at=NULL,
            lease_generation=lease_generation+1,revision=revision+1,updated_at=?,
            state=?,error_code=?,error_message=? WHERE job_id=?""",
            (now,JobState.AMBIGUOUS.value if uncertain else job.state.value,
             "unknown_outcome" if uncertain else job.error_code,
             "Lease ended during a remote write; reconcile before retry" if uncertain else job.error_message,
             job.job_id))

    def release(self, expected: MemoryJob) -> MemoryJob:
        """Relinquish ownership; an unfinished remote write becomes ambiguous."""
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            now = self._now()
            current = self._owned(expected,now)
            self._clear_lease(current,now)
            result = self.get(expected.job_id)
        return result

    def recover_expired(self, *, target_id: str, scope: MemoryScope) -> tuple[str, ...]:
        """Clear expired owners only, preserving progress and remote identifiers.

        Safe stages can be claimed again; appending/committing are quarantined
        as ambiguous. This does not cancel an outstanding remote request.
        """
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            now = self._now()
            rows = self._db.execute("""SELECT * FROM memory_jobs WHERE target_id=?
                AND user_id=? AND project_id=? AND lease_owner IS NOT NULL
                AND lease_expires_at<=? ORDER BY created_at,rowid""",
                (target_id,scope.user_id,scope.project_id,now)).fetchall()
            for row in rows:
                self._clear_lease(self._decode(row),now)
        return tuple(row["job_id"] for row in rows)

    def get_scan_cursor(self, scan_key: str):
        """Return (exclusive event boundary, prefix hash, last disposition)."""
        row = self._db.execute("SELECT event_end,prefix_sha256,last_reason FROM memory_scan_cursors WHERE scan_key=?",
                               (scan_key,)).fetchone()
        return tuple(row) if row else None

    def advance_scan_cursor(self, scan_key: str, *, expected_end: int,
                            end: int, prefix_sha256: str, reason: str):
        """Advance only after a job is durable or the policy explicitly skips.

        Separate from create deliberately: a crash in between reuses the unique
        batch on replay. This method does not reserve work or establish a lease.
        """
        if type(expected_end) is not int or type(end) is not int or not 0 <= expected_end < end:
            raise ValueError("Cursor must advance")
        with self._db:
            self._db.execute("INSERT INTO memory_scan_cursors VALUES (?,0,'','initial',?) ON CONFLICT(scan_key) DO NOTHING",
                             (scan_key,time.time()))
            result = self._db.execute("""UPDATE memory_scan_cursors
                SET event_end=?,prefix_sha256=?,last_reason=?,updated_at=?
                WHERE scan_key=? AND event_end=?""",
                (end,prefix_sha256,reason,time.time(),scan_key,expected_end))
            if result.rowcount != 1:
                raise StaleJobError("Scan cursor moved; reread the saved snapshot")

    def reconcile(self, expected: MemoryJob, *, target_id: str, scope: MemoryScope,
                  decision: str, evidence_ref: str, actor: str,
                  remote_task_id: str | None = None) -> MemoryJob:
        """Trusted operator resolution, NOT a model tool or automatic retry.

        Caller must stop/fence old remote writers and verify the EXACT batch.
        appended: all messages confirmed pending, no commit in flight.
        committed: archive/task association verified for this batch.
        unresolved: insufficient evidence; keep quarantined. cancel: abandon.
        No reset-to-pending option: absence from current messages is not proof.
        evidence_ref is a non-secret reference to independently reviewed evidence.
        """
        states = {"appended": JobState.APPENDED, "committed": JobState.POLLING,
                  "unresolved": JobState.AMBIGUOUS, "cancel": JobState.CANCELLED}
        if decision not in states:
            raise ValueError("Unsupported reconciliation decision")
        for value in (evidence_ref, actor):
            if not isinstance(value, str) or not value.strip() or len(value) > 500:
                raise ValueError("Nonempty bounded evidence reference and actor required")
        if decision == "committed":
            if not isinstance(remote_task_id, str) or not remote_task_id.strip():
                raise ValueError("Verified task ID required")
        elif remote_task_id is not None:
            raise ValueError("Task ID only allowed for committed resolution")
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            current = self.get(expected.job_id)
            if (current is None or current.revision != expected.revision
                    or current.target_id != target_id or current.scope != scope
                    or current.state != JobState.AMBIGUOUS or current.lease_owner is not None):
                raise StaleJobError("Only current unowned ambiguous job in configured scope can be reconciled")
            if decision in {"appended", "committed"} and not current.remote_session_id:
                raise ValueError("Saved remote session required")
            if decision == "appended" and current.remote_task_id:
                raise ValueError("Known task must be reconciled as committed")
            if current.remote_task_id and remote_task_id and current.remote_task_id != remote_task_id:
                raise ValueError("Cannot replace known remote task")
            now = self._now()
            self._db.execute("""UPDATE memory_jobs SET state=?,revision=revision+1,
                lease_generation=lease_generation+1,lease_expires_at=NULL,
                remote_task_id=COALESCE(?,remote_task_id),updated_at=?,
                error_code=?,error_message=? WHERE job_id=?""",
                (states[decision].value, remote_task_id, now,
                 current.error_code if decision == "unresolved" else None,
                 current.error_message if decision == "unresolved" else None, current.job_id))
            result = self.get(current.job_id)
            self._db.execute("INSERT INTO memory_reconciliations VALUES (?,?,?,?,?,?)",
                (current.job_id,result.revision,decision,evidence_ref,actor,now))
        return result

    def reconciliation_history(self, job_id: str) -> tuple[dict, ...]:
        """Local audit trail; no raw remote bodies or credentials are stored."""
        return tuple(dict(row) for row in self._db.execute(
            "SELECT * FROM memory_reconciliations WHERE job_id=? ORDER BY revision", (job_id,)))

    def record_read_failure(self, expected: MemoryJob, *, error_code: str,
                            max_retries: int = 3, base_delay: float = 2,
                            max_delay: float = 60) -> MemoryJob:
        """Only a failed read of an already-known remote task is safe to retry.

        retry_count counts scheduled automatic retries over this job's lifetime.
        A running response does not reset it. Remote task failure is not a read error.
        """
        if type(max_retries) is not int or not 0 <= max_retries <= 100:
            raise ValueError("max_retries must be between 0 and 100")
        self._lease_duration(base_delay)
        self._lease_duration(max_delay)
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            now = self._now()
            current = self._owned(expected,now)
            if current.state != JobState.POLLING or not current.remote_task_id:
                raise ValueError("Only known-task polling failures may use this path")
            retry = error_code in {"timeout", "unavailable"} and current.retry_count < max_retries
            delay = min(max_delay, base_delay * (2 ** min(current.retry_count,100))) if retry else 0
            if not math.isfinite(now + delay):
                raise ValueError("Retry deadline overflow")
            self._db.execute("""UPDATE memory_jobs SET state=?,retry_count=retry_count+?,
                next_run_at=?,retry_state='polling',error_code=?,error_message=?,
                revision=revision+1,lease_generation=lease_generation+1,
                lease_owner=NULL,lease_expires_at=NULL,updated_at=? WHERE job_id=?""",
                ('polling' if retry else 'failed',int(retry),now+delay,error_code,
                 'Remote task read failed; retry scheduled' if retry else
                 'Remote task read failed; inspect before explicit retry',now,current.job_id))
            result = self.get(current.job_id)
            self._db.execute("INSERT INTO memory_retry_events VALUES (?,?,?,?,?,?)",
                (current.job_id,result.revision,'scheduled' if retry else 'failed',error_code,None,now))
        return result

    def retry_failed(self, expected: MemoryJob, *, target_id: str, scope: MemoryScope,
                     actor: str) -> MemoryJob:
        """Explicit one-more-read after operator fixes cause; never resend writes.

        Does not reset lifetime automatic retry budget. No ambiguous/remote failed
        processing restart: those require reconciliation or a separate reviewed action.
        """
        if not isinstance(actor,str) or not actor.strip() or len(actor)>500:
            raise ValueError("Bounded nonempty actor required")
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            current = self.get(expected.job_id)
            if (current is None or current.revision != expected.revision
                    or current.target_id != target_id or current.scope != scope
                    or current.state != JobState.FAILED or current.lease_owner is not None):
                raise StaleJobError("Current unowned failed job in configured scope required")
            if current.retry_state != 'polling' or not current.remote_task_id or current.error_code == 'remote_task_failed':
                raise ValueError("No proven safe read retry for this failure")
            now = self._now()
            self._db.execute("""UPDATE memory_jobs SET state='polling',next_run_at=?,
                revision=revision+1,lease_generation=lease_generation+1,updated_at=?
                WHERE job_id=?""",(now,now,current.job_id))
            result = self.get(current.job_id)
            self._db.execute("INSERT INTO memory_retry_events VALUES (?,?,?,?,?,?)",
                (current.job_id,result.revision,'manual',current.error_code,actor,now))
        return result

    def retry_history(self, job_id: str) -> tuple[dict, ...]:
        return tuple(dict(row) for row in self._db.execute(
            "SELECT * FROM memory_retry_events WHERE job_id=? ORDER BY revision",(job_id,)))
