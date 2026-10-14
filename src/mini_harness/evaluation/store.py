"""Local durable evaluation queue. One worker holds an OS lock per store."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import time
import threading
from uuid import uuid4

from .dataset import canonical, fingerprint
from .models import EvaluationDataset


class EvaluationStore:
    def __init__(self, root=".mini-harness/evals"):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "evaluation.sqlite3"
        self._worker_thread = None
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS datasets (
                    digest TEXT PRIMARY KEY, snapshot TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, dataset_sha256 TEXT NOT NULL,
                    configuration TEXT NOT NULL, configuration_sha256 TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'live_rules', status TEXT NOT NULL,
                    verdict TEXT NOT NULL DEFAULT 'incomplete', reason TEXT,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS case_attempts (
                    attempt_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                    case_id TEXT NOT NULL, status TEXT NOT NULL,
                    session_id TEXT NOT NULL, artifact_dir TEXT NOT NULL,
                    report_ref TEXT, error_type TEXT,
                    started_at REAL NOT NULL, ended_at REAL);
                CREATE TABLE IF NOT EXISTS metric_results (
                    attempt_id TEXT NOT NULL, metric_id TEXT NOT NULL, target_id TEXT NOT NULL,
                    metric_version TEXT NOT NULL, data TEXT NOT NULL,
                    PRIMARY KEY(attempt_id,metric_id,target_id,metric_version));
                CREATE TABLE IF NOT EXISTS score_attempts (
                    score_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL,
                    configuration TEXT NOT NULL, status TEXT NOT NULL,
                    report_ref TEXT, error_type TEXT, started_at REAL NOT NULL, ended_at REAL);
                CREATE INDEX IF NOT EXISTS attempts_run ON case_attempts(run_id,case_id);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def worker_lock(self):
        # A persistent lock FILE is harmless: ownership is the live OS handle.
        handle = (self.root / "worker.lock").open("a+b")
        if handle.seek(0, 2) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        locked = False
        try:
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError as error:
                raise RuntimeError("An evaluation worker already owns this store") from error
            self._worker_thread = threading.get_ident()
            yield
        finally:
            if locked:
                self._worker_thread = None
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def enqueue(self, dataset, configuration=None, *, mode="live_rules"):
        if mode not in ("live_rules", "live_judge"):
            raise ValueError("Use rescore on an existing completed attempt")
        dataset = EvaluationDataset.model_validate(dataset.model_dump())
        snapshot = dataset.model_dump(mode="json")
        config = configuration or {}
        if mode == "live_judge":
            from .scoring import JudgeConfig
            config = {**config, "judge": JudgeConfig.model_validate(config.get("judge", {})).model_dump(mode="json")}
        digest = fingerprint(snapshot)
        identity = uuid4().hex
        now = time.time()
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO datasets VALUES (?,?)", (digest, canonical(snapshot)))
            db.execute("INSERT INTO runs(run_id,dataset_sha256,configuration,configuration_sha256,status,created_at,updated_at,mode) VALUES (?,?,?,?,?,?,?,?)",
                       (identity, digest, canonical(config), fingerprint(config), "queued", now, now, mode))
        return identity

    def get_run(self, run_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return dict(row)

    def dataset(self, run_id):
        run = self.get_run(run_id)
        with self.connection() as db:
            row = db.execute("SELECT snapshot FROM datasets WHERE digest=?", (run["dataset_sha256"],)).fetchone()
        return EvaluationDataset.model_validate_json(row[0])

    def attempts(self, run_id):
        with self.connection() as db:
            return [dict(row) for row in db.execute("SELECT * FROM case_attempts WHERE run_id=? ORDER BY started_at,rowid", (run_id,))]

    def cancel(self, run_id):
        self.get_run(run_id)
        with self.connection() as db:
            db.execute("UPDATE runs SET status=CASE WHEN status='queued' THEN 'interrupted' ELSE 'cancelling' END, reason='cancelled',updated_at=? WHERE run_id=? AND status IN ('queued','running')", (time.time(), run_id))

    def retry(self, run_id):
        with self.connection() as db:
            changed = db.execute("UPDATE runs SET status='queued',reason=NULL,updated_at=? WHERE run_id=? AND status IN ('interrupted','failed')", (time.time(), run_id)).rowcount
        if not changed:
            raise ValueError("Only interrupted/failed runs can be retried")

    def recover(self):
        """Caller MUST hold worker_lock; never interrupt another active worker."""
        self._require_worker()
        with self.connection() as db:
            db.execute("UPDATE score_attempts SET status='interrupted',error_type='WorkerInterrupted',ended_at=? WHERE status='running'", (time.time(),))
            db.execute("UPDATE case_attempts SET status='interrupted',error_type='WorkerInterrupted',ended_at=? WHERE status='running'", (time.time(),))
            db.execute("UPDATE runs SET status='interrupted',reason=COALESCE(reason,'worker_interrupted'),updated_at=? WHERE status IN ('running','cancelling')", (time.time(),))

    def _require_worker(self):
        if self._worker_thread != threading.get_ident():
            raise RuntimeError("Operation requires the live worker lock")

    def claim(self, run_id=None):
        self._require_worker()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT run_id FROM runs WHERE status='queued' AND (? IS NULL OR run_id=?) ORDER BY created_at,rowid LIMIT 1", (run_id, run_id)).fetchone()
            if row is None:
                return None
            db.execute("UPDATE runs SET status='running',updated_at=? WHERE run_id=?", (time.time(), row[0]))
            return row[0]

    def start_attempt(self, run_id, case_id):
        self._require_worker()
        identity, session_id = uuid4().hex, uuid4().hex
        relative = f"{run_id}/{case_id}/{identity}"
        with self.connection() as db:
            db.execute("INSERT INTO case_attempts(attempt_id,run_id,case_id,status,session_id,artifact_dir,started_at) VALUES (?,?,?,'running',?,?,?)",
                       (identity, run_id, case_id, session_id, relative, time.time()))
        return identity, session_id, self.root / relative

    def finish_attempt(self, attempt_id, status, report_ref, error_type=None):
        with self.connection() as db:
            db.execute("UPDATE case_attempts SET status=?,report_ref=?,error_type=?,ended_at=? WHERE attempt_id=? AND status='running'",
                       (status, canonical(report_ref), error_type, time.time(), attempt_id))

    def finish_run(self, run_id, status, reason=None):
        with self.connection() as db:
            if status in ("failed", "interrupted"):
                db.execute("UPDATE case_attempts SET status=?,error_type=COALESCE(error_type,?),ended_at=? WHERE run_id=? AND status='running'", (status, reason, time.time(), run_id))
            # Cancellation wins a race against normal completion.
            db.execute("UPDATE runs SET status=CASE WHEN status='cancelling' THEN 'interrupted' ELSE ? END,reason=COALESCE(reason,?),updated_at=? WHERE run_id=?", (status, reason, time.time(), run_id))

    def save_metrics(self, attempt_id, metrics):
        from .registry import validate_metric_result
        rows = [validate_metric_result(metric) for metric in metrics]
        with self.connection() as db:
            db.executemany("INSERT OR REPLACE INTO metric_results VALUES (?,?,?,?,?)", [
                (attempt_id, m.metric_id, m.target_id, m.metric_version, m.model_dump_json()) for m in rows])
            if any(m.passed is False for m in rows):
                db.execute("UPDATE runs SET verdict='failed' WHERE run_id=(SELECT run_id FROM case_attempts WHERE attempt_id=?)", (attempt_id,))

    def metrics(self, attempt_id):
        with self.connection() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT data FROM metric_results WHERE attempt_id=? ORDER BY metric_id,target_id", (attempt_id,))]

    def start_score(self, score_id, attempt_id, config):
        self._require_worker()
        with self.connection() as db:
            if db.execute("SELECT 1 FROM case_attempts WHERE attempt_id=?", (attempt_id,)).fetchone() is None:
                raise KeyError(attempt_id)
            db.execute("INSERT INTO score_attempts(score_id,attempt_id,configuration,status,started_at) VALUES (?,?,?,'running',?)",
                       (score_id, attempt_id, canonical(config), time.time()))

    def finish_score(self, score_id, status, ref, error_type=None):
        with self.connection() as db:
            db.execute("UPDATE score_attempts SET status=?,report_ref=?,error_type=?,ended_at=? WHERE score_id=?",
                       (status, canonical(ref), error_type, time.time(), score_id))

    def scores(self, attempt_id):
        with self.connection() as db:
            return [dict(row) for row in db.execute("SELECT * FROM score_attempts WHERE attempt_id=? ORDER BY started_at", (attempt_id,))]
