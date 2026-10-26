"""Opt-in background memory execution; no Session/UI access or automatic mounting."""
import math
import threading
from uuid import uuid4

from .job_store import MemoryJobStore, StaleJobError
from .worker import MemoryWorker


class _HeartbeatWorker(MemoryWorker):
    def __init__(self, *args, db_path, heartbeat_interval, claim_lock, stop_event, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_path, self.heartbeat_interval = db_path, heartbeat_interval
        self.claim_lock,self.stop_event=claim_lock,stop_event

    def _claim(self):
        with self.claim_lock:
            if self.stop_event.is_set(): return None
            return super()._claim()

    def _invoke(self, job, operation, *args):
        stopped = threading.Event()
        errors = []
        def renew():
            try:
                with MemoryJobStore(self.db_path) as store:
                    while not stopped.wait(self.heartbeat_interval):
                        store.renew_inflight(job,lease_seconds=self.lease_seconds)
            except Exception:
                errors.append(True)
        thread = threading.Thread(target=renew,name="memory-heartbeat",daemon=True)
        thread.start()
        try:
            result = super()._invoke(job,operation,*args)
        finally:
            stopped.set()
            thread.join()
            if errors:
                # Remote outcome may be unknown; leave durable intent for recovery.
                raise StaleJobError("Heartbeat failed; discard late RPC result")
        return result


class MemoryBackgroundRunner:
    """One-shot lifecycle. backend_factory returns a context manager, entered and
    closed on the execution thread. Every SQLite connection stays on its creator
    thread. stop(timeout) returns False if a bounded remote call is still running;
    never forcibly close another thread's backend or database.
    """
    def __init__(self, db_path, backend_factory, *, target_id, scope,
                 poll_interval=1.0, lease_seconds=300, heartbeat_interval=30):
        for value in (poll_interval,lease_seconds,heartbeat_interval):
            MemoryJobStore._lease_duration(value)
        if heartbeat_interval >= lease_seconds/2:
            raise ValueError("Heartbeat interval must be less than half the lease")
        if not isinstance(target_id,str) or not target_id.strip():
            raise ValueError("Nonempty target required")
        self.db_path,self.backend_factory=db_path,backend_factory
        self.target_id,self.scope=target_id,scope
        self.poll_interval,self.lease_seconds=poll_interval,lease_seconds
        self.heartbeat_interval=heartbeat_interval
        self._stop=threading.Event(); self._wake=threading.Event()
        self._lock=threading.Lock(); self._thread=None
        self.last_error=None

    def start(self):
        with self._lock:
            if self._thread is not None or self._stop.is_set():
                raise RuntimeError("Runner can only be started once")
            self._thread=threading.Thread(target=self._run,name="memory-worker",daemon=True)
            self._thread.start()

    def wake(self):
        """Hint only: durable jobs are found even if notification is lost."""
        self._wake.set()

    def stop(self, timeout=5.0):
        if timeout is not None and (isinstance(timeout,bool) or not isinstance(timeout,(int,float))
                                    or not math.isfinite(timeout) or timeout < 0):
            raise ValueError("Nonnegative timeout required")
        with self._lock:
            self._stop.set(); self._wake.set(); thread=self._thread
        if thread is not None:
            thread.join(timeout)
            return not thread.is_alive()
        return True

    def _run(self):
        try:
            with MemoryJobStore(self.db_path) as store, self.backend_factory() as backend:
                worker=_HeartbeatWorker(store,backend,target_id=self.target_id,scope=self.scope,
                    worker_id=uuid4().hex,lease_seconds=self.lease_seconds,
                    db_path=self.db_path,heartbeat_interval=self.heartbeat_interval,claim_lock=self._lock,stop_event=self._stop)
                while not self._stop.is_set():
                    # Clear before scan: a notification during scan remains visible.
                    self._wake.clear()
                    try:
                        result=worker.run_once()
                        if result.outcome == "yielded":
                            try: store.defer_poll(result.job,seconds=self.poll_interval)
                            except StaleJobError: pass
                            self.last_error=None
                            continue
                        self.last_error=None
                    except Exception:
                        # Keep persisted intent; expired write leases become ambiguous.
                        self.last_error="memory_iteration_failed"
                    self._wake.wait(self.poll_interval)
        except Exception:
            self.last_error="memory_runtime_failed"
