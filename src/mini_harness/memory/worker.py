"""One leased job per synchronous invocation; no sleep or background thread."""
from dataclasses import dataclass
from typing import Literal

from .contracts import (
    AppendResult, CommitResult, MemoryBackend, MemoryBackendError,
    MemoryErrorCode, MemoryScope, MemoryTask, ResumableMemoryBackend, TaskState,
)
from .job_store import JobState, MemoryJob, MemoryJobStore, StaleJobError


@dataclass(frozen=True)
class WorkerResult:
    outcome: Literal["idle", "yielded", "completed", "failed", "cancelled", "ambiguous", "lease_lost"]
    job: MemoryJob | None


class _RemoteFailure(Exception):
    def __init__(self, code: MemoryErrorCode):
        self.code = code


def _call(operation, *args):
    # Only the backend invocation is caught here. Local persistence errors must
    # propagate, keeping the last durable intent for later lease recovery.
    try:
        return operation(*args)
    except MemoryBackendError as error:
        raise _RemoteFailure(error.code) from None
    except Exception:
        # Do not persist arbitrary SDK exception text, URLs, credentials or body.
        raise _RemoteFailure(MemoryErrorCode.BACKEND_ERROR) from None


class MemoryWorker:
    """Caller owns store/backend lifetime and trusted target-to-scope mapping.

    Backend requests need their own finite timeout. Lease fencing rejects late
    results; this synchronous worker cannot interrupt a blocked backend call.
    """
    def __init__(self, store: MemoryJobStore, backend: MemoryBackend, *,
                 target_id: str, scope: MemoryScope, worker_id: str,
                 lease_seconds: float = 300):
        if not isinstance(target_id,str) or not target_id.strip():
            raise ValueError("Nonempty target required")
        if not isinstance(worker_id,str) or not worker_id.strip():
            raise ValueError("Nonempty worker required")
        self.store, self.backend = store, backend
        self.target_id, self.scope, self.worker_id = target_id, scope, worker_id
        # Same validation as the store, before recovery can mutate any records.
        MemoryJobStore._lease_duration(lease_seconds)
        self.lease_seconds = lease_seconds

    def _save(self, job, state, **fields):
        return self.store.save_progress(job.job_id, expected_revision=job.revision,
                                        lease=job, state=state, **fields)

    def run_once(self) -> WorkerResult:
        self.store.recover_expired(target_id=self.target_id,scope=self.scope)
        job = self.store.claim_next(self.worker_id,target_id=self.target_id,
                                   scope=self.scope,lease_seconds=self.lease_seconds)
        if job is None:
            return WorkerResult("idle",None)
        stage = "restore"
        try:
            try:
                if job.state in {JobState.APPENDED,JobState.POLLING}:
                    if not job.remote_session_id or (job.state == JobState.POLLING and not job.remote_task_id):
                        raise _RemoteFailure(MemoryErrorCode.INVALID_INPUT)
                    if isinstance(self.backend,ResumableMemoryBackend):
                        _call(self.backend.restore_binding,self.scope,
                              job.remote_session_id,job.remote_task_id)

                if job.state == JobState.PENDING:
                    stage = "append"
                    session_id = job.remote_session_id or job.session_id
                    job = self._save(job,JobState.APPENDING,remote_session_id=session_id)
                    receipt = _call(self.backend.append,self.scope,session_id,job.messages)
                    if (not isinstance(receipt,AppendResult) or receipt.session_id != session_id
                            or type(receipt.appended_count) is not int
                            or receipt.appended_count != len(job.messages)):
                        raise _RemoteFailure(MemoryErrorCode.UNKNOWN_OUTCOME)
                    job = self._save(job,JobState.APPENDED)

                if job.state == JobState.APPENDED:
                    stage = "commit"
                    job = self.store.heartbeat(job,lease_seconds=self.lease_seconds)
                    job = self._save(job,JobState.COMMITTING)
                    receipt = _call(self.backend.commit,self.scope,job.remote_session_id)
                    if (not isinstance(receipt,CommitResult)
                            or receipt.session_id != job.remote_session_id
                            or receipt.status != "accepted"
                            or not isinstance(receipt.task_id,str) or not receipt.task_id.strip()):
                        # noop after confirmed append does not prove this batch
                        # was processed; leave it for explicit reconciliation.
                        raise _RemoteFailure(MemoryErrorCode.UNKNOWN_OUTCOME)
                    job = self._save(job,JobState.POLLING,remote_task_id=receipt.task_id)

                stage = "poll"
                job = self.store.heartbeat(job,lease_seconds=self.lease_seconds)
                task = _call(self.backend.get_task,self.scope,job.remote_task_id)
                if (not isinstance(task,MemoryTask) or task.task_id != job.remote_task_id
                        or task.session_id != job.remote_session_id
                        or not isinstance(task.state,TaskState)):
                    raise _RemoteFailure(MemoryErrorCode.BACKEND_ERROR)
                if task.state in {TaskState.PENDING,TaskState.RUNNING}:
                    return WorkerResult("yielded",self.store.release(job))
                if task.state == TaskState.COMPLETED:
                    return WorkerResult("completed",self._save(job,JobState.COMPLETED))
                if task.state == TaskState.CANCELLED:
                    return WorkerResult("cancelled",self._save(job,JobState.CANCELLED))
                return WorkerResult("failed",self._save(job,JobState.FAILED,
                    error_code="remote_task_failed",error_message="Remote memory processing failed"))
            except _RemoteFailure as error:
                write_uncertain = stage in {"append","commit"} and error.code not in {
                    MemoryErrorCode.INVALID_INPUT,MemoryErrorCode.PERMISSION_DENIED,MemoryErrorCode.NOT_FOUND}
                state = JobState.AMBIGUOUS if write_uncertain else JobState.FAILED
                job = self._save(job,state,
                    error_code=MemoryErrorCode.UNKNOWN_OUTCOME.value if write_uncertain else error.code.value,
                    error_message=f"Memory {stage} failed; " +
                        ("verify remote outcome before retry" if write_uncertain else "inspect before retry"))
                return WorkerResult("ambiguous" if write_uncertain else "failed",job)
        except StaleJobError:
            # Do not release, mark failed, or overwrite results for a new owner.
            return WorkerResult("lease_lost",self.store.get(job.job_id))
