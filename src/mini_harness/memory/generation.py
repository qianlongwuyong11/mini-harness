"""Generate durable jobs from a saved JSONL snapshot; no remote operations."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from ..persistence import JsonlStore
from .contracts import MemoryScope
from .job_store import MemoryJobStore
from .policy import POLICY_VERSION, evaluate_turn


def _digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
                                    separators=(",", ":"),allow_nan=False).encode("utf-8")).hexdigest()


def _prefix(events, end):
    return _digest([dict(id=e.id,type=e.type,ts=e.ts,data=dict(e.data)) for e in events[:end]])


@dataclass(frozen=True)
class ScanResult:
    scan_key: str
    event_end: int
    job_ids: tuple[str, ...]
    skipped: tuple[tuple[int, int, str], ...]
    status: str
    reason: str


def scan_saved_session(path: str | Path, store: MemoryJobStore, *,
                       target_id: str, scope: MemoryScope) -> ScanResult:
    """Call after JsonlStore.save returns, or on startup for known saved files.

    Never accepts a live Session. Does not prove concurrent writer fsync, repair
    torn tails, or discover files. Owner must serialize save versus scan.
    """
    if not isinstance(target_id,str) or not target_id.strip():
        raise ValueError("Nonempty target required")
    session = JsonlStore(path).read_snapshot()
    events = session.events
    key = _digest([target_id,asdict(scope),session.session_id,POLICY_VERSION])
    cursor = store.get_scan_cursor(key)
    start = cursor[0] if cursor else 0
    if start > len(events) or (cursor and cursor[1] != _prefix(events,start)):
        raise ValueError("Saved history differs from the scanned prefix")
    jobs, skipped = [], []
    while start < len(events):
        if events[start].type != "turn/start":
            return ScanResult(key,start,tuple(jobs),tuple(skipped),"reject","missing_turn_start")
        end = next((i+1 for i in range(start+1,len(events)) if events[i].type == "turn/end"),len(events))
        decision = evaluate_turn(session.session_id,events,start=start,end=end,durable_end=len(events))
        if decision.status in ("wait","reject"):
            return ScanResult(key,start,tuple(jobs),tuple(skipped),decision.status,decision.reason)
        if decision.status == "submit":
            job = store.create(target_id=target_id,scope=scope,session_id=session.session_id,
                               start=start,end=end,policy_version=POLICY_VERSION,messages=decision.messages)
            jobs.append(job.job_id)
        else:
            skipped.append((start,end,decision.reason))
        store.advance_scan_cursor(key,expected_end=start,end=end,
                                  prefix_sha256=_prefix(events,end),reason=decision.reason)
        start = end
    return ScanResult(key,start,tuple(jobs),tuple(skipped),"complete","snapshot_scanned")
