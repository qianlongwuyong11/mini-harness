"""Pure capture/commit policy. No IO, queue, cursor mutation or remote deduplication."""
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Literal

from ..session import SessionEvent
from .capture import capture_messages
from .contracts import MemoryMessage

POLICY_VERSION = "completed-turn-v1"

@dataclass(frozen=True)
class CaptureDecision:
    status: Literal["submit", "wait", "skip", "reject"]
    reason: str
    messages: tuple[MemoryMessage, ...] = ()
    policy_version: str = POLICY_VERSION


def evaluate_turn(session_id: str, events: Sequence[SessionEvent], *,
                  start: int, end: int, durable_end: int) -> CaptureDecision:
    """Evaluate exactly one [start,end) turn in a full, zero-based snapshot.

    durable_end is the exclusive boundary confirmed by the persistence owner;
    this pure function cannot verify disk durability. Repeated evaluation is
    deterministic, not a reservation or an exactly-once guarantee.
    """
    snapshot = tuple(events)
    # Reuse source/range validation even for skipped or pending turns.
    messages = capture_messages(session_id, snapshot, start=start, end=end)
    if type(durable_end) is not int or not 0 <= durable_end <= len(snapshot):
        raise ValueError("Invalid durable boundary")
    selected = snapshot[start:end]
    if not selected or selected[0].type != "turn/start":
        return CaptureDecision("reject", "missing_turn_start")
    if any(e.type == "turn/start" for e in selected[1:]):
        return CaptureDecision("reject", "multiple_turns")
    endings = [i for i, e in enumerate(selected) if e.type == "turn/end"]
    if not endings:
        return CaptureDecision("wait", "unfinished_turn")
    if endings != [len(selected)-1]:
        return CaptureDecision("reject", "invalid_turn_end")
    if durable_end < end:
        return CaptureDecision("wait", "not_persisted")
    if selected[-1].data.get("reason") != "completed":
        return CaptureDecision("skip", "unsuccessful_turn")
    if not any(m.role == "user" for m in messages):
        return CaptureDecision("skip", "no_user_input")
    if len(messages) > 100:
        return CaptureDecision("reject", "message_limit")
    return CaptureDecision("submit", "completed_durable_turn", messages)
