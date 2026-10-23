"""将原始事件快照的指定范围转换成消息；不修改日志、不进行网络或文件 IO。"""

from collections.abc import Sequence

from ..session import SessionEvent
from .contracts import MemoryMessage


def capture_messages(
    session_id: str,
    events: Sequence[SessionEvent],
    *,
    start: int = 0,
    end: int | None = None,
) -> tuple[MemoryMessage, ...]:
    """读取完整快照中的 [start, end) 事件 ID 范围（end 默认快照长度）。

    必须传原始 Session.events，不传 derive_messages/surface 投影。未知来源
    （包括旧日志）保守跳过。稳定来源 ID 用于追溯，不代表远端写入幂等。
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("A stable session_id is required")
    snapshot = tuple(events)
    stop = len(snapshot) if end is None else end
    if type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= len(snapshot):
        raise ValueError("Expected 0 <= start <= end <= snapshot length")
    if any(type(event.id) is not int or event.id != index for index, event in enumerate(snapshot)):
        raise ValueError("Expected a complete contiguous event snapshot starting at zero")
    messages = []
    for event in snapshot[start:stop]:
        data = event.data
        if "surface_op" in data:
            continue
        if event.type == "user/message" and data.get("source") == "user_input":
            role = "user"
        elif (event.type == "assistant/message" and data.get("source") == "assistant_final"
              and not data.get("tool_calls")):
            role = "assistant"
        else:
            continue
        content = data.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        messages.append(MemoryMessage(
            source_message_id=f"{session_id}:event:{event.id}", role=role, content=content,
        ))
    return tuple(messages)
