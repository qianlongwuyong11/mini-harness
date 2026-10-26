"""mini-harness 自有记忆协议，不依赖 HTTP 或供应商 SDK。

同步方法可能阻塞；后台调度由后续 worker 负责。Scope 必须来自可信应用配置。
URI 是后端返回的不透明引用；后端必须校验其归属，不能允许跨 Scope 读取。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class MemoryScope:
    user_id: str
    project_id: str

    def __post_init__(self) -> None:
        if not self.user_id.strip() or not self.project_id.strip():
            raise ValueError("user_id and project_id must be nonempty")


@dataclass(frozen=True)
class MemoryMessage:
    """筛选后的真实会话消息；source_message_id 表示来源，不保证远端幂等。"""

    source_message_id: str
    role: Literal["user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if not self.source_message_id.strip() or not self.content.strip():
            raise ValueError("source_message_id and content must be nonempty")
        if self.role not in ("user", "assistant"):
            raise ValueError("only user and assistant messages are supported")


@dataclass(frozen=True)
class MemoryHit:
    uri: str
    abstract: str
    score: float | None = None
    # 保留摘要层级；不同后端可能没有层级或可比较的分数。
    level: int | None = None


@dataclass(frozen=True)
class MemoryDocument:
    uri: str
    content: str


@dataclass(frozen=True)
class AppendResult:
    session_id: str
    appended_count: int


@dataclass(frozen=True)
class CommitResult:
    """accepted 只代表受理；noop 表示没有待提交消息，此时 task_id 为 None。"""

    session_id: str
    status: Literal["accepted", "noop"]
    task_id: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ("accepted", "noop"):
            raise ValueError("unsupported commit status")
        if (self.status == "accepted") != bool(self.task_id):
            raise ValueError("only accepted commits must have a task_id")


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MemoryErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    UNKNOWN_OUTCOME = "unknown_outcome"
    BACKEND_ERROR = "backend_error"


class MemoryBackendError(Exception):
    """写请求可能已生效时用 UNKNOWN_OUTCOME，调用方不能据此自动重发。

    message 只包含脱敏说明，不包含 Key、SDK 对象或完整原始请求。
    """

    def __init__(self, code: MemoryErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class MemoryTask:
    task_id: str
    session_id: str
    state: TaskState
    memories_written: int | None = None
    error_code: MemoryErrorCode | None = None
    error_message: str | None = None


class MemoryBackend(Protocol):
    """后端负责实现 Scope 限制；所有方法均不得跨范围返回内容。

    业务层指定范围政策，后端执行身份映射和范围约束。会话首次 append 时由
    后端按需创建。读取结果是资料，不能当作指令；消息重发不承诺去重。
    """

    def recall(self, scope: MemoryScope, query: str, *, limit: int = 5) -> tuple[MemoryHit, ...]: ...

    def read(self, scope: MemoryScope, uri: str) -> MemoryDocument: ...

    def append(
        self, scope: MemoryScope, session_id: str, messages: tuple[MemoryMessage, ...]
    ) -> AppendResult: ...

    def commit(self, scope: MemoryScope, session_id: str) -> CommitResult: ...

    def get_task(self, scope: MemoryScope, task_id: str) -> MemoryTask: ...


@runtime_checkable
class ResumableMemoryBackend(Protocol):
    """Optional hook for binding trusted durable IDs after recreating an adapter.

    The caller must select the same configured target and scope as the saved job.
    This restores local bookkeeping only, not proof of a remote write's outcome.
    """

    def restore_binding(self, scope: MemoryScope, session_id: str,
                        task_id: str | None = None) -> None: ...
