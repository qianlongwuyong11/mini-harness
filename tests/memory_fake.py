"""确定性测试夹具：不做模型抽取、向量检索、持久化或线程调度。"""

from dataclasses import replace

from mini_harness.memory.contracts import (
    AppendResult, CommitResult, MemoryBackendError, MemoryDocument, MemoryErrorCode,
    MemoryHit, MemoryMessage, MemoryScope, MemoryTask, TaskState,
)


class FakeMemoryBackend:
    def __init__(self) -> None:
        self._documents: dict[tuple[MemoryScope, str], MemoryDocument] = {}
        self._pending: dict[tuple[MemoryScope, str], list[MemoryMessage]] = {}
        self._tasks: dict[tuple[MemoryScope, str], MemoryTask] = {}
        self._batches: dict[tuple[MemoryScope, str], tuple[MemoryMessage, ...]] = {}
        self._sequence = 0

    def seed(self, scope: MemoryScope, document: MemoryDocument) -> None:
        """测试专用：显式预置召回内容。"""
        self._documents[scope, document.uri] = document

    def recall(self, scope: MemoryScope, query: str, *, limit: int = 5) -> tuple[MemoryHit, ...]:
        if limit < 1 or not query.strip():
            raise MemoryBackendError(MemoryErrorCode.INVALID_INPUT, "query and positive limit required")
        return tuple(
            MemoryHit(doc.uri, doc.content, level=2)
            for (owner, _), doc in self._documents.items()
            if owner == scope and query.casefold() in doc.content.casefold()
        )[:limit]

    def read(self, scope: MemoryScope, uri: str) -> MemoryDocument:
        try:
            return self._documents[scope, uri]
        except KeyError:
            raise MemoryBackendError(MemoryErrorCode.NOT_FOUND, "memory not found in scope") from None

    def append(self, scope: MemoryScope, session_id: str,
               messages: tuple[MemoryMessage, ...]) -> AppendResult:
        if not session_id.strip() or not messages:
            raise MemoryBackendError(MemoryErrorCode.INVALID_INPUT, "session and messages required")
        self._pending.setdefault((scope, session_id), []).extend(messages)
        return AppendResult(session_id, len(messages))

    def commit(self, scope: MemoryScope, session_id: str) -> CommitResult:
        if (scope, session_id) not in self._pending:
            raise MemoryBackendError(MemoryErrorCode.NOT_FOUND, "session not found in scope")
        messages = self._pending[scope, session_id]
        if not messages:
            return CommitResult(session_id, "noop")
        self._sequence += 1
        task_id = f"fake-task-{self._sequence}"
        self._tasks[scope, task_id] = MemoryTask(task_id, session_id, TaskState.PENDING)
        self._batches[scope, task_id] = tuple(messages)
        messages.clear()
        return CommitResult(session_id, "accepted", task_id)

    def get_task(self, scope: MemoryScope, task_id: str) -> MemoryTask:
        try:
            return self._tasks[scope, task_id]
        except KeyError:
            raise MemoryBackendError(MemoryErrorCode.NOT_FOUND, "task not found in scope") from None

    def complete(self, scope: MemoryScope, task_id: str,
                 documents: tuple[MemoryDocument, ...] = ()) -> None:
        """测试专用：手动发布指定记忆，模拟后台完成。"""
        task = self.get_task(scope, task_id)
        if task.state != TaskState.PENDING:
            raise ValueError("task already finished")
        for document in documents:
            self.seed(scope, document)
        self._tasks[scope, task_id] = replace(
            task, state=TaskState.COMPLETED, memories_written=len(documents))
