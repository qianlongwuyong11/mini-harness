"""OpenViking 0.5.1.1 的只读 HTTP 适配器（P1-07）。

每个实例绑定一个可信的本地 Scope 和远端身份。当前 dev 服务端以 ROOT 身份
运行，因此这里的范围校验是调用 mini-harness 适配器时的保护，不能代替服务鉴权。
"""

from __future__ import annotations

from math import isfinite
import re
from urllib.parse import urlsplit

import httpx

from .contracts import (
    MemoryBackendError,
    MemoryDocument,
    MemoryErrorCode,
    MemoryHit,
    MemoryScope,
    MemoryMessage, AppendResult, CommitResult, MemoryTask, TaskState,
)


_IDENTITY = re.compile(r"[A-Za-z0-9_-]+\Z")


class OpenVikingReadBackend:
    """一个实例对应一个本地 Scope；远端身份映射由可信配置提供。"""

    def __init__(
        self,
        *,
        base_url: str,
        scope: MemoryScope,
        account_id: str,
        remote_user_id: str,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment
        ):
            raise ValueError("base_url must be an HTTP(S) origin without credentials or path")
        if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("unencrypted HTTP is only allowed on loopback")
        if not _IDENTITY.fullmatch(account_id) or not _IDENTITY.fullmatch(remote_user_id):
            raise ValueError("remote account and user IDs must use ASCII letters, digits, '_' or '-'")
        if not isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number")

        self.scope = scope
        self.account_id = account_id
        self.remote_user_id = remote_user_id
        self._memories_uri = f"viking://user/{remote_user_id}/memories"
        self._client = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenVikingReadBackend:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _check_scope(self, scope: MemoryScope) -> None:
        if scope != self.scope:
            raise MemoryBackendError(MemoryErrorCode.PERMISSION_DENIED, "memory scope mismatch")

    def _check_uri(self, uri: str) -> None:
        # Reject ambiguous spellings instead of relying on URL normalization or a string prefix.
        if (
            not isinstance(uri, str) or not uri or "%" in uri or "\\" in uri
            or any(ord(char) < 32 or ord(char) == 127 for char in uri)
            or uri != uri.strip() or uri.endswith(("?", "#"))
        ):
            raise MemoryBackendError(MemoryErrorCode.PERMISSION_DENIED, "memory URI is outside scope")
        parts = urlsplit(uri)
        path = parts.path.split("/")
        if (
            parts.scheme != "viking" or parts.netloc != "user"
            or parts.query or parts.fragment
            or len(path) < 4 or path[:3] != ["", self.remote_user_id, "memories"]
            or any(segment in {"", ".", ".."} for segment in path[3:])
        ):
            raise MemoryBackendError(MemoryErrorCode.PERMISSION_DENIED, "memory URI is outside scope")

    def _request(self, method: str, path: str, **kwargs: object) -> object:
        headers = {
            "X-OpenViking-Account": self.account_id,
            "X-OpenViking-User": self.remote_user_id,
        }
        try:
            response = self._client.request(method, path, headers=headers, **kwargs)
            response.raise_for_status()
        except httpx.TimeoutException:
            raise MemoryBackendError(MemoryErrorCode.TIMEOUT, "memory service timed out") from None
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            code = (
                MemoryErrorCode.INVALID_INPUT if status in {400, 422} else
                MemoryErrorCode.PERMISSION_DENIED if status in {401, 403} else
                MemoryErrorCode.NOT_FOUND if status == 404 else
                MemoryErrorCode.UNAVAILABLE if status == 429 or status >= 500 else
                MemoryErrorCode.BACKEND_ERROR
            )
            raise MemoryBackendError(code, f"memory service returned HTTP {status}") from None
        except httpx.RequestError:
            raise MemoryBackendError(MemoryErrorCode.UNAVAILABLE, "memory service unavailable") from None
        try:
            payload = response.json()
        except ValueError:
            raise MemoryBackendError(MemoryErrorCode.BACKEND_ERROR, "invalid memory response") from None
        if not isinstance(payload, dict) or payload.get("status") != "ok" or "result" not in payload:
            raise MemoryBackendError(MemoryErrorCode.BACKEND_ERROR, "invalid memory response")
        return payload["result"]

    def recall(self, scope: MemoryScope, query: str, *, limit: int = 5) -> tuple[MemoryHit, ...]:
        self._check_scope(scope)
        if not isinstance(query, str) or not query.strip() or type(limit) is not int or limit < 1:
            raise MemoryBackendError(MemoryErrorCode.INVALID_INPUT, "nonempty query and positive limit required")
        result = self._request(
            "POST", "/api/v1/search/find",
            json={"query": query, "target_uri": self._memories_uri, "limit": limit},
        )
        if not isinstance(result, dict) or not isinstance(result.get("memories"), list):
            raise MemoryBackendError(MemoryErrorCode.BACKEND_ERROR, "invalid memory search result")
        hits: list[MemoryHit] = []
        for raw in result["memories"]:
            if not isinstance(raw, dict) or not isinstance(raw.get("abstract"), str):
                raise MemoryBackendError(MemoryErrorCode.BACKEND_ERROR, "invalid memory hit")
            uri = raw.get("uri")
            self._check_uri(uri)
            score = raw.get("score")
            level = raw.get("level")
            if score is not None and (type(score) not in {int, float} or not isfinite(score)):
                raise MemoryBackendError(MemoryErrorCode.BACKEND_ERROR, "invalid memory score")
            if level is not None and type(level) is not int:
                raise MemoryBackendError(MemoryErrorCode.BACKEND_ERROR, "invalid memory level")
            hits.append(MemoryHit(uri, raw["abstract"], float(score) if score is not None else None, level))
        return tuple(hits)

    def read(self, scope: MemoryScope, uri: str) -> MemoryDocument:
        self._check_scope(scope)
        self._check_uri(uri)
        result = self._request("GET", "/api/v1/content/read", params={"uri": uri})
        if not isinstance(result, str):
            raise MemoryBackendError(MemoryErrorCode.BACKEND_ERROR, "invalid memory document")
        return MemoryDocument(uri, result)


class OpenVikingBackend(OpenVikingReadBackend):
    """五方法后端；可从可信持久任务记录恢复本地会话/任务登记。"""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._sessions: set[str] = set()
        self._tasks: dict[str, str] = {}

    def restore_binding(self, scope: MemoryScope, session_id: str,
                        task_id: str | None = None) -> None:
        """Internal runtime hook; IDs must come from a trusted scoped job store.

        Does not query the server or reconcile ambiguous writes. Subsequent task
        reads still verify task type, resource ID and task ID in the response.
        """
        self._check_scope(scope)
        self._check_id(session_id)
        if task_id is not None:
            self._check_id(task_id)
            previous = self._tasks.get(task_id)
            if previous is not None and previous != session_id:
                raise MemoryBackendError(MemoryErrorCode.PERMISSION_DENIED,
                                         "task already bound to another session")
        self._sessions.add(session_id)
        if task_id is not None:
            self._tasks[task_id] = session_id

    @staticmethod
    def _check_id(value: str) -> None:
        if not isinstance(value, str) or not _IDENTITY.fullmatch(value):
            raise MemoryBackendError(MemoryErrorCode.INVALID_INPUT, "invalid session or task ID")

    def _write(self, path: str, body: dict) -> object:
        try:
            return self._request("POST", path, json=body)
        except MemoryBackendError as exc:
            if exc.code in {MemoryErrorCode.TIMEOUT, MemoryErrorCode.UNAVAILABLE,
                            MemoryErrorCode.BACKEND_ERROR}:
                raise MemoryBackendError(MemoryErrorCode.UNKNOWN_OUTCOME,
                                         "write outcome unknown; do not resend automatically") from None
            raise

    def append(self, scope: MemoryScope, session_id: str,
               messages: tuple[MemoryMessage, ...]) -> AppendResult:
        self._check_scope(scope)
        self._check_id(session_id)
        if not 1 <= len(messages) <= 100 or any(not isinstance(m, MemoryMessage) for m in messages):
            raise MemoryBackendError(MemoryErrorCode.INVALID_INPUT, "append requires 1 to 100 messages")
        result = self._write(f"/api/v1/sessions/{session_id}/messages/batch", {
            "messages": [{"role": m.role, "content": m.content,
                          "source_message_ids": [m.source_message_id]} for m in messages],
        })
        if (not isinstance(result, dict) or result.get("session_id") != session_id
                or type(result.get("added")) is not int or result["added"] != len(messages)):
            raise MemoryBackendError(MemoryErrorCode.UNKNOWN_OUTCOME, "append confirmation incomplete")
        self._sessions.add(session_id)
        return AppendResult(session_id, result["added"])

    def commit(self, scope: MemoryScope, session_id: str) -> CommitResult:
        self._check_scope(scope)
        self._check_id(session_id)
        if session_id not in self._sessions:
            raise MemoryBackendError(MemoryErrorCode.PERMISSION_DENIED, "session not registered in this scope")
        result = self._write(f"/api/v1/sessions/{session_id}/commit", {})
        if isinstance(result, dict) and result.get("session_id") == session_id:
            if result.get("status") == "skipped" and result.get("reason") == "no_messages":
                return CommitResult(session_id, "noop")
            task_id = result.get("task_id")
            if result.get("status") == "accepted" and isinstance(task_id, str) and _IDENTITY.fullmatch(task_id):
                self._tasks[task_id] = session_id
                return CommitResult(session_id, "accepted", task_id)
        raise MemoryBackendError(MemoryErrorCode.UNKNOWN_OUTCOME, "commit confirmation incomplete")

    def get_task(self, scope: MemoryScope, task_id: str) -> MemoryTask:
        self._check_scope(scope)
        self._check_id(task_id)
        if task_id not in self._tasks:
            raise MemoryBackendError(MemoryErrorCode.PERMISSION_DENIED, "task not registered in this scope")
        result = self._request("GET", f"/api/v1/tasks/{task_id}")
        session_id = self._tasks[task_id]
        if (not isinstance(result, dict) or result.get("task_id") != task_id
                or result.get("resource_id") != session_id or result.get("task_type") != "session_commit"):
            raise MemoryBackendError(MemoryErrorCode.BACKEND_ERROR, "invalid task identity")
        try:
            state = TaskState(result.get("status"))
        except ValueError:
            raise MemoryBackendError(MemoryErrorCode.BACKEND_ERROR, "unknown task state") from None
        count = None
        if state == TaskState.COMPLETED:
            details = result.get("result") or {}
            counts = details.get("memories_extracted", {}) if isinstance(details, dict) else {}
            if isinstance(counts, dict) and all(type(n) is int and n >= 0 for n in counts.values()):
                count = sum(counts.values())
        failed = state == TaskState.FAILED
        return MemoryTask(task_id, session_id, state, count,
                          MemoryErrorCode.BACKEND_ERROR if failed else None,
                          "remote memory processing failed" if failed else None)
