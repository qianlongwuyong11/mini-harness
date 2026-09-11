"""知识库收件箱监听插件：稳定文件自动导入、归档并审计。"""

from __future__ import annotations

import shutil
import sys
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, RLock, Thread, current_thread
from typing import Any

from .cordis import Context, depends
from .document_import import DocumentImportService, SUPPORTED_SUFFIXES
from .knowledge_base import AUDIT_RULES
from .sandbox import SandboxPolicy
from .settings import SettingsScope

TEMP_SUFFIXES = frozenset({".crdownload", ".part", ".tmp"})


@dataclass(frozen=True)
class WatchRecord:
    timestamp: str
    status: str
    source: str
    stored_source: str | None = None
    destination: str | None = None
    audit_summary: Mapping[str, int] | None = None
    error: str | None = None


class KnowledgeWatchService:
    """用轮询实现零额外依赖的可停止目录监听器。"""

    def __init__(
        self,
        *,
        workspace: Path,
        importer: DocumentImportService,
        sandbox: SandboxPolicy,
        watch_dir: Path,
        archive_dir: Path,
        failed_dir: Path,
        interval_seconds: float,
        stable_scans: int,
        reporter: Callable[[WatchRecord], None] | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.importer = importer
        self.sandbox = sandbox
        self.watch_dir = self._workspace_path(watch_dir, "kb_watch_dir")
        self.archive_dir = self._workspace_path(archive_dir, "kb_archive_dir")
        self.failed_dir = self._workspace_path(failed_dir, "kb_failed_dir")
        self.interval_seconds = interval_seconds
        self.stable_scans = stable_scans
        self.reporter = reporter or _print_record
        self._validate_layout()
        self._observed: dict[Path, tuple[int, int, int]] = {}
        self._records: deque[WatchRecord] = deque(maxlen=100)
        self._stop = Event()
        self._thread: Thread | None = None
        self._lock = RLock()
        self._scan_lock = Lock()

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def records(self) -> tuple[WatchRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def start(self) -> Callable[[], None]:
        with self._lock:
            if self.running:
                return self.stop
            for directory in (
                self.watch_dir,
                self.importer.knowledge_base.root,
                self.archive_dir,
                self.failed_dir,
            ):
                self.sandbox.fence_write(directory).mkdir(
                    parents=True, exist_ok=True
                )
            self._stop.clear()
            self._thread = Thread(
                target=self._run,
                daemon=True,
                name="mini-harness-kb-watch",
            )
            self._thread.start()
        return self.stop

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            self._stop.set()
        if thread is not None and thread is not current_thread():
            thread.join()
        with self._lock:
            if self._thread is thread:
                self._thread = None

    def scan_once(self) -> tuple[WatchRecord, ...]:
        """执行一次扫描；公开此方法便于确定性测试和手动诊断。"""
        with self._scan_lock:
            return self._scan_once()

    def _scan_once(self) -> tuple[WatchRecord, ...]:
        before = len(self.records)
        current_paths: set[Path] = set()
        for source in self._candidates():
            current_paths.add(source)
            try:
                stat = source.stat()
            except FileNotFoundError:
                continue
            previous = self._observed.get(source)
            stable = (
                previous[2] + 1
                if previous is not None
                and previous[:2] == (stat.st_size, stat.st_mtime_ns)
                else 1
            )
            self._observed[source] = (stat.st_size, stat.st_mtime_ns, stable)
            if stable < self.stable_scans:
                continue
            self._observed.pop(source, None)
            self._process(source)
        for missing in set(self._observed) - current_paths:
            self._observed.pop(missing, None)
        return self.records[before:]

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.scan_once()
            except Exception as error:  # noqa: BLE001 - 监听循环必须保持存活
                self._record(
                    WatchRecord(
                        timestamp=_timestamp(),
                        status="watch_error",
                        source=self.watch_dir.relative_to(self.workspace).as_posix(),
                        error=f"{type(error).__name__}: {error}",
                    )
                )
            self._stop.wait(self.interval_seconds)

    def _candidates(self) -> list[Path]:
        if not self.watch_dir.exists():
            return []
        candidates: list[Path] = []
        for path in self.watch_dir.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(self.watch_dir)
            except ValueError:
                continue
            if _is_candidate(relative):
                candidates.append(resolved)
        return sorted(candidates, key=lambda path: path.as_posix().casefold())

    def _process(self, source: Path) -> None:
        relative = source.relative_to(self.watch_dir)
        original = source.relative_to(self.workspace).as_posix()
        archived = self._unique_target(self.archive_dir / relative)
        try:
            self._move(source, archived)
            destination = relative.with_suffix(".md").as_posix()
            imported = self.importer.import_document(
                source_path=str(archived),
                destination=destination,
            )
            report = self.importer.knowledge_base.audit(
                sorted(AUDIT_RULES - {"stale"})
            )
            self._record(
                WatchRecord(
                    timestamp=_timestamp(),
                    status="imported",
                    source=original,
                    stored_source=archived.relative_to(self.workspace).as_posix(),
                    destination=imported.destination,
                    audit_summary=dict(report.summary),
                )
            )
        except Exception as error:  # noqa: BLE001 - 单文件失败不能停止监听
            failed_source: Path | None = None
            candidate = archived if archived.exists() else source
            if candidate.exists():
                try:
                    failed_source = self._unique_target(self.failed_dir / relative)
                    self._move(candidate, failed_source)
                except Exception as move_error:  # noqa: BLE001
                    error = ExceptionGroup(
                        "导入失败，且无法移动到 failed", [error, move_error]
                    )
            self._record(
                WatchRecord(
                    timestamp=_timestamp(),
                    status="failed",
                    source=original,
                    stored_source=(
                        failed_source.relative_to(self.workspace).as_posix()
                        if failed_source is not None
                        else None
                    ),
                    error=f"{type(error).__name__}: {error}",
                )
            )

    def _move(self, source: Path, target: Path) -> None:
        fenced = self.sandbox.fence_write(target)
        fenced.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(fenced))

    def _unique_target(self, preferred: Path) -> Path:
        if not preferred.exists():
            return preferred
        for index in range(1, 10_000):
            candidate = preferred.with_name(
                f"{preferred.stem}-{index}{preferred.suffix}"
            )
            if not candidate.exists():
                return candidate
        raise FileExistsError(f"无法为归档文件分配唯一名称: {preferred}")

    def _record(self, record: WatchRecord) -> None:
        with self._lock:
            self._records.append(record)
        try:
            self.reporter(record)
        except Exception:  # noqa: BLE001 - 日志回调不能破坏监听
            pass

    def _workspace_path(self, path: Path, name: str) -> Path:
        resolved = path.resolve() if path.is_absolute() else (self.workspace / path).resolve()
        try:
            resolved.relative_to(self.workspace)
        except ValueError as error:
            raise ValueError(f"{name} 必须位于当前 workspace 内") from error
        return resolved

    def _validate_layout(self) -> None:
        try:
            self.importer.knowledge_base.root.relative_to(self.watch_dir)
        except ValueError:
            pass
        else:
            raise ValueError("kb_root 不能等于或位于 kb_watch_dir 内")
        for name, target in (
            ("kb_archive_dir", self.archive_dir),
            ("kb_failed_dir", self.failed_dir),
        ):
            if _overlaps(self.watch_dir, target):
                raise ValueError(f"{name} 不能位于 kb_watch_dir 内或包含该目录")
        if _overlaps(self.archive_dir, self.failed_dir):
            raise ValueError("kb_archive_dir 与 kb_failed_dir 不能重叠")


@depends("document_import", "filesystem", "agent_settings")
def knowledge_watch_provider(ctx: Context, _config: Any) -> None:
    importer = ctx.document_import
    filesystem = ctx.filesystem
    settings = ctx.agent_settings
    assert isinstance(importer, DocumentImportService)
    assert isinstance(settings, SettingsScope)
    config = settings.get()
    if not _required_bool(config["kb_watch_enabled"], "kb_watch_enabled"):
        return
    service = KnowledgeWatchService(
        workspace=Path(filesystem.workspace),
        importer=importer,
        sandbox=filesystem.sandbox,
        watch_dir=Path(_required_str(config["kb_watch_dir"], "kb_watch_dir")),
        archive_dir=Path(_required_str(config["kb_archive_dir"], "kb_archive_dir")),
        failed_dir=Path(_required_str(config["kb_failed_dir"], "kb_failed_dir")),
        interval_seconds=_positive_number(
            config["kb_watch_interval_seconds"], "kb_watch_interval_seconds"
        ),
        stable_scans=_positive_int(
            config["kb_watch_stable_scans"], "kb_watch_stable_scans"
        ),
    )
    ctx.provide("knowledge_watch", service)
    ctx.effect(service.start)


def _is_candidate(path: Path) -> bool:
    name = path.name
    return (
        not name.startswith(("~$", "."))
        and not any(part.startswith(".") for part in path.parts)
        and path.suffix.casefold() in SUPPORTED_SUFFIXES
        and path.suffix.casefold() not in TEMP_SUFFIXES
    )


def _overlaps(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _print_record(record: WatchRecord) -> None:
    data = asdict(record)
    details = ", ".join(
        f"{key}={value}"
        for key, value in data.items()
        if key not in {"timestamp", "status"} and value is not None
    )
    print(f"[kb-watch] {record.status}: {details}", file=sys.stderr, flush=True)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} 必须是非空字符串")
    return value


def _required_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} 必须是布尔值")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TypeError(f"{name} 必须是正整数")
    return value


def _positive_number(value: Any, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value <= 0
    ):
        raise TypeError(f"{name} 必须是正数")
    return float(value)
