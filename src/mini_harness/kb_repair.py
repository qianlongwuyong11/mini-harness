"""知识库批量修复编排：只读并发调查 + 串行写入 + 终态复审。

`kb_audit` 已经把「哪里有问题」变成确定性事实，模型不需要并发去发现问题；
并发的价值在审计之后：每个问题都要读上下文才能决定怎么改。所以这里的分工是

1. 主 Agent 按写集（write_paths）把 issue 聚合成 RepairTask，并做冲突分批；
2. 只读子 Agent 在受限工具集下并发调查，产出结构化 ChangeSet，不碰磁盘；
3. 主 Agent 串行应用 ChangeSet，再用 kb_audit 以文件终态验收。

并发只用于调查，是因为写盘冲突、跨文件全局决策和乐观锁都无法靠线程解决；
分批依据是 write_paths 是否相交，`index.md` 这类共享入口页会自然独占一批。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .capabilities import FilesystemService
from .client import ChatClient, Tool
from .cordis import Context, depends
from .knowledge_base import GovernanceIssue, KbAuditReport, KnowledgeBaseService
from .prompt import PromptAssembler
from .registry import ToolRegistry
from .settings import SettingsScope
from .tool_subagent import run_tool_subagent
from .workflow import WorkflowEngine

# `stale` 只报告不自动修复：它需要用户决定内容是否仍然有效。
REPAIRABLE_RULES = (
    "broken_link",
    "duplicate_title",
    "missing_frontmatter",
    "missing_title",
    "orphan",
)

REPAIR_SUBAGENT_TOOLS = ("read", "grep", "glob", "kb_search")

# 防越权写入与无限委派：这些工具即使存在于主注册表也不投影给子 Agent。
FORBIDDEN_SUBAGENT_TOOLS = frozenset(
    {
        "ask_user_question",
        "edit",
        "interrupt_agent",
        "kb_import_document",
        "kb_repair",
        "send_message",
        "shell",
        "subagent",
        "subagent_fork",
        "web_fetch",
        "web_search",
        "workflow",
        "write",
    }
)

SUBAGENT_SYSTEM_PROMPT = (
    "你是知识库修复调查员。只读工具：read / grep / glob / kb_search。\n"
    "先阅读相关笔记确认事实，再输出唯一一个 JSON 对象：\n"
    '{"task_id": "...", "changes": [{"path": "...", "old_string": "...", '
    '"new_string": "...", "reason": "..."}], "issue_ids": ["..."], "notes": "..."}\n'
    "约束：changes[].path 只能是任务列出的 write_paths；old_string 必须是目标"
    "文件中唯一出现的原文片段；不得删除正文、不得重写整篇笔记、不得移动或"
    "重命名笔记、不得输出 JSON 以外的内容。"
)


@dataclass(frozen=True)
class RepairTask:
    """一个可独立交给子 Agent 的修复单元。"""

    id: str
    rules: tuple[str, ...]
    issue_ids: tuple[str, ...]
    write_paths: tuple[str, ...]
    read_paths: tuple[str, ...]
    instruction: str


@dataclass(frozen=True)
class KnowledgeChange:
    """一次最小编辑，语义对齐 filesystem.edit。"""

    path: str
    old_string: str
    new_string: str
    reason: str = ""


@dataclass(frozen=True)
class KnowledgeChangeSet:
    """子 Agent 的结构化产出。"""

    task_id: str
    changes: tuple[KnowledgeChange, ...]
    issue_ids: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class RepairTaskResult:
    task_id: str
    status: str  # planned / applied / skipped / failed
    change_count: int
    applied_paths: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class KnowledgeRepairReport:
    dry_run: bool
    batches: int
    tasks: tuple[RepairTaskResult, ...]
    applied_changes: int
    audit_before: Mapping[str, int]
    audit_after: Mapping[str, int] | None = None
    changesets: tuple[KnowledgeChangeSet, ...] = ()
    truncated: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, allow_nan=False)


def project_tools(registry: ToolRegistry, allowed: Sequence[str]) -> tuple[Tool, ...]:
    """从主注册表投影只读子集；缺失工具直接报错，避免静默降级。"""
    forbidden = sorted(set(allowed) & FORBIDDEN_SUBAGENT_TOOLS)
    if forbidden:
        raise ValueError(f"只读子 Agent 不得使用写入类工具: {', '.join(forbidden)}")
    projected: list[Tool] = []
    missing: list[str] = []
    for name in dict.fromkeys(allowed):
        tool = registry.get(name)
        if tool is None:
            missing.append(name)
        else:
            projected.append(tool)
    if missing:
        raise LookupError(f"只读工具投影缺少: {', '.join(missing)}")
    return tuple(projected)


def plan_tasks(
    report: KbAuditReport,
    *,
    issue_ids: Sequence[str] | None = None,
    rules: Sequence[str] | None = None,
    index_page: str | None = None,
    kb_prefix: str = "",
    max_tasks: int = 16,
) -> tuple[RepairTask, ...]:
    """把 issue 按 write_paths 聚合成任务；超出 max_tasks 的部分被截断。"""
    if not isinstance(max_tasks, int) or isinstance(max_tasks, bool) or max_tasks < 1:
        raise ValueError("max_tasks 必须是正整数")
    return plan_all_tasks(
        report,
        issue_ids=issue_ids,
        rules=rules,
        index_page=index_page,
        kb_prefix=kb_prefix,
    )[:max_tasks]


def plan_all_tasks(
    report: KbAuditReport,
    *,
    issue_ids: Sequence[str] | None = None,
    rules: Sequence[str] | None = None,
    index_page: str | None = None,
    kb_prefix: str = "",
) -> tuple[RepairTask, ...]:
    """不截断的规划结果；服务层用它区分「没规划出来」和「超过上限」。"""
    selected_rules = frozenset(rules) if rules else frozenset(REPAIRABLE_RULES)
    wanted = frozenset(issue_ids) if issue_ids is not None else None
    candidates = [
        issue
        for issue in report.issues
        if issue.rule in REPAIRABLE_RULES and issue.rule in selected_rules
    ]
    if wanted is not None:
        known = {issue.id for issue in candidates}
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError(f"issue_ids 不在可自动修复的审计结果中: {', '.join(unknown)}")
        candidates = [issue for issue in candidates if issue.id in wanted]

    groups: dict[tuple[str, ...], list[GovernanceIssue]] = {}
    for issue in candidates:
        key = _write_paths(issue, index_page)
        groups.setdefault(key, []).append(issue)

    tasks: list[RepairTask] = []
    for index, (write_paths, issues) in enumerate(groups.items(), start=1):
        read_paths = tuple(
            _workspace_path(kb_prefix, path)
            for path in dict.fromkeys(
                (*write_paths, *(page for page in (index_page,) if page))
            )
        )
        tasks.append(
            RepairTask(
                id=f"task-{index:02d}",
                rules=tuple(sorted({issue.rule for issue in issues})),
                issue_ids=tuple(issue.id for issue in issues),
                write_paths=write_paths,
                read_paths=read_paths,
                instruction=_instruction(
                    f"task-{index:02d}", issues, write_paths, read_paths, kb_prefix
                ),
            )
        )
    return tuple(tasks)


def batch_tasks(tasks: Sequence[RepairTask]) -> tuple[tuple[RepairTask, ...], ...]:
    """贪心分批：同一批内任意两个任务的 write_paths 交集为空。"""
    batches: list[list[RepairTask]] = []
    claimed: list[set[str]] = []
    for task in tasks:
        paths = set(task.write_paths)
        for index, taken in enumerate(claimed):
            if not (taken & paths):
                batches[index].append(task)
                taken.update(paths)
                break
        else:
            batches.append([task])
            claimed.append(set(paths))
    return tuple(tuple(batch) for batch in batches)


def parse_changeset(raw: str, task: RepairTask) -> KnowledgeChangeSet:
    """严格解析子 Agent 的 JSON 产出；越界路径让整个 ChangeSet 作废。"""
    payload = json.loads(_strip_code_fence(raw))
    if not isinstance(payload, dict):
        raise TypeError("ChangeSet 顶层必须是 JSON 对象")
    raw_changes = payload.get("changes", [])
    if not isinstance(raw_changes, list):
        raise TypeError("changes 必须是数组")
    allowed = frozenset(task.write_paths)
    changes: list[KnowledgeChange] = []
    for item in raw_changes:
        if not isinstance(item, Mapping):
            raise TypeError("change 必须是对象")
        path = Path(_required_text(item, "path")).as_posix()
        if path not in allowed:
            raise PermissionError(
                f"change.path {path!r} 不在任务 {task.id} 的 write_paths 白名单内"
            )
        old_string = _required_text(item, "old_string")
        new_string = item.get("new_string")
        if not isinstance(new_string, str):
            raise TypeError("new_string 必须是字符串")
        if old_string == new_string:
            raise ValueError(f"change.old_string 与 new_string 相同: {path}")
        reason = item.get("reason", "")
        changes.append(
            KnowledgeChange(
                path=path,
                old_string=old_string,
                new_string=new_string,
                reason=reason if isinstance(reason, str) else "",
            )
        )
    raw_issue_ids = payload.get("issue_ids", task.issue_ids)
    issue_ids = (
        tuple(item for item in raw_issue_ids if isinstance(item, str))
        if isinstance(raw_issue_ids, list)
        else task.issue_ids
    )
    notes = payload.get("notes", "")
    return KnowledgeChangeSet(
        task_id=task.id,
        changes=tuple(changes),
        issue_ids=issue_ids,
        notes=notes if isinstance(notes, str) else "",
    )


def apply_changeset(
    changeset: KnowledgeChangeSet,
    *,
    filesystem: FilesystemService,
    kb_root: Path,
    allowed: frozenset[str],
) -> RepairTaskResult:
    """串行应用一个 ChangeSet；单条失败不回滚其他条目，但如实记录。"""
    applied: list[str] = []
    problems: list[str] = []
    for change in changeset.changes:
        try:
            target = _resolve_target(change.path, kb_root, allowed)
            # 先读：同时满足 read-before-write 并刷新 ObservationTracker。
            filesystem.read({"path": str(target), "limit": 2000})
            occurrences = target.read_text(encoding="utf-8").count(change.old_string)
            if occurrences != 1:
                problems.append(
                    f"{change.path}: old_string 匹配 {occurrences} 处（要求恰好 1 处），已跳过"
                )
                continue
            filesystem.edit(
                {
                    "path": str(target),
                    "old_string": change.old_string,
                    "new_string": change.new_string,
                }
            )
            applied.append(change.path)
        except Exception as error:  # noqa: BLE001 - 每条编辑独立结算
            problems.append(f"{change.path}: {type(error).__name__}: {error}")
    if applied:
        status = "applied"
    elif problems:
        status = "failed"
    else:
        status = "skipped"
    return RepairTaskResult(
        task_id=changeset.task_id,
        status=status,
        change_count=len(changeset.changes),
        applied_paths=tuple(dict.fromkeys(applied)),
        error="; ".join(problems) if problems else None,
    )


@dataclass(frozen=True)
class _Outcome:
    task_id: str
    changeset: KnowledgeChangeSet | None
    error: str | None


@dataclass(frozen=True)
class KnowledgeRepairService:
    """编排「审计 → 规划 → 只读并发调查 → 串行应用 → 复审」的闭环。"""

    knowledge_base: KnowledgeBaseService
    filesystem: FilesystemService
    registry: ToolRegistry
    client: ChatClient
    workflow: WorkflowEngine
    max_tasks: int = 16
    subagent_max_steps: int = 6
    auto_reaudit: bool = True
    subagent_tools: tuple[str, ...] = REPAIR_SUBAGENT_TOOLS

    def repair(
        self,
        *,
        issue_ids: Sequence[str] | None = None,
        rules: Sequence[str] | None = None,
        dry_run: bool = True,
        max_tasks: int | None = None,
    ) -> KnowledgeRepairReport:
        selected = tuple(rules) if rules else REPAIRABLE_RULES
        unknown = sorted(set(selected) - set(REPAIRABLE_RULES))
        if unknown:
            raise ValueError(f"不支持自动修复的规则: {', '.join(unknown)}")

        before = self.knowledge_base.audit(sorted(selected), max_issues=1000)
        planned = plan_all_tasks(
            before,
            issue_ids=issue_ids,
            rules=selected,
            index_page=self._index_page(),
            kb_prefix=self._kb_prefix(),
        )
        limit = min(
            max_tasks if max_tasks is not None else self.max_tasks,
            self.max_tasks,
            self.workflow.max_agents,
        )
        tasks, deferred = planned[:limit], planned[limit:]
        if not tasks:
            return KnowledgeRepairReport(
                dry_run=dry_run,
                batches=0,
                tasks=(),
                applied_changes=0,
                audit_before=dict(before.summary),
            )

        # 缺少任何一个只读工具都立即失败，而不是让子 Agent 静默降级。
        tools = project_tools(self.registry, self.subagent_tools)
        outcomes = self._investigate(batch_tasks(tasks), tools)

        results: list[RepairTaskResult] = []
        changesets: list[KnowledgeChangeSet] = []
        applied_changes = 0
        for task in tasks:
            outcome = outcomes[task.id]
            if outcome.changeset is None:
                results.append(
                    RepairTaskResult(task.id, "failed", 0, error=outcome.error)
                )
                continue
            changesets.append(outcome.changeset)
            if dry_run:
                results.append(
                    RepairTaskResult(
                        task.id,
                        "planned" if outcome.changeset.changes else "skipped",
                        len(outcome.changeset.changes),
                    )
                )
                continue
            result = apply_changeset(
                outcome.changeset,
                filesystem=self.filesystem,
                kb_root=self.knowledge_base.root,
                allowed=frozenset(task.write_paths),
            )
            applied_changes += len(result.applied_paths)
            results.append(result)
        for task in deferred:
            results.append(
                RepairTaskResult(
                    task.id, "skipped", 0, error=f"超过 max_tasks={limit} 上限，本次未处理"
                )
            )

        audit_after = (
            dict(
                self.knowledge_base.audit(sorted(selected), max_issues=1000).summary
            )
            if not dry_run and self.auto_reaudit
            else None
        )
        return KnowledgeRepairReport(
            dry_run=dry_run,
            batches=len(batch_tasks(tasks)),
            tasks=tuple(results),
            applied_changes=applied_changes,
            audit_before=dict(before.summary),
            audit_after=audit_after,
            changesets=tuple(changesets),
            truncated=bool(deferred),
        )

    def _investigate(
        self,
        batches: Sequence[Sequence[RepairTask]],
        tools: Sequence[Tool],
    ) -> dict[str, _Outcome]:
        """批内并发、批间串行；单个子 Agent 失败不影响同批其他任务。"""
        outcomes: dict[str, _Outcome] = {}
        for batch in batches:
            settled = self.workflow.parallel(
                [lambda task=task: self._run_one(task, tools) for task in batch]
            )
            for task, outcome in zip(batch, settled):
                outcomes[task.id] = (
                    outcome
                    if isinstance(outcome, _Outcome)
                    else _Outcome(task.id, None, "子 Agent 调查未返回结果")
                )
        return outcomes

    def _run_one(self, task: RepairTask, tools: Sequence[Tool]) -> _Outcome:
        try:
            result = run_tool_subagent(
                self.client,
                task.instruction,
                SUBAGENT_SYSTEM_PROMPT,
                tools,
                max_steps=self.subagent_max_steps,
            )
            if result.stop_reason != "completed":
                detail = result.diagnostic or result.output[:300] or "（无输出）"
                return _Outcome(
                    task.id, None, f"子 Agent 未完成（{result.stop_reason}）: {detail}"
                )
            return _Outcome(task.id, parse_changeset(result.output, task), None)
        except Exception as error:  # noqa: BLE001 - 任务级失败结算
            return _Outcome(task.id, None, f"{type(error).__name__}: {error}")

    def _kb_prefix(self) -> str:
        try:
            relative = self.knowledge_base.root.relative_to(
                Path(self.filesystem.workspace).resolve()
            ).as_posix()
        except ValueError:
            return ""
        return "" if relative == "." else relative

    def _index_page(self) -> str | None:
        root = self.knowledge_base.root
        if not root.is_dir():
            return None
        for path in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
            if path.is_file() and _normal_path(path.name) in (
                self.knowledge_base.index_pages
            ):
                return path.name
        return None


@depends(
    "knowledge_base", "filesystem", "tools", "llm", "workflow", "agent_settings"
)
def kb_repair_provider(ctx: Context, _config: Any) -> None:
    knowledge_base = ctx.knowledge_base
    filesystem = ctx.filesystem
    tools = ctx.tools
    client = ctx.llm
    workflow = ctx.workflow
    settings = ctx.agent_settings
    assert isinstance(knowledge_base, KnowledgeBaseService)
    assert isinstance(filesystem, FilesystemService)
    assert isinstance(tools, ToolRegistry)
    assert isinstance(workflow, WorkflowEngine)
    assert isinstance(settings, SettingsScope)
    config = settings.get()
    if not _required_bool(config["kb_repair_enabled"], "kb_repair_enabled"):
        return
    max_concurrency = _positive_int(
        config["kb_repair_max_concurrency"], "kb_repair_max_concurrency"
    )
    engine = (
        workflow
        if workflow.max_concurrency == max_concurrency
        else WorkflowEngine(
            max_concurrency=max_concurrency, max_agents=workflow.max_agents
        )
    )
    ctx.provide(
        "kb_repair",
        KnowledgeRepairService(
            knowledge_base=knowledge_base,
            filesystem=filesystem,
            registry=tools,
            client=client,
            workflow=engine,
            max_tasks=_positive_int(config["kb_repair_max_tasks"], "kb_repair_max_tasks"),
            subagent_max_steps=_positive_int(
                config["kb_repair_subagent_max_steps"], "kb_repair_subagent_max_steps"
            ),
            auto_reaudit=_required_bool(
                config["kb_repair_auto_reaudit"], "kb_repair_auto_reaudit"
            ),
        ),
    )


@depends("kb_repair", "tools", "prompt")
def kb_repair_consumer(ctx: Context, _config: Any) -> None:
    service = ctx.kb_repair
    tools = ctx.tools
    prompt = ctx.prompt
    assert isinstance(service, KnowledgeRepairService)
    assert isinstance(tools, ToolRegistry)
    assert isinstance(prompt, PromptAssembler)
    ctx.effect(
        lambda: prompt.section(
            "knowledge-base:repair",
            (
                "批量修复时：先 kb_audit 取得 issue.id，再用 kb_repair(dry_run=true)"
                "生成方案并向用户展示；用户确认后用 kb_repair(dry_run=false) 应用"
                "并自动复审。单点、简单的修改仍直接用 read + edit，不必走 "
                "kb_repair。kb_repair 不会删除、移动或重命名笔记。"
            ),
            order=12,
        )
    )
    ctx.effect(
        lambda: tools.register(
            Tool(
                "kb_repair",
                (
                    "批量修复 kb_audit 发现的知识库问题：只读子 Agent 并发调查并"
                    "产出修改方案，主 Agent 串行应用并自动复审。"
                    "dry_run=true 只返回预览，不写任何文件。"
                ),
                {
                    "type": "object",
                    "properties": {
                        "issue_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "uniqueItems": True,
                            "description": (
                                "来自 kb_audit 的 issue.id；省略表示全部可自动修复的问题"
                            ),
                        },
                        "rules": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": list(REPAIRABLE_RULES),
                            },
                            "uniqueItems": True,
                        },
                        "dry_run": {"type": "boolean", "default": True},
                        "max_tasks": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 32,
                        },
                    },
                },
                lambda arguments: _repair_tool(service, arguments),
            )
        )
    )


def _repair_tool(
    service: KnowledgeRepairService, arguments: Mapping[str, Any]
) -> str:
    issue_ids = _optional_str_list(arguments, "issue_ids")
    rules = _optional_str_list(arguments, "rules")
    dry_run = arguments.get("dry_run", True)
    if not isinstance(dry_run, bool):
        raise TypeError("dry_run 必须是布尔值")
    max_tasks = arguments.get("max_tasks")
    if max_tasks is not None and (
        not isinstance(max_tasks, int) or isinstance(max_tasks, bool) or max_tasks < 1
    ):
        raise TypeError("max_tasks 必须是正整数")
    return service.repair(
        issue_ids=issue_ids,
        rules=rules,
        dry_run=dry_run,
        max_tasks=max_tasks,
    ).to_json()


def _write_paths(issue: GovernanceIssue, index_page: str | None) -> tuple[str, ...]:
    """每条 issue 的修复会动到哪些文件——这是冲突检测唯一的键。"""
    if issue.rule == "duplicate_title":
        # 改名是全局决策：所有同名笔记必须交给同一个子 Agent。
        return tuple(sorted(issue.targets or (issue.path,)))
    if issue.rule == "orphan" and index_page is not None:
        return tuple(sorted({issue.path, index_page}))
    return (issue.path,)


def _instruction(
    task_id: str,
    issues: Sequence[GovernanceIssue],
    write_paths: Sequence[str],
    read_paths: Sequence[str],
    kb_prefix: str,
) -> str:
    lines = [
        f"任务编号：{task_id}",
        f"知识库根目录（相对 workspace）：{kb_prefix or '.'}",
        "changes[].path 必须使用下列相对知识库根的路径：",
        *(f"  - {path}" for path in write_paths),
        "调用 read 工具时请使用下列相对 workspace 的路径：",
        *(f"  - {path}" for path in read_paths),
        "需要修复的审计问题：",
    ]
    for issue in issues:
        targets = f"（相关文件：{', '.join(issue.targets)}）" if issue.targets else ""
        lines.append(f"  - [{issue.rule}] {issue.path}: {issue.message}{targets}")
    lines.append(
        "先读上下文确认事实，再输出唯一一个 JSON 对象，不要输出任何解释文字。"
    )
    return "\n".join(lines)


def _resolve_target(path: str, kb_root: Path, allowed: frozenset[str]) -> Path:
    if not path or Path(path).is_absolute():
        raise ValueError(f"path 必须是知识库根目录内的相对路径: {path!r}")
    normalized = Path(path).as_posix()
    if normalized not in allowed:
        raise PermissionError(f"path 不在本任务的 write_paths 白名单内: {normalized}")
    target = (kb_root / normalized).resolve()
    try:
        target.relative_to(kb_root)
    except ValueError as error:
        raise PermissionError(f"path 逃出知识库根目录: {normalized}") from error
    if not target.is_file():
        raise FileNotFoundError(f"目标文件不存在: {normalized}")
    return target


def _workspace_path(kb_prefix: str, relative: str) -> str:
    return f"{kb_prefix}/{relative}" if kb_prefix else relative


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    body = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
    return "\n".join(body).strip()


def _normal_path(value: str) -> str:
    return Path(value).as_posix().casefold()


def _required_text(item: Mapping[str, Any], name: str) -> str:
    value = item.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} 必须是非空字符串")
    return value


def _optional_str_list(
    arguments: Mapping[str, Any], name: str
) -> tuple[str, ...] | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise TypeError(f"{name} 必须是非空字符串数组")
    return tuple(value)


def _required_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} 必须是布尔值")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TypeError(f"{name} 必须是正整数")
    return value
