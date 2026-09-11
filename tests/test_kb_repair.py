from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from mini_harness.agent import Agent
from mini_harness.bundle import BundleConfig, headless_bundle
from mini_harness.capabilities import FilesystemService
from mini_harness.client import Message, Tool
from mini_harness.cordis import Context
from mini_harness.fs_tools import ObservationTracker
from mini_harness.kb_repair import (
    KnowledgeChange,
    KnowledgeChangeSet,
    KnowledgeRepairService,
    RepairTask,
    apply_changeset,
    batch_tasks,
    plan_tasks,
    project_tools,
)
from mini_harness.knowledge_base import (
    GovernanceIssue,
    KbAuditReport,
    KnowledgeBaseService,
)
from mini_harness.prompt import PromptAssembler
from mini_harness.registry import ToolRegistry
from mini_harness.sandbox import SandboxPolicy
from mini_harness.workflow import WorkflowEngine


class FakeClient:
    MODEL = "fake"

    def chat(
        self, _messages: list[Message], _tools: list[Tool] | None = None
    ) -> Message:
        return Message(role="assistant", content="unused")


class ChangeSetClient:
    """按任务指令中的关键字返回预设的子 Agent 输出。"""

    MODEL = "changeset-fake"

    def __init__(self, answers: tuple[tuple[str, str], ...]) -> None:
        self._answers = answers
        self.tasks: list[str] = []

    def chat(
        self, messages: list[Message], _tools: list[Tool] | None = None
    ) -> Message:
        instruction = next(
            message.content or "" for message in messages if message.role == "user"
        )
        self.tasks.append(instruction)
        for marker, payload in self._answers:
            if marker in instruction:
                return Message(role="assistant", content=payload)
        return Message(role="assistant", content='{"changes": []}')


class PlanTaskTests(unittest.TestCase):
    def test_same_file_broken_links_are_merged_into_one_task(self) -> None:
        report = _report(
            (
                _issue("i1", "broken_link", "index.md", "断链 a", ("path:a.md",)),
                _issue("i2", "broken_link", "index.md", "断链 b", ("path:b.md",)),
                _issue("i3", "missing_title", "no-title.md", "缺少标题"),
            )
        )

        tasks = plan_tasks(report)

        self.assertEqual(len(tasks), 2)
        merged = next(task for task in tasks if task.write_paths == ("index.md",))
        self.assertEqual(merged.issue_ids, ("i1", "i2"))
        self.assertEqual(merged.rules, ("broken_link",))

    def test_duplicate_title_targets_share_one_task(self) -> None:
        report = _report(
            (
                _issue(
                    "d1",
                    "duplicate_title",
                    "dup-a.md",
                    "多个文档使用相同标题",
                    ("dup-a.md", "dup-b.md"),
                ),
            )
        )

        tasks = plan_tasks(report)

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].write_paths, ("dup-a.md", "dup-b.md"))

    def test_orphan_task_also_claims_the_index_page(self) -> None:
        report = _report((_issue("o1", "orphan", "lonely.md", "没有被引用"),))

        tasks = plan_tasks(report, index_page="index.md")

        self.assertEqual(tasks[0].write_paths, ("index.md", "lonely.md"))

    def test_stale_rule_is_never_planned(self) -> None:
        report = _report((_issue("s1", "stale", "old.md", "超过 180 天未更新"),))

        self.assertEqual(plan_tasks(report), ())

    def test_unknown_issue_ids_fail_loudly(self) -> None:
        report = _report((_issue("i1", "broken_link", "index.md", "断链"),))

        with self.assertRaises(ValueError):
            plan_tasks(report, issue_ids=("missing-id",))

    def test_read_paths_are_workspace_relative(self) -> None:
        report = _report((_issue("i1", "broken_link", "index.md", "断链"),))

        tasks = plan_tasks(report, kb_prefix="notes")

        self.assertEqual(tasks[0].write_paths, ("index.md",))
        self.assertEqual(tasks[0].read_paths, ("notes/index.md",))
        self.assertIn("notes/index.md", tasks[0].instruction)


class BatchTaskTests(unittest.TestCase):
    def test_write_paths_never_overlap_inside_a_batch(self) -> None:
        tasks = (
            _task("task-01", ("index.md",)),
            _task("task-02", ("no-title.md",)),
            _task("task-03", ("dup-a.md", "dup-b.md")),
            _task("task-04", ("index.md", "lonely.md")),
        )

        batches = batch_tasks(tasks)

        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]), 3)
        self.assertEqual(len(batches[1]), 1)
        for batch in batches:
            seen: set[str] = set()
            for task in batch:
                self.assertFalse(seen & set(task.write_paths))
                seen.update(task.write_paths)

    def test_every_task_is_scheduled_exactly_once(self) -> None:
        tasks = tuple(_task(f"task-{index:02d}", ("shared.md",)) for index in range(4))

        batches = batch_tasks(tasks)

        self.assertEqual(len(batches), 4)
        self.assertEqual(sum(len(batch) for batch in batches), 4)


class ProjectToolsTests(unittest.TestCase):
    def test_missing_tool_fails_instead_of_degrading(self) -> None:
        registry = ToolRegistry()
        registry.register(_stub_tool("read"))

        with self.assertRaises(LookupError):
            project_tools(registry, ("read", "kb_search"))

    def test_write_tools_are_rejected(self) -> None:
        registry = ToolRegistry()
        registry.register(_stub_tool("edit"))

        with self.assertRaises(ValueError):
            project_tools(registry, ("edit",))


class ApplyChangeSetTests(unittest.TestCase):
    def test_path_outside_the_whitelist_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _write(workspace / "notes/index.md", "# Index\n")
            _write(workspace / "outside.md", "# Outside\n")
            filesystem = _filesystem(workspace)

            result = apply_changeset(
                KnowledgeChangeSet(
                    "task-01",
                    (KnowledgeChange("../outside.md", "Outside", "Changed"),),
                ),
                filesystem=filesystem,
                kb_root=(workspace / "notes").resolve(),
                allowed=frozenset({"index.md"}),
            )

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.applied_paths, ())
            assert result.error is not None
            self.assertIn("write_paths", result.error)
            self.assertEqual(
                (workspace / "outside.md").read_text(encoding="utf-8"), "# Outside\n"
            )

    def test_ambiguous_old_string_is_skipped_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _write(workspace / "notes/index.md", "todo\ntodo\n")
            filesystem = _filesystem(workspace)

            result = apply_changeset(
                KnowledgeChangeSet(
                    "task-01", (KnowledgeChange("index.md", "todo", "done"),)
                ),
                filesystem=filesystem,
                kb_root=(workspace / "notes").resolve(),
                allowed=frozenset({"index.md"}),
            )

            assert result.error is not None
            self.assertIn("匹配 2 处", result.error)
            self.assertEqual(
                (workspace / "notes/index.md").read_text(encoding="utf-8"),
                "todo\ntodo\n",
            )

    def test_missing_old_string_is_skipped_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _write(workspace / "notes/index.md", "# Index\n")
            filesystem = _filesystem(workspace)

            result = apply_changeset(
                KnowledgeChangeSet(
                    "task-01", (KnowledgeChange("index.md", "缺席", "替换"),)
                ),
                filesystem=filesystem,
                kb_root=(workspace / "notes").resolve(),
                allowed=frozenset({"index.md"}),
            )

            assert result.error is not None
            self.assertIn("匹配 0 处", result.error)

    def test_partial_failure_still_applies_the_valid_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _write(workspace / "notes/index.md", "# Index\n\n[A](a.md)\n")
            filesystem = _filesystem(workspace)

            result = apply_changeset(
                KnowledgeChangeSet(
                    "task-01",
                    (
                        KnowledgeChange("index.md", "(a.md)", "(b.md)"),
                        KnowledgeChange("index.md", "不存在", "x"),
                    ),
                ),
                filesystem=filesystem,
                kb_root=(workspace / "notes").resolve(),
                allowed=frozenset({"index.md"}),
            )

            self.assertEqual(result.status, "applied")
            self.assertEqual(result.applied_paths, ("index.md",))
            self.assertIsNotNone(result.error)
            self.assertIn(
                "(b.md)", (workspace / "notes/index.md").read_text(encoding="utf-8")
            )


class RepairServiceTests(unittest.TestCase):
    def test_dry_run_previews_without_touching_any_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _seed_broken_link(workspace)
            before = _snapshot(workspace)
            service = _service(workspace, _link_fix_client())

            report = service.repair(rules=("broken_link",), dry_run=True)

            self.assertTrue(report.dry_run)
            self.assertEqual(report.applied_changes, 0)
            self.assertEqual([task.status for task in report.tasks], ["planned"])
            self.assertEqual(len(report.changesets), 1)
            self.assertIsNone(report.audit_after)
            self.assertEqual(_snapshot(workspace), before)

    def test_apply_fixes_the_issue_and_re_audits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _seed_broken_link(workspace)
            service = _service(workspace, _link_fix_client())

            report = service.repair(rules=("broken_link",), dry_run=False)

            self.assertEqual(report.applied_changes, 1)
            self.assertEqual([task.status for task in report.tasks], ["applied"])
            self.assertEqual(report.audit_before.get("broken_link"), 1)
            assert report.audit_after is not None
            self.assertEqual(report.audit_after.get("broken_link", 0), 0)
            self.assertIn(
                "(generator.md)",
                (workspace / "notes/index.md").read_text(encoding="utf-8"),
            )
            self.assertNotEqual(report.to_json(), "")

    def test_non_json_output_fails_one_task_without_blocking_the_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _seed_broken_link(workspace)
            _write(workspace / "notes/no-title.md", "没有一级标题的正文\n")
            client = ChangeSetClient(
                (
                    ("no-title.md", "我已经把标题补好了。"),
                    ("index.md", _link_fix_payload()),
                )
            )
            service = _service(workspace, client)

            report = service.repair(
                rules=("broken_link", "missing_title"), dry_run=False
            )

            statuses = {task.task_id: task.status for task in report.tasks}
            self.assertEqual(sorted(statuses.values()), ["applied", "failed"])
            self.assertEqual(report.applied_changes, 1)
            failed = next(task for task in report.tasks if task.status == "failed")
            assert failed.error is not None
            self.assertIn("JSON", failed.error.upper())

    def test_changeset_writing_outside_write_paths_is_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _seed_broken_link(workspace)
            escaping = json.dumps(
                {
                    "task_id": "task-01",
                    "changes": [
                        {
                            "path": "../../escape.md",
                            "old_string": "a",
                            "new_string": "b",
                        }
                    ],
                }
            )
            service = _service(workspace, ChangeSetClient((("index.md", escaping),)))

            report = service.repair(rules=("broken_link",), dry_run=False)

            self.assertEqual([task.status for task in report.tasks], ["failed"])
            self.assertEqual(report.applied_changes, 0)
            self.assertFalse((workspace / "escape.md").exists())

    def test_max_tasks_defers_the_remainder_instead_of_dropping_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _write(workspace / "notes/index.md", "# Index\n")
            _write(workspace / "notes/a.md", "没有标题 A\n")
            _write(workspace / "notes/b.md", "没有标题 B\n")
            service = _service(workspace, ChangeSetClient(()))

            report = service.repair(
                rules=("missing_title",), dry_run=True, max_tasks=1
            )

            self.assertTrue(report.truncated)
            self.assertEqual(len(report.tasks), 2)
            self.assertEqual(
                sum(1 for task in report.tasks if task.status == "skipped"), 2
            )

    def test_missing_read_only_tool_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _seed_broken_link(workspace)
            service = _service(
                workspace, _link_fix_client(), tool_names=("read", "grep")
            )

            with self.assertRaises(LookupError):
                service.repair(rules=("broken_link",), dry_run=True)


class RepairPluginTests(unittest.TestCase):
    def test_bundle_registers_kb_repair(self) -> None:
        with _workspace() as workspace:
            _write(workspace / "notes/index.md", "# Index\n")
            ctx = Context()
            try:
                ctx.plugin(
                    headless_bundle,
                    BundleConfig(settings_document={}, client=FakeClient()),
                )
                agent = ctx.require("agent")
                prompt = ctx.require("prompt")
                assert isinstance(agent, Agent)
                assert isinstance(prompt, PromptAssembler)

                self.assertEqual(len(agent.tool_names), 28)
                self.assertIn("kb_repair", agent.tool_names)
                self.assertIn("kb_repair", ctx.service_names)
                self.assertIn(
                    "knowledge-base:repair",
                    {section.name for section in prompt.sections},
                )
                handle = next(
                    item for item in ctx.handles if item.name == "kb_repair_provider"
                )
                self.assertEqual(handle.state, "active")
            finally:
                ctx.dispose()

    def test_disposing_the_provider_removes_tool_and_prompt(self) -> None:
        with _workspace() as workspace:
            _write(workspace / "notes/index.md", "# Index\n")
            ctx = Context()
            try:
                ctx.plugin(
                    headless_bundle,
                    BundleConfig(settings_document={}, client=FakeClient()),
                )
                agent = ctx.require("agent")
                prompt = ctx.require("prompt")
                assert isinstance(agent, Agent)
                assert isinstance(prompt, PromptAssembler)

                provider = next(
                    item for item in ctx.handles if item.name == "kb_repair_provider"
                )
                provider.dispose()

                self.assertNotIn("kb_repair", ctx.service_names)
                self.assertNotIn("kb_repair", agent.tool_names)
                self.assertIn("kb_audit", agent.tool_names)
                self.assertNotIn(
                    "knowledge-base:repair",
                    {section.name for section in prompt.sections},
                )
            finally:
                ctx.dispose()

    def test_registered_tool_runs_the_full_loop(self) -> None:
        with _workspace() as workspace:
            _seed_broken_link(workspace)
            ctx = Context()
            try:
                ctx.plugin(
                    headless_bundle,
                    BundleConfig(
                        settings_document={"agent": {"kb_watch_enabled": False}},
                        client=_link_fix_client(),
                        enable_console_questions=False,
                    ),
                )
                tools = ctx.require("tools")
                assert isinstance(tools, ToolRegistry)
                tool = tools.get("kb_repair")
                self.assertIsNotNone(tool)
                assert tool is not None

                preview = json.loads(
                    tool.execute({"rules": ["broken_link"], "dry_run": True})
                )
                self.assertTrue(preview["dry_run"])
                self.assertEqual(preview["applied_changes"], 0)
                self.assertEqual(len(preview["changesets"]), 1)

                applied = json.loads(
                    tool.execute({"rules": ["broken_link"], "dry_run": False})
                )
                self.assertEqual(applied["applied_changes"], 1)
                self.assertEqual(applied["audit_after"].get("broken_link", 0), 0)
            finally:
                ctx.dispose()

    def test_disabled_flag_keeps_the_other_tools(self) -> None:
        with _workspace() as workspace:
            _write(workspace / "notes/index.md", "# Index\n")
            ctx = Context()
            try:
                ctx.plugin(
                    headless_bundle,
                    BundleConfig(
                        settings_document={"agent": {"kb_repair_enabled": False}},
                        client=FakeClient(),
                    ),
                )
                agent = ctx.require("agent")
                assert isinstance(agent, Agent)

                self.assertEqual(len(agent.tool_names), 27)
                self.assertNotIn("kb_repair", agent.tool_names)
                self.assertNotIn("kb_repair", ctx.service_names)
                self.assertIn("kb_audit", agent.tool_names)
            finally:
                ctx.dispose()


class _workspace:
    """临时 workspace + chdir，Bundle 里的 filesystem 以 cwd 为根。"""

    def __enter__(self) -> Path:
        self._tmp = tempfile.TemporaryDirectory()
        self._previous = Path.cwd()
        os.chdir(self._tmp.name)
        return Path(self._tmp.name)

    def __exit__(self, *_exc: object) -> None:
        os.chdir(self._previous)
        self._tmp.cleanup()


def _report(issues: tuple[GovernanceIssue, ...]) -> KbAuditReport:
    return KbAuditReport(
        root="notes",
        document_count=len(issues),
        issue_count=len(issues),
        summary={},
        issues=issues,
        truncated=False,
    )


def _issue(
    id: str,
    rule: str,
    path: str,
    message: str,
    targets: tuple[str, ...] = (),
) -> GovernanceIssue:
    return GovernanceIssue(
        id=id, rule=rule, severity="error", path=path, message=message, targets=targets
    )


def _task(id: str, write_paths: tuple[str, ...]) -> RepairTask:
    return RepairTask(
        id=id,
        rules=("broken_link",),
        issue_ids=(f"{id}-issue",),
        write_paths=write_paths,
        read_paths=write_paths,
        instruction="fixture",
    )


def _stub_tool(name: str) -> Tool:
    return Tool(name, name, {"type": "object", "properties": {}}, lambda _a: "")


def _filesystem(workspace: Path) -> FilesystemService:
    return FilesystemService(
        workspace.resolve(),
        SandboxPolicy("workspace-write", workspace.resolve()),
        ObservationTracker(),
    )


def _service(
    workspace: Path,
    client: ChangeSetClient,
    *,
    tool_names: tuple[str, ...] = ("read", "grep", "glob", "kb_search"),
) -> KnowledgeRepairService:
    filesystem = _filesystem(workspace)
    knowledge_base = KnowledgeBaseService(
        root=(workspace / "notes").resolve(),
        include_globs=("**/*.md", "**/*.txt"),
        exclude_dirs=(".git", ".mini-harness"),
        required_frontmatter=(),
        stale_days=180,
        index_pages=("index.md",),
    )
    registry = ToolRegistry()
    for name in tool_names:
        registry.register(_stub_tool(name))
    return KnowledgeRepairService(
        knowledge_base=knowledge_base,
        filesystem=filesystem,
        registry=registry,
        client=client,
        workflow=WorkflowEngine(max_concurrency=2, max_agents=16),
        subagent_tools=("read", "grep", "glob", "kb_search"),
    )


def _seed_broken_link(workspace: Path) -> None:
    _write(workspace / "notes/index.md", "# Index\n\n[生成器](generators.md)\n")
    _write(workspace / "notes/generator.md", "# Python 生成器\n\n按需产生值。\n")


def _link_fix_payload() -> str:
    return json.dumps(
        {
            "task_id": "task-01",
            "changes": [
                {
                    "path": "index.md",
                    "old_string": "(generators.md)",
                    "new_string": "(generator.md)",
                    "reason": "目标笔记实际是 generator.md",
                }
            ],
            "issue_ids": [],
            "notes": "",
        },
        ensure_ascii=False,
    )


def _link_fix_client() -> ChangeSetClient:
    return ChangeSetClient((("index.md", _link_fix_payload()),))


def _snapshot(workspace: Path) -> dict[str, str]:
    return {
        path.relative_to(workspace).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(workspace.rglob("*"))
        if path.is_file()
    }


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
