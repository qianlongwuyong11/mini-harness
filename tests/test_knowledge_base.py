from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date
from pathlib import Path

from mini_harness.agent import Agent
from mini_harness.bundle import BundleConfig, headless_bundle
from mini_harness.client import Message, Tool, ToolCall
from mini_harness.cordis import Context
from mini_harness.knowledge_base import KnowledgeBaseService
from mini_harness.prompt import PromptAssembler


class FakeClient:
    MODEL = "fake"

    def chat(
        self, _messages: list[Message], _tools: list[Tool] | None = None
    ) -> Message:
        return Message(role="assistant", content="unused")


class AuditClient:
    MODEL = "audit-fake"

    def __init__(self) -> None:
        self.calls = 0

    def chat(
        self, _messages: list[Message], _tools: list[Tool] | None = None
    ) -> Message:
        self.calls += 1
        if self.calls == 1:
            return Message(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        id="audit-1",
                        name="kb_audit",
                        arguments='{"rules":["broken_link"]}',
                    ),
                ),
            )
        return Message(role="assistant", content="审计完成")


class KnowledgeBaseServiceTests(unittest.TestCase):
    def test_clean_knowledge_base_has_no_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "index.md", "# Index\n\n[Python](notes/python.md)\n")
            _write(root / "notes/python.md", "# Python\n")

            report = _service(root).audit(
                ["missing_title", "broken_link", "duplicate_title"]
            )

            self.assertEqual(report.document_count, 2)
            self.assertEqual(report.issue_count, 0)

    def test_all_governance_rules_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / "index.md",
                "# Index\n\n[Existing](notes/a.md)\n[Missing](notes/missing.md)\n",
            )
            _write(
                root / "notes/a.md",
                "---\ntitle: Shared\nupdated: 2020-01-01\n---\nBody\n",
            )
            _write(root / "notes/b.md", "---\ntitle: Shared\n---\nBody\n")
            _write(root / "notes/no-title.md", "Body without a heading\n")

            report = _service(
                root,
                required_frontmatter=("tags",),
                reference_date=date(2026, 9, 11),
            ).audit()
            rules = {issue.rule for issue in report.issues}

            self.assertEqual(
                rules,
                {
                    "missing_title",
                    "missing_frontmatter",
                    "broken_link",
                    "duplicate_title",
                    "orphan",
                    "stale",
                },
            )
            self.assertEqual(report.summary["broken_link"], 1)

    def test_external_anchor_and_attachment_links_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / "index.md",
                "\n".join(
                    [
                        "# Index",
                        "[Web](https://example.com)",
                        "[Mail](mailto:me@example.com)",
                        "[Section](#details)",
                        "[PDF](files/report.pdf)",
                    ]
                ),
            )

            report = _service(root).audit(["broken_link"])

            self.assertEqual(report.issue_count, 0)

    def test_excluded_directories_and_root_escape_are_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "kb"
            _write(root / "index.md", "# Index\n\n[Outside](../outside.md)\n")
            _write(root / ".mini-harness/private.md", "# Private\n")
            _write(root.parent / "outside.md", "# Outside\n")

            report = _service(root).audit(["broken_link"])

            self.assertEqual(report.document_count, 1)
            self.assertEqual(report.issue_count, 1)
            self.assertEqual(report.issues[0].rule, "broken_link")

    def test_issue_ids_and_order_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "b.md", "no title\n")
            _write(root / "a.md", "no title\n")
            service = _service(root)

            first = service.audit(["missing_title"])
            second = service.audit(["missing_title"])

            self.assertEqual(first.issues, second.issues)

    def test_symlink_outside_root_is_not_scanned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "kb"
            _write(root / "index.md", "# Index\n")
            _write(base / "private.md", "# Private\n")
            try:
                (root / "linked.md").symlink_to(base / "private.md")
            except OSError as error:
                self.skipTest(f"当前系统不允许创建符号链接: {error}")

            report = _service(root).audit(["missing_title"])

            self.assertEqual(report.document_count, 1)

    def test_search_weights_title_metadata_and_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / "async.md",
                "---\ntitle: 异步并发控制\ntags: asyncio semaphore\n---\n"
                "# 异步并发控制\n\n使用信号量限制异步协程并发数量。\n",
            )
            _write(root / "other.md", "# 其他内容\n\n普通文件操作。\n")

            hits = _service(root).search("如何限制异步任务同时执行")

            self.assertTrue(hits)
            self.assertEqual(hits[0].path, "async.md")
            self.assertIn("title", hits[0].matched_in)
            self.assertIn("content", hits[0].matched_in)

    def test_search_rejects_empty_query_and_invalid_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = _service(Path(tmp))
            with self.assertRaises(ValueError):
                service.search(" ")
            with self.assertRaises(ValueError):
                service.search("Python", limit=21)


class KnowledgeBasePluginTests(unittest.TestCase):
    def test_agent_executes_kb_audit_through_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "notes" / "index.md", "# Index\n\n[Missing](missing.md)\n")
            previous = Path.cwd()
            os.chdir(root)
            ctx = Context()
            try:
                client = AuditClient()
                ctx.plugin(
                    headless_bundle,
                    BundleConfig(settings_document={}, client=client),
                )
                agent = ctx.require("agent")
                self.assertIsInstance(agent, Agent)
                agent.followup("审计断链")
                session = agent.run()

                results = [
                    event
                    for event in session.events
                    if event.type == "tool/result"
                    and event.data.get("call_id") == "audit-1"
                ]
                self.assertEqual(len(results), 1)
                self.assertIn('"broken_link"', results[0].data["content"])
                self.assertEqual(client.calls, 2)
            finally:
                ctx.dispose()
                os.chdir(previous)

    def test_consumer_registration_is_reversible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            os.chdir(tmp)
            ctx = Context()
            try:
                ctx.plugin(
                    headless_bundle,
                    BundleConfig(settings_document={}, client=FakeClient()),
                )
                agent = ctx.require("agent")
                prompt = ctx.require("prompt")
                self.assertIsInstance(agent, Agent)
                self.assertIsInstance(prompt, PromptAssembler)
                self.assertEqual(len(agent.tool_names), 27)
                self.assertIn("kb_audit", agent.tool_names)
                self.assertIn("kb_search", agent.tool_names)
                self.assertIn("kb_import_document", agent.tool_names)
                self.assertIn(
                    "knowledge-base:governance",
                    {section.name for section in prompt.sections},
                )

                consumer = next(
                    handle for handle in ctx.handles if handle.name == "kb_consumer"
                )
                consumer.dispose()

                self.assertNotIn("kb_audit", agent.tool_names)
                self.assertNotIn("kb_search", agent.tool_names)
                self.assertIn("kb_import_document", agent.tool_names)
                self.assertNotIn(
                    "knowledge-base:governance",
                    {section.name for section in prompt.sections},
                )
            finally:
                ctx.dispose()
                os.chdir(previous)

    def test_bundle_can_disable_knowledge_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            os.chdir(tmp)
            ctx = Context()
            try:
                ctx.plugin(
                    headless_bundle,
                    BundleConfig(
                        settings_document={},
                        client=FakeClient(),
                        enable_knowledge_base=False,
                    ),
                )
                agent = ctx.require("agent")
                self.assertIsInstance(agent, Agent)
                self.assertEqual(len(agent.tool_names), 24)
                self.assertNotIn("kb_audit", agent.tool_names)
                self.assertNotIn("kb_search", agent.tool_names)
                self.assertNotIn("knowledge_base", ctx.service_names)
            finally:
                ctx.dispose()
                os.chdir(previous)


def _service(
    root: Path,
    *,
    required_frontmatter: tuple[str, ...] = (),
    reference_date: date | None = None,
) -> KnowledgeBaseService:
    return KnowledgeBaseService(
        root=root,
        include_globs=("**/*.md", "**/*.txt"),
        exclude_dirs=(".git", ".mini-harness", ".agents", "node_modules"),
        required_frontmatter=required_frontmatter,
        stale_days=180,
        index_pages=("README.md", "index.md", "INDEX.md"),
        reference_date=reference_date,
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
