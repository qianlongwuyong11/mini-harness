from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from threading import Event

from mini_harness.bundle import BundleConfig, headless_bundle
from mini_harness.client import Message, Tool
from mini_harness.cordis import Context
from mini_harness.document_import import DocumentImportService
from mini_harness.knowledge_base import KnowledgeBaseService
from mini_harness.knowledge_watch import KnowledgeWatchService, WatchRecord
from mini_harness.sandbox import SandboxPolicy, WORKSPACE_WRITE


class FakeClient:
    MODEL = "fake"

    def chat(
        self, _messages: list[Message], _tools: list[Tool] | None = None
    ) -> Message:
        return Message(role="assistant", content="unused")


class KnowledgeWatchServiceTests(unittest.TestCase):
    def test_stable_file_is_imported_archived_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _write(
                workspace / "inbox" / "topics" / "agent.txt",
                "插件生命周期与服务装配。",
            )
            _write(workspace / "notes" / "index.md", "# Index\n")
            records: list[WatchRecord] = []
            watcher = _watcher(workspace, reporter=records.append)

            self.assertEqual(watcher.scan_once(), ())
            emitted = watcher.scan_once()

            note = workspace / "notes" / "topics" / "agent.md"
            archived = workspace / "archive" / "topics" / "agent.txt"
            self.assertTrue(note.is_file())
            self.assertTrue(archived.is_file())
            self.assertFalse((workspace / "inbox" / "topics" / "agent.txt").exists())
            self.assertEqual(len(emitted), 1)
            self.assertEqual(emitted[0].status, "imported")
            self.assertEqual(emitted[0].destination, "topics/agent.md")
            self.assertEqual(emitted[0].audit_summary, {"orphan": 1})
            self.assertIn(
                'source: "archive/topics/agent.txt"',
                note.read_text(encoding="utf-8"),
            )
            self.assertEqual(records, list(emitted))

    def test_changed_file_must_be_stable_again_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            source = workspace / "inbox" / "draft.txt"
            _write(source, "first")
            watcher = _watcher(workspace)

            watcher.scan_once()
            _write(source, "second version")
            self.assertEqual(watcher.scan_once(), ())
            self.assertTrue(source.exists())
            watcher.scan_once()

            self.assertFalse(source.exists())
            content = (workspace / "notes" / "draft.md").read_text(encoding="utf-8")
            self.assertIn("second version", content)

    def test_invalid_document_moves_to_failed_without_stopping_others(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _write(workspace / "inbox" / "bad.json", "{not-json")
            _write(workspace / "inbox" / "good.txt", "valid")
            watcher = _watcher(workspace)

            watcher.scan_once()
            emitted = watcher.scan_once()

            by_source = {record.source: record for record in emitted}
            self.assertEqual(by_source["inbox/bad.json"].status, "failed")
            self.assertIn("JSONDecodeError", by_source["inbox/bad.json"].error or "")
            self.assertEqual(by_source["inbox/good.txt"].status, "imported")
            self.assertTrue((workspace / "failed" / "bad.json").is_file())
            self.assertTrue((workspace / "notes" / "good.md").is_file())

    def test_temporary_hidden_and_unsupported_files_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            sources = (
                workspace / "inbox" / "~$writing.docx",
                workspace / "inbox" / ".hidden.txt",
                workspace / "inbox" / "copy.part",
                workspace / "inbox" / "legacy.doc",
            )
            for source in sources:
                _write(source, "ignored")
            watcher = _watcher(workspace, stable_scans=1)

            self.assertEqual(watcher.scan_once(), ())
            self.assertTrue(all(source.exists() for source in sources))

    def test_archive_collision_gets_a_unique_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _write(workspace / "archive" / "same.txt", "older")
            _write(workspace / "inbox" / "same.txt", "newer")
            watcher = _watcher(workspace, stable_scans=1)

            emitted = watcher.scan_once()

            self.assertEqual(emitted[0].stored_source, "archive/same-1.txt")
            self.assertTrue((workspace / "archive" / "same-1.txt").is_file())
            note = (workspace / "notes" / "same.md").read_text(encoding="utf-8")
            self.assertIn('source: "archive/same-1.txt"', note)

    def test_background_thread_detects_new_file_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            imported = Event()

            def report(record: WatchRecord) -> None:
                if record.status == "imported":
                    imported.set()

            watcher = _watcher(
                workspace,
                stable_scans=1,
                reporter=report,
            )
            stop = watcher.start()
            try:
                _write(workspace / "inbox" / "background.txt", "detected")
                self.assertTrue(imported.wait(timeout=1))
                self.assertTrue((workspace / "notes" / "background.md").is_file())
            finally:
                stop()
            self.assertFalse(watcher.running)

    def test_rejects_overlapping_watch_and_archive_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            with self.assertRaises(ValueError):
                _watcher(workspace, archive_dir=Path("inbox/archive"))


class KnowledgeWatchPluginTests(unittest.TestCase):
    def test_bundle_starts_watcher_creates_layout_and_stops_on_dispose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            os.chdir(tmp)
            ctx = Context()
            try:
                ctx.plugin(
                    headless_bundle,
                    BundleConfig(settings_document={}, client=FakeClient()),
                )
                watcher = ctx.require("knowledge_watch")
                self.assertIsInstance(watcher, KnowledgeWatchService)
                assert isinstance(watcher, KnowledgeWatchService)
                self.assertTrue(watcher.running)
                for name in ("inbox", "notes", "archive", "failed"):
                    self.assertTrue(Path(name).is_dir())

                provider = next(
                    handle
                    for handle in ctx.handles
                    if handle.name == "knowledge_watch_provider"
                )
                provider.dispose()

                self.assertNotIn("knowledge_watch", ctx.service_names)
                self.assertFalse(watcher.running)
            finally:
                ctx.dispose()
                os.chdir(previous)

    def test_watcher_can_be_disabled_without_disabling_import_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            os.chdir(tmp)
            ctx = Context()
            try:
                ctx.plugin(
                    headless_bundle,
                    BundleConfig(
                        settings_document={
                            "agent": {"kb_watch_enabled": False}
                        },
                        client=FakeClient(),
                    ),
                )

                self.assertNotIn("knowledge_watch", ctx.service_names)
                tools = ctx.require("tools")
                self.assertIsNotNone(tools.get("kb_import_document"))
                self.assertFalse(Path("inbox").exists())
            finally:
                ctx.dispose()
                os.chdir(previous)


def _watcher(
    workspace: Path,
    *,
    archive_dir: Path = Path("archive"),
    stable_scans: int = 2,
    reporter=None,
) -> KnowledgeWatchService:
    knowledge_base = KnowledgeBaseService(
        root=workspace / "notes",
        include_globs=("**/*.md", "**/*.txt"),
        exclude_dirs=(".git", ".mini-harness", ".agents", "node_modules"),
        required_frontmatter=(),
        stale_days=180,
        index_pages=("README.md", "index.md", "INDEX.md"),
    )

    def writer(arguments: dict[str, object]) -> str:
        path = Path(str(arguments["path"]))
        path.write_text(str(arguments["content"]), encoding="utf-8")
        return "written"

    importer = DocumentImportService(
        workspace=workspace,
        knowledge_base=knowledge_base,
        writer=writer,
        import_dir=".",
        max_bytes=1024 * 1024,
    )
    return KnowledgeWatchService(
        workspace=workspace,
        importer=importer,
        sandbox=SandboxPolicy(WORKSPACE_WRITE, workspace),
        watch_dir=Path("inbox"),
        archive_dir=archive_dir,
        failed_dir=Path("failed"),
        interval_seconds=0.01,
        stable_scans=stable_scans,
        reporter=reporter or (lambda _record: None),
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
