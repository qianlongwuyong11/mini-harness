from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from mini_harness.agent import Agent
from mini_harness.bundle import BundleConfig, headless_bundle
from mini_harness.client import Message, Tool
from mini_harness.cordis import Context
from mini_harness.prompt import PromptAssembler


class FakeClient:
    MODEL = "fake"

    def chat(
        self, _messages: list[Message], _tools: list[Tool] | None = None
    ) -> Message:
        return Message(role="assistant", content="unused")


class RagBootstrapTests(unittest.TestCase):
    def test_rag_is_disabled_by_default(self) -> None:
        config = BundleConfig(
            settings_document={},
            client=FakeClient(),
            enable_knowledge_base=False,
        )
        self.assertFalse(config.enable_rag)

        with tempfile.TemporaryDirectory() as tmp:
            ctx = Context()
            try:
                ctx.plugin(headless_bundle, config)
                self.assertIsInstance(ctx.require("agent"), Agent)
                self.assertNotIn("rag_plugin", {handle.name for handle in ctx.handles})
                self.assertNotIn("paper_retrieval", ctx.service_names)
            finally:
                ctx.dispose()

    def test_enabling_rag_installs_lazy_services_and_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            ctx: Context | None = None
            try:
                # Keep filesystem-based plugins isolated from the repository.
                os.chdir(tmp)
                ctx = Context()
                ctx.plugin(
                    headless_bundle,
                    BundleConfig(
                        settings_document={},
                        client=FakeClient(),
                        enable_knowledge_base=False,
                        enable_rag=True,
                    ),
                )
                agent = ctx.require("agent")
                prompt = ctx.require("prompt")
                self.assertIsInstance(agent, Agent)
                self.assertIsInstance(prompt, PromptAssembler)

                rag = next(
                    handle for handle in ctx.handles if handle.name == "rag_plugin"
                )
                self.assertEqual(rag.state, "active")
                self.assertIn("paper_catalog", ctx.service_names)
                self.assertTrue(Path(".mini-harness/rag/vector-catalog.sqlite3").exists())
                self.assertTrue({'list_papers','search_papers','read_section'} <= set(agent.tool_names))
                self.assertIn('本轮证据', prompt.render({'name':'test'}))
                rag.dispose()
                self.assertFalse({'list_papers','search_papers','read_section'} & set(agent.tool_names))
                self.assertNotIn('本轮证据', prompt.render({'name':'test'}))
                self.assertNotIn("paper_catalog", ctx.service_names)
                self.assertNotIn("paper_retrieval", ctx.service_names)
                self.assertFalse(any(name.startswith("paper_") for name in agent.tool_names))
            finally:
                if ctx is not None:
                    ctx.dispose()
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
