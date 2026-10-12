from __future__ import annotations

import json
import unittest

from mini_harness.client import Message, ToolCall
from mini_harness.meter import (
    TokenMeter,
    estimate_message,
    estimate_tokens,
    estimate_tools,
)


class TokenCountTests(unittest.TestCase):
    def test_uses_official_vocabulary(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("Hello!"), 2)
        self.assertEqual(estimate_tokens("<｜User｜>"), 1)

    def test_user_and_tool_messages_add_role_markers(self) -> None:
        text = "帮我审查个人知识库"
        self.assertEqual(
            estimate_message(Message(role="user", content=text)),
            estimate_tokens(text) + 1,
        )
        self.assertEqual(
            estimate_message(Message(role="tool", content=text, tool_call_id="c1")),
            estimate_tokens(text) + 2,
        )
        self.assertEqual(
            estimate_message(Message(role="system", content=text)),
            estimate_tokens(text),
        )

    def test_assistant_counts_reasoning_and_tool_calls(self) -> None:
        plain = Message(role="assistant", content="好的")
        self.assertEqual(estimate_message(plain), estimate_tokens("好的") + 2)

        call = ToolCall(id="c1", name="kb_audit", arguments='{"rules": ["orphan"]}')
        with_call = Message(
            role="assistant",
            content="",
            reasoning_content="先审计",
            tool_calls=(call,),
        )
        self.assertGreater(
            estimate_message(with_call),
            estimate_tokens("先审计") + estimate_tokens(call.arguments) + 2,
        )

    def test_measure_sums_messages_and_tools(self) -> None:
        messages = [
            Message(role="system", content="你是小知"),
            Message(role="user", content="Hello!"),
        ]
        schemas = [{"name": "calculator", "description": "计算", "parameters": {}}]
        measurement = TokenMeter(context_window=100).measure(messages, schemas)

        self.assertEqual(
            [item["tokens"] for item in measurement.messages],
            [estimate_message(m) for m in messages],
        )
        self.assertEqual(
            measurement.tools_tokens,
            estimate_tokens(json.dumps(schemas, ensure_ascii=False)),
        )
        self.assertEqual(estimate_tools([]), 0)
        self.assertEqual(
            measurement.total_tokens,
            measurement.message_tokens + measurement.tools_tokens,
        )


if __name__ == "__main__":
    unittest.main()
