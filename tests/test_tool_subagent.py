from __future__ import annotations

import unittest

from mini_harness.client import Message, Tool, ToolCall
from mini_harness.tool_subagent import run_tool_subagent


class ScriptedClient:
    MODEL = "scripted-fake"

    def __init__(self, replies: list[Message], repeat_last: bool = False) -> None:
        self._replies = list(replies)
        self._repeat_last = repeat_last
        self.requests: list[list[Message]] = []
        self.offered: tuple[str, ...] = ()

    def chat(
        self, messages: list[Message], tools: list[Tool] | None = None
    ) -> Message:
        self.requests.append(list(messages))
        self.offered = tuple(tool.name for tool in tools or ())
        if len(self._replies) > 1 or not self._repeat_last:
            return self._replies.pop(0)
        return self._replies[0]


class AlwaysCancelled:
    def is_set(self) -> bool:
        return True


def _echo_tool() -> Tool:
    return Tool(
        "echo",
        "回显 value。",
        {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        lambda arguments: f"echo:{arguments['value']}",
    )


def _tool_messages(request: list[Message]) -> list[Message]:
    return [message for message in request if message.role == "tool"]


class ToolSubagentTests(unittest.TestCase):
    def test_tool_is_called_and_result_is_fed_back(self) -> None:
        client = ScriptedClient(
            [
                Message(
                    role="assistant",
                    tool_calls=(ToolCall("call-1", "echo", '{"value":"hi"}'),),
                ),
                Message(role="assistant", content="调查完成"),
            ]
        )

        result = run_tool_subagent(
            client, "回显 hi", "system", (_echo_tool(),), max_steps=4
        )

        self.assertEqual(result.stop_reason, "completed")
        self.assertEqual(result.output, "调查完成")
        self.assertEqual(client.offered, ("echo",))
        replayed = _tool_messages(client.requests[1])
        self.assertEqual(len(replayed), 1)
        self.assertEqual(replayed[0].content, "echo:hi")
        self.assertEqual(replayed[0].tool_call_id, "call-1")

    def test_unauthorized_tool_is_paired_instead_of_raised(self) -> None:
        client = ScriptedClient(
            [
                Message(
                    role="assistant",
                    tool_calls=(ToolCall("call-1", "edit", "{}"),),
                ),
                Message(role="assistant", content="改用只读工具"),
            ]
        )

        result = run_tool_subagent(
            client, "越权尝试", "system", (_echo_tool(),), max_steps=4
        )

        self.assertEqual(result.stop_reason, "completed")
        replayed = _tool_messages(client.requests[1])
        self.assertEqual(len(replayed), 1)
        assert replayed[0].content is not None
        self.assertIn("未授权的工具", replayed[0].content)

    def test_invalid_arguments_are_settled_as_tool_error(self) -> None:
        client = ScriptedClient(
            [
                Message(
                    role="assistant",
                    tool_calls=(ToolCall("call-1", "echo", "not-json"),),
                ),
                Message(role="assistant", content="重试后完成"),
            ]
        )

        result = run_tool_subagent(
            client, "参数非法", "system", (_echo_tool(),), max_steps=4
        )

        self.assertEqual(result.stop_reason, "completed")
        replayed = _tool_messages(client.requests[1])
        assert replayed[0].content is not None
        self.assertIn("工具执行出错", replayed[0].content)

    def test_max_steps_keeps_partial_text(self) -> None:
        client = ScriptedClient(
            [
                Message(
                    role="assistant",
                    content="仍在阅读",
                    tool_calls=(ToolCall("call-1", "echo", '{"value":"x"}'),),
                )
            ],
            repeat_last=True,
        )

        result = run_tool_subagent(
            client, "永不结束", "system", (_echo_tool(),), max_steps=3
        )

        self.assertEqual(result.stop_reason, "max-steps")
        self.assertEqual(result.output, "仍在阅读")
        self.assertEqual(len(client.requests), 3)

    def test_cancellation_returns_before_calling_the_model(self) -> None:
        client = ScriptedClient([Message(role="assistant", content="不应被调用")])

        result = run_tool_subagent(
            client,
            "已取消",
            "system",
            (_echo_tool(),),
            max_steps=3,
            cancelled=AlwaysCancelled(),
        )

        self.assertEqual(result.stop_reason, "interrupted")
        self.assertEqual(client.requests, [])

    def test_client_failure_is_settled_as_error(self) -> None:
        class BrokenClient:
            MODEL = "broken-fake"

            def chat(
                self, _messages: list[Message], _tools: list[Tool] | None = None
            ) -> Message:
                raise RuntimeError("上游不可用")

        result = run_tool_subagent(
            BrokenClient(), "任务", "system", (_echo_tool(),), max_steps=2
        )

        self.assertEqual(result.stop_reason, "error")
        self.assertEqual(result.diagnostic, "上游不可用")

    def test_invalid_max_steps_is_rejected(self) -> None:
        client = ScriptedClient([Message(role="assistant", content="x")])
        with self.assertRaises(ValueError):
            run_tool_subagent(client, "任务", "system", (), max_steps=0)


if __name__ == "__main__":
    unittest.main()
