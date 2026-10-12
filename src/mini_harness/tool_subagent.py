"""带受限工具集的子 Agent 运行器。

`subagent.run_subagent` 只做纯文本委派：它不给子 Agent 任何工具，因此
子 Agent 只能凭想象描述事实。本模块把工具执行接进同一条子 Agent 循环，
但工具集由调用方投影后传入——子 Agent 不能自行扩权，也不能再往下委派。

与 `Agent` 主循环的分工：主循环负责用户可见的一轮对话、事件广播和插件
接缝；这里只负责一次受限的、可结算的子任务运行，因此不引入 Context。
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence

from .client import ChatClient, Message, Tool, ToolCall
from .session import Session
from .subagent import CancellationSignal, SubagentResult


def run_tool_subagent(
    client: ChatClient,
    task: str,
    system_prompt: str,
    tools: Sequence[Tool],
    *,
    max_steps: int = 6,
    cancelled: CancellationSignal | None = None,
) -> SubagentResult:
    """独立 Session + 受限工具集的子 Agent；复用 SubagentResult 语义。

    与 `run_subagent` 相同的地方：父历史一个字都不带、失败保留部分文本、
    全程可被 `cancelled` 中断。不同的地方：每一步都把 `tools` 交给模型，
    并把工具结果以 role="tool" 回灌，直到模型给出不含工具调用的回答。
    """
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps 必须是正整数")
    if not task.strip():
        raise ValueError("task 必须是非空字符串")

    available = {tool.name: tool for tool in tools}
    session = Session()
    session.append("turn/start", {"turn": 1})
    session.append("user/message", {"content": task})

    partial = ""
    try:
        for _step in range(max_steps):
            if _interrupted(cancelled):
                return _settle(session, partial, "interrupted")
            reply = client.chat(
                [
                    Message(role="system", content=system_prompt),
                    *session.derive_messages(),
                ],
                list(tools),
            )
            print(f"[kb-repair-subagent:model] {reply!r}", file=sys.stderr, flush=True)
            if _interrupted(cancelled):
                return _settle(session, partial, "interrupted")
            session.append(
                "assistant/message",
                {
                    "content": reply.content,
                    **(
                        {"reasoning_content": reply.reasoning_content}
                        if reply.reasoning_content
                        else {}
                    ),
                    "tool_calls": [
                        {
                            "id": call.id,
                            "name": call.name,
                            "arguments": call.arguments,
                        }
                        for call in reply.tool_calls
                    ],
                },
            )
            if reply.content:
                partial = reply.content
            if not reply.tool_calls:
                if reply.content:
                    return _settle(session, reply.content, "completed")
                continue
            for call in reply.tool_calls:
                session.append(
                    "tool/call",
                    {
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                )
                content, is_error = _execute(available, call)
                session.append(
                    "tool/result",
                    {"call_id": call.id, "content": content, "is_error": is_error},
                )
        return _settle(session, partial, "max-steps")
    except Exception as error:  # noqa: BLE001 - 子 Agent 需要结算所有失败路径
        session.append(
            "turn/end", {"turn": 1, "reason": "error", "message": str(error)}
        )
        return SubagentResult(partial, "error", str(error))


def _execute(available: Mapping[str, Tool], call: ToolCall) -> tuple[str, bool]:
    """未授权工具、参数非法和执行异常一律配对成 tool 结果，不中断循环。"""
    tool = available.get(call.name)
    if tool is None:
        return (
            f"Error: 子 Agent 请求了未授权的工具 {call.name!r}；"
            f"可用工具为 {', '.join(sorted(available)) or '（无）'}",
            True,
        )
    try:
        arguments = json.loads(call.arguments or "{}")
        if not isinstance(arguments, dict):
            raise TypeError("工具参数必须是 JSON 对象")
        return tool.execute(arguments), False
    except Exception as error:  # noqa: BLE001 - 工具边界必须把失败配对成 result
        return f"工具执行出错: {error}", True


def _settle(session: Session, output: str, reason: str) -> SubagentResult:
    session.append("turn/end", {"turn": 1, "reason": reason})
    return SubagentResult(output=output, stop_reason=reason)


def _interrupted(cancelled: CancellationSignal | None) -> bool:
    return cancelled is not None and cancelled.is_set()
