"""Token 计量：感知上下文压力（第 09 章首次实现）。

对应官方 packages/llm/token-meter。官方用「4 字符 ≈ 1 token」的启发式，
这里改用 DeepSeek 官方离线分词器（API 文档提供的 deepseek_v4_tokenizer.zip
中的 tokenizer.json），并按其 chat template 的角色标记拼接消息后计数。
服务端对工具清单等字段的拼接方式未公开，计费仍以 API 返回的 usage 为准。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from .client import Message

TOKENIZER_FILE = (
    Path(__file__).parent / "assets" / "deepseek_v4_tokenizer" / "tokenizer.json"
)
# DeepSeek 官方适配器默认上下文容量：1e6 token
DEFAULT_CONTEXT_WINDOW = 1_000_000
# 压力阈值：占用 >= 80% 触发（对齐官方压缩包 DEFAULT_THRESHOLD_RATIO=.8）
PRESSURE_THRESHOLD = 0.8


@dataclass(frozen=True)
class Measurement:
    """一次计量快照。"""

    messages: list[dict[str, Any]]
    message_tokens: int
    tools_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class Pressure:
    """压力换算：总量、容量、占比、是否越线。"""

    total_tokens: int
    context_window: int
    ratio: float
    over_threshold: bool


@lru_cache(maxsize=1)
def _tokenizer() -> Tokenizer:
    return Tokenizer.from_file(str(TOKENIZER_FILE))


def estimate_tokens(text: str) -> int:
    """官方分词器对一段文本的 token 数；文本里的角色标记按词表识别为特殊 token。"""
    if not text:
        return 0
    return len(_tokenizer().encode(text, add_special_tokens=False).ids)


def render_message(message: Message) -> str:
    """按官方 chat template 把一条消息渲染成模型实际看到的文本。
    reasoning_content 会随请求回传（见 DeepSeekClient._wire_message），一并计入。"""
    content = message.content or ""
    if message.role == "user":
        return "<｜User｜>" + content
    if message.role == "tool":
        return "<｜tool▁output▁begin｜>" + content + "<｜tool▁output▁end｜>"
    if message.role == "assistant":
        calls = "\n".join(
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>"
            f"{call.name}\n```json\n{call.arguments}\n```<｜tool▁call▁end｜>"
            for call in message.tool_calls
        )
        if calls:
            calls = "<｜tool▁calls▁begin｜>" + calls + "<｜tool▁calls▁end｜>"
        reasoning = message.reasoning_content or ""
        return "<｜Assistant｜>" + reasoning + content + calls + "<｜end▁of▁sentence｜>"
    return content


def estimate_message(message: Message) -> int:
    """一条消息的 token 数 = 渲染后文本（内容 + 角色标记）的分词长度。"""
    return estimate_tokens(render_message(message))


def estimate_tools(tools: list[Any]) -> int:
    """工具 schema 序列化后的分词长度。
    工具清单每次请求都要随 system 一起发送，是实打实的输入成本。"""
    if not tools:
        return 0
    return estimate_tokens(json.dumps(tools, ensure_ascii=False))


class TokenMeter:
    """计量服务。对应官方单例 ctx.tokenMeter。"""

    def __init__(self, context_window: int = DEFAULT_CONTEXT_WINDOW) -> None:
        self.context_window = context_window

    def measure(self, messages: list[Message], tools: list[Any] | None = None) -> Measurement:
        """计量一段会话：消息面 + 工具的 envelope。"""
        encodings = _tokenizer().encode_batch(
            [render_message(m) for m in messages], add_special_tokens=False
        )
        metered = [
            {
                "role": m.role,
                "length": len(m.content or ""),
                "tokens": len(encoding.ids),
            }
            for m, encoding in zip(messages, encodings)
        ]
        message_tokens = sum(item["tokens"] for item in metered)
        tools_tokens = estimate_tools(tools or [])
        return Measurement(
            messages=metered,
            message_tokens=message_tokens,
            tools_tokens=tools_tokens,
            total_tokens=message_tokens + tools_tokens,
        )

    def pressure(self, measurement: Measurement) -> Pressure:
        """把计量换算成压力读数。判定（是否压缩）留给消费方——
        计量与压缩解耦是官方明确的设计（token-meter 不依赖压缩包）。"""
        ratio = measurement.total_tokens / self.context_window
        return Pressure(
            total_tokens=measurement.total_tokens,
            context_window=self.context_window,
            ratio=ratio,
            over_threshold=ratio >= PRESSURE_THRESHOLD,
        )
