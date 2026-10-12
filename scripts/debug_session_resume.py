#!/usr/bin/env python3
"""调试 Session 加载、Agent 完成回复以及 JSONL 追加持久化。

流程：

    原 Session JSONL
      -> 复制为独立调试文件
      -> JsonlStore.load() / Session.from_log()
      -> 注入 Agent
      -> Agent 完成一轮本地确定性回复
      -> JsonlStore.save() 追加回复和收尾事件
      -> 再次 load() 校验磁盘与内存一致

脚本不会修改原 Session，也不会请求真实模型。
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from mini_harness.agent import Agent  # noqa: E402
from mini_harness.bundle import BundleConfig, headless_bundle  # noqa: E402
from mini_harness.capabilities import load_settings_document  # noqa: E402
from mini_harness.client import Message, Tool  # noqa: E402
from mini_harness.cordis import Context  # noqa: E402
from mini_harness.persistence import JsonlStore  # noqa: E402
from mini_harness.session import Session, SessionEvent  # noqa: E402

DEFAULT_PROMPT = "这是一条用于调试 Session 续写和持久化的用户消息。"
DEFAULT_REPLY = "调试回复已经完成，并将追加到 Session 持久化文件。"


class LocalReplyClient:
    """返回固定回复，不访问网络；MODEL 保持项目的生产模型标识。"""

    MODEL = "deepseek-chat"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.requests = 0

    def chat(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
    ) -> Message:
        del messages, tools
        self.requests += 1
        return Message(role="assistant", content=self.reply)


def create_debug_copy(source: Path, output_file: Path | None) -> Path:
    """复制真实 Session；后续所有追加只发生在调试副本。"""
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Session 文件不存在: {source}")

    if output_file is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output_file = (
            REPO_ROOT
            / ".mini-harness"
            / "debug-session-persistence"
            / f"{source.stem}-{stamp}.jsonl"
        )
    target = output_file.resolve()
    if target == source:
        raise ValueError("输出文件不能与原 Session 相同")
    if target.exists():
        raise FileExistsError(f"拒绝覆盖已有调试文件: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def load_session(session_file: Path) -> tuple[JsonlStore, Session]:
    """调试入口一：持久化文件 -> 内存 Session。"""
    store = JsonlStore(session_file)

    # 建议断点：persistence.py:JsonlStore.load、session.py:Session.from_log。
    session = store.load()
    return store, session


def build_agent_with_session(
    store: JsonlStore,
    session: Session,
    client: LocalReplyClient,
) -> Agent:
    """调试入口二：通过当前项目 Bundle 把 Session 注入 Agent。"""
    ctx = Context()
    ctx.plugin(
        headless_bundle,
        BundleConfig(
            settings_document=load_settings_document(),
            enable_console_questions=False,
            checkpoint_flush=store.save,
            client=client,
            session=session,
        ),
    )

    # 建议断点：bundle.py:agent_provider、agent.py:Agent.__init__。
    agent = cast(Agent, ctx.require("agent"))
    if agent.session is not session:
        agent.close()
        raise AssertionError("Agent 没有持有刚恢复的 Session 实例")
    return agent


def persisted_event_count(path: Path) -> int:
    """JSONL 第一行是 header，其余每行是一条已持久化事件。"""
    return len(path.read_text(encoding="utf-8").splitlines()) - 1


def print_events(title: str, events: tuple[SessionEvent, ...]) -> None:
    print(f"\n{title} ({len(events)} events)")
    for event in events:
        data = json.dumps(dict(event.data), ensure_ascii=False, separators=(",", ":"))
        if len(data) > 240:
            data = data[:237] + "..."
        print(f"  #{event.id:04d} {event.type:<20} {data}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="加载 Session，完成一条本地回复，并追加写入调试 JSONL 副本。"
    )
    parser.add_argument("session_file", type=Path, help="作为起点的 Session JSONL")
    parser.add_argument("--output-file", type=Path, help="调试副本路径；默认自动生成")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="本轮用户消息")
    parser.add_argument("--reply", default=DEFAULT_REPLY, help="本地构造的完成回复")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_file = args.session_file.resolve()
    debug_file = create_debug_copy(source_file, args.output_file)
    original_bytes = debug_file.read_bytes()

    store, session = load_session(debug_file)
    original_event_count = len(session.events)
    client = LocalReplyClient(args.reply)
    agent = build_agent_with_session(store, session, client)

    try:
        print(f"source file : {source_file}")
        print(f"debug file  : {debug_file}")
        print(f"agent source: {inspect.getsourcefile(Agent)}")
        print(f"loaded      : {original_event_count} events")
        print(f"same object : {agent.session is session}")

        agent.followup(args.prompt)

        # 调试入口三：进入 Agent.run，观察回复如何写入内存 Session。
        agent.run(max_turns=10_000)
        in_memory_events = agent.session.events
        persisted_before_final_save = persisted_event_count(debug_file)
        pending_events = in_memory_events[persisted_before_final_save:]
        pending_types = [event.type for event in pending_events]
        expected_pending_types = ["assistant/message", "step/end", "turn/end"]
        if pending_types != expected_pending_types:
            raise AssertionError(
                f"最终待持久化事件为 {pending_types}，预期 {expected_pending_types}"
            )

        print_events(
            "本轮写入内存的全部事件",
            in_memory_events[original_event_count:],
        )
        print_events(
            "回复完成后仍待最终持久化的事件",
            pending_events,
        )

        # 调试入口四：此时回复已经完成并写入 Session；下面执行追加落盘。
        store.save(agent.session)
    finally:
        agent.close()

    final_bytes = debug_file.read_bytes()
    if not final_bytes.startswith(original_bytes):
        raise AssertionError("持久化文件没有保持原内容前缀，违反 append-only")

    reloaded = JsonlStore(debug_file).load()
    if reloaded.events != session.events:
        raise AssertionError("最终磁盘事件与内存 Session 不一致")
    if persisted_event_count(debug_file) != len(session.events):
        raise AssertionError("JSONL 事件行数与 Session 事件数不一致")

    print("\nPersisted")
    print(f"  before events : {original_event_count}")
    print(f"  after events  : {len(session.events)}")
    print(f"  appended bytes: {len(final_bytes) - len(original_bytes)}")
    print(f"  model requests: {client.requests} (local only)")
    print(f"  output file   : {debug_file}")
    print("\nPASS: 回复完成后的 Session 事件已追加持久化，并可完整重放。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
