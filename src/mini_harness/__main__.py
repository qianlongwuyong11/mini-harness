"""mini_harness 入口：一次性任务运行器与交互模式。

用法（在项目根目录运行已声明的命令行入口）：
    uv run mini-harness "你的任务"
    uv run mini-harness --interactive
    uv run mini-harness -i --resume .mini-harness/sessions/<id>.jsonl

也可以在本章 src 目录运行：
    uv run python -m mini_harness "你的任务"

对应官方 bundle/headless 的 runner 语义：
创建 Agent、把任务作为普通用户消息提交、等待完全停稳、
把最后一条 assistant 文本写入 stdout；最终 turn/end 完成 → 退出码 0，
否则 1。进程不打开任何监听端口。
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast
from uuid import uuid4

from .agent import Agent
from .bundle import BundleConfig, headless_bundle
from .capabilities import load_settings_document
from .client import Message
from .cordis import Context
from .meter import TokenMeter
from .persistence import JsonlStore
from .session import Session

SESSION_DIR = Path(".mini-harness") / "sessions"
INTERACTIVE_MAX_TURNS = 10_000


def build_agent(
    *,
    enable_console_questions: bool = True,
    checkpoint_flush: Callable[[Session], None] | None = None,
    session: Session | None = None,
    session_id: str | None = None,
    rag_runtime=None,
    memory_config=None,
    memory_sync_config=None,
) -> Agent:
    """创建 Context 并挂载 headless Bundle；能力由插件依赖自动接线。"""
    ctx = Context()
    ctx.plugin(
        headless_bundle,
        BundleConfig(
            settings_document=load_settings_document(),
            enable_console_questions=enable_console_questions,
            checkpoint_flush=checkpoint_flush,
            session=session,
            session_id=session_id,
            enable_rag=rag_runtime is not None,
            rag_runtime=rag_runtime,
            memory_config=memory_config,
            memory_sync_config=memory_sync_config,
        ),
    )
    return cast(Agent, ctx.require("agent"))


def run_task(task: str, session_file: str | Path | None = None) -> tuple[str, bool]:
    """跑一个一次性任务，返回 (最后一条 assistant 文本, 是否正常完成)。"""
    path = Path(session_file) if session_file is not None else _new_session_file()
    store = JsonlStore(path)
    agent = build_agent(checkpoint_flush=store.save)
    meter = TokenMeter(context_window=100_000)

    try:
        completed = _run_one_turn(agent, task, meter)
        final_text = _final_assistant_text(agent.session)
        return final_text, completed
    finally:
        try:
            store.save(agent.session)
            print(f"[persist] 会话已保存到 {store.path}", file=sys.stderr)
        finally:
            agent.close()


def run_interactive(
    *,
    session_file: str | Path | None = None,
    resume: str | Path | None = None,
    initial_task: str | None = None,
) -> int:
    """交互模式：复用同一 Agent 与 Session，持续接收用户输入。"""
    if resume is not None:
        path = Path(resume)
        store = JsonlStore(path)
        session = store.load()
        print(
            f"[persist] 已从 {store.path} 恢复 {len(session.events)} 条事件",
            file=sys.stderr,
        )
    else:
        path = Path(session_file) if session_file is not None else _new_session_file()
        store = JsonlStore(path)
        session = None

    agent = build_agent(checkpoint_flush=store.save, session=session)
    meter = TokenMeter(context_window=100_000)
    exit_code = 0

    print(f"[session] {store.path}", file=sys.stderr)
    print(
        "交互模式：输入问题继续对话，/exit、quit 或 Ctrl+D 退出",
        file=sys.stderr,
    )

    try:
        if initial_task:
            if not _run_one_turn(agent, initial_task, meter):
                exit_code = 1

        while True:
            try:
                user_input = input("> ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print(file=sys.stderr)
                break

            if not user_input:
                continue
            if user_input in {"/exit", "exit", "quit", "/quit"}:
                break

            if not _run_one_turn(agent, user_input, meter):
                exit_code = 1
    finally:
        try:
            store.save(agent.session)
            print(f"[persist] 会话已保存到 {store.path}", file=sys.stderr)
        finally:
            agent.close()

    return exit_code


def _run_one_turn(agent: Agent, text: str, meter: TokenMeter) -> bool:
    agent.followup(text)
    session = agent.run(max_turns=INTERACTIVE_MAX_TURNS)
    pressure = meter.pressure(meter.measure(_messages_of(session)))
    print(f"[meter] 上下文占用 {pressure.ratio:.1%}", file=sys.stderr)
    final_text = _final_assistant_text(session)
    if final_text:
        print(final_text)
    return _turn_completed(session)


def _final_assistant_text(session: Session) -> str:
    final_text = ""
    for message in session.derive_messages():
        if message.role == "assistant" and message.content:
            final_text = message.content
    return final_text


def _turn_completed(session: Session) -> bool:
    turn_ends = [event for event in session.events if event.type == "turn/end"]
    return bool(turn_ends and turn_ends[-1].data.get("reason") == "completed")


def _new_session_file() -> Path:
    """为任务分配新日志，避免下一次运行覆盖上一份会话。"""
    return SESSION_DIR / f"{uuid4().hex}.jsonl"


def _messages_of(session: Session) -> list[Message]:
    system = Message(role="system", content="(组装提示词)")
    return [system, *session.derive_messages()]


def _print_usage() -> None:
    print(
        "\n".join(
            [
                "用法:",
                '  mini-harness "你的任务"                    一次性任务',
                "  mini-harness --interactive                   交互模式，持续提问",
                '  mini-harness -i "先看一下项目结构"            交互模式并发送首条消息',
                "  mini-harness -i --resume <会话.jsonl>        恢复历史会话并继续",
                "  mini-harness -i --session <路径.jsonl>       指定新会话文件路径",
                "  mini-harness --rpc                           JSON-RPC 模式",
            ]
        ),
        file=sys.stderr,
    )


def _parse_cli(arguments: list[str]) -> tuple[bool, str | None, str | None, list[str]]:
    interactive = False
    resume_path: str | None = None
    session_path: str | None = None
    remaining: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in ("--interactive", "-i"):
            interactive = True
            index += 1
            continue
        if argument == "--resume":
            index += 1
            if index >= len(arguments):
                raise ValueError("--resume 需要指定会话文件路径")
            resume_path = arguments[index]
            index += 1
            continue
        if argument == "--session":
            index += 1
            if index >= len(arguments):
                raise ValueError("--session 需要指定会话文件路径")
            session_path = arguments[index]
            index += 1
            continue
        remaining.append(argument)
        index += 1
    return interactive, resume_path, session_path, remaining


def main() -> None:
    arguments = sys.argv[1:]
    if arguments == ["--rpc"]:
        _run_rpc()
        return

    if not arguments:
        _print_usage()
        sys.exit(2)

    try:
        interactive, resume_path, session_path, remaining = _parse_cli(arguments)
    except ValueError as error:
        print(f"错误: {error}", file=sys.stderr)
        sys.exit(2)

    if interactive:
        if resume_path and session_path:
            print("错误: --resume 与 --session 不能同时使用", file=sys.stderr)
            sys.exit(2)
        initial_task = " ".join(remaining).strip() or None
        exit_code = run_interactive(
            resume=resume_path,
            session_file=session_path,
            initial_task=initial_task,
        )
        sys.exit(exit_code)

    task = " ".join(remaining).strip()
    if not task:
        _print_usage()
        sys.exit(2)

    final_text, completed = run_task(task)
    if final_text:
        print(final_text)
    sys.exit(0 if completed else 1)


def _run_rpc() -> None:
    """stdio 上的 JSON-RPC line transport；不打开监听端口。"""
    store = JsonlStore(_new_session_file())
    print(f"[session] {store.path}", file=sys.stderr)
    agent = build_agent(
        enable_console_questions=False,
        checkpoint_flush=store.save,
    )
    dispatcher = agent.rpc_dispatcher
    if dispatcher is None:
        raise RuntimeError("JSON-RPC dispatcher 未组装")
    try:
        for line in sys.stdin:
            response = dispatcher.dispatch(line)
            print(json.dumps(response, ensure_ascii=False), flush=True)
    finally:
        try:
            store.save(agent.session)
            print(f"[persist] 会话已保存到 {store.path}", file=sys.stderr)
        finally:
            agent.close()


if __name__ == "__main__":
    main()
