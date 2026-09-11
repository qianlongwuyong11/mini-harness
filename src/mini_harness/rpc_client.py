"""通过 JSON-RPC 驱动独立 mini-harness Agent 进程的命令行客户端。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from threading import Thread
from typing import Any, TextIO


class RpcClientError(RuntimeError):
    """Agent 子进程或 JSON-RPC 协议错误。"""


class RpcProcessClient:
    """管理 Agent 子进程，并在其 stdin/stdout 上同步收发 JSON-RPC。"""

    def __init__(self, command: Sequence[str] | None = None) -> None:
        self.command = list(
            command or (sys.executable, "-m", "mini_harness", "--rpc")
        )
        self._process: subprocess.Popen[str] | None = None
        self._next_id = 1
        self._stderr_thread: Thread | None = None

    def start(self) -> None:
        if self._process is not None:
            raise RpcClientError("Agent 进程已经启动")
        environment = os.environ.copy()
        environment.setdefault("PYTHONIOENCODING", "utf-8")
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        assert self._process.stderr is not None
        self._stderr_thread = Thread(
            target=_forward_stderr,
            args=(self._process.stderr,),
            daemon=True,
            name="mini-harness-rpc-stderr",
        )
        self._stderr_thread.start()

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        process = self._require_process()
        if process.poll() is not None:
            raise RpcClientError(
                f"Agent 进程已退出，退出码为 {process.returncode}"
            )
        request_id = self._next_id
        self._next_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }

        assert process.stdin is not None
        assert process.stdout is not None
        try:
            process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise RpcClientError("无法向 Agent 进程发送请求") from error

        line = process.stdout.readline()
        if not line:
            process.wait()
            raise RpcClientError(
                f"Agent 进程未返回响应，退出码为 {process.returncode}"
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise RpcClientError(f"Agent 返回了非法 JSON: {line.rstrip()}") from error
        if not isinstance(response, dict) or response.get("jsonrpc") != "2.0":
            raise RpcClientError(f"Agent 返回了非法 JSON-RPC 响应: {response!r}")
        if response.get("id") != request_id:
            raise RpcClientError(
                f"响应 id 不匹配：期待 {request_id!r}，实际 {response.get('id')!r}"
            )
        if "error" in response:
            error = response["error"]
            if isinstance(error, dict):
                code = error.get("code", "unknown")
                message = error.get("message", "未知错误")
                raise RpcClientError(f"JSON-RPC 错误 {code}: {message}")
            raise RpcClientError(f"Agent 返回了非法错误响应: {error!r}")
        if "result" not in response:
            raise RpcClientError(f"Agent 响应缺少 result: {response!r}")
        return response["result"]

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        self._process = None

        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1)
            self._stderr_thread = None

    def __enter__(self) -> RpcProcessClient:
        self.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _require_process(self) -> subprocess.Popen[str]:
        if self._process is None:
            raise RpcClientError("Agent 进程尚未启动")
        return self._process


def run_client(initial_task: str | None = None) -> int:
    """启动 Agent 子进程并提供持续对话的命令行界面。"""
    try:
        with RpcProcessClient() as client:
            print("mini-harness JSON-RPC 客户端")
            print("Agent 已在独立进程中启动。输入 /help 查看命令。")
            if initial_task:
                _run_agent(client, initial_task)

            while True:
                try:
                    text = input("you> ").strip()
                except EOFError:
                    break
                except KeyboardInterrupt:
                    print()
                    break
                if not text:
                    continue
                if text.lower() in {"/exit", "/quit", "exit", "quit"}:
                    break
                if text == "/help":
                    _print_repl_help()
                    continue
                if text == "/settings":
                    settings = client.request(
                        "settings.get", {"namespace": "agent"}
                    )
                    print(json.dumps(settings, ensure_ascii=False, indent=2))
                    continue
                if text.startswith("/plan "):
                    value = text.removeprefix("/plan ").strip().lower()
                    if value not in {"on", "off"}:
                        print("用法：/plan on 或 /plan off", file=sys.stderr)
                        continue
                    result = client.request("plan.set", {"active": value == "on"})
                    print(json.dumps(result, ensure_ascii=False))
                    continue

                _run_agent(client, text)
        return 0
    except (OSError, RpcClientError) as error:
        print(f"客户端错误: {error}", file=sys.stderr)
        return 1


def _run_agent(client: RpcProcessClient, task: str) -> None:
    result = client.request("agent.run", {"task": task})
    if not isinstance(result, dict):
        raise RpcClientError(f"agent.run 返回值不是对象: {result!r}")
    output = result.get("output")
    if isinstance(output, str) and output:
        print(f"agent> {output}")
    else:
        print("agent> （本轮没有文本输出）")


def _forward_stderr(stream: TextIO) -> None:
    for line in stream:
        print(f"[agent] {line.rstrip()}", file=sys.stderr)


def _print_repl_help() -> None:
    print(
        "\n".join(
            [
                "可用命令：",
                "  /settings  查看 Agent 配置",
                "  /plan on   开启计划模式",
                "  /plan off  关闭计划模式",
                "  /exit      关闭 Agent 子进程并退出",
                "其他输入会作为 agent.run 请求发送。",
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="启动独立 Agent 进程并通过 JSON-RPC 持续对话"
    )
    parser.add_argument("task", nargs="*", help="可选的首条任务")
    arguments = parser.parse_args()
    initial_task = " ".join(arguments.task).strip() or None
    raise SystemExit(run_client(initial_task))


if __name__ == "__main__":
    main()
