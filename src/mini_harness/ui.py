"""Local browser UI. One worker owns the Agent and all session mutations."""
from __future__ import annotations

import argparse
import copy
import json
import re
import secrets
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4


class Workspace:
    def __init__(self, root: Path, factory=None):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.factory = factory
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ui-agent")
        self.lock = threading.Lock()
        self.agent = None
        self.store = None
        self.unsubscribe = None
        self.state = dict(busy=False, session_id=None, messages=[], activity=[], error=None)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def sessions(self):
        result = []
        for path in sorted(self.root.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            if re.fullmatch(r"[0-9a-f]{32}", path.stem):
                result.append(dict(id=path.stem, modified=path.stat().st_mtime))
        return result

    def submit(self, action, value):
        if action not in {"chat", "session"}:
            raise ValueError("未知操作")
        if action == "chat" and (not isinstance(value, str) or not value.strip() or len(value) > 32000):
            raise ValueError("请输入 1–32000 字的消息")
        if action == "session" and value is not None:
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value):
                raise ValueError("会话编号无效")
            if not (self.root / f"{value}.jsonl").is_file():
                raise ValueError("会话文件不存在")
        with self.lock:
            if self.state["busy"]:
                raise RuntimeError("当前任务仍在执行，请完成后再操作")
            self.state.update(busy=True, error=None, activity=[])
        self.pool.submit(self._perform, action, value)

    def _publish_messages(self):
        if self.agent is None:
            return
        messages = [dict(role=m.role, content=m.content) for m in self.agent.session.derive_messages()
                    if m.role in {"user", "assistant"} and m.content]
        with self.lock:
            self.state["messages"] = messages

    def _event(self, event):
        labels = {"tool/call": "调用工具", "tool/result": "工具返回", "step/start": "分析与执行", "turn/end": "本轮结束"}
        if event.type in labels:
            name = event.data.get("name", "")
            with self.lock:
                self.state["activity"].append(labels[event.type] + (f" · {name}" if name else ""))
                self.state["activity"] = self.state["activity"][-30:]
        if event.type in {"user/message", "assistant/message"}:
            self._publish_messages()

    def _release(self):
        if self.agent is not None:
            # A failed save must prevent switching away from the live session.
            self.store.save(self.agent.session)
            if self.unsubscribe:
                self.unsubscribe()
            try:
                self.agent.close()
            finally:
                self.agent = None
                self.unsubscribe = None

    def _open(self, session_id=None):
        from .persistence import JsonlStore
        from .session import Session
        from .__main__ import build_agent
        self._release()
        identifier = session_id or uuid4().hex
        store = JsonlStore(self.root / f"{identifier}.jsonl")
        session = store.load() if session_id else Session()
        factory = self.factory or build_agent
        agent = factory(session=session, checkpoint_flush=store.save, enable_console_questions=False)
        self.agent, self.store = agent, store
        self.unsubscribe = session.subscribe(self._event)
        with self.lock:
            self.state.update(session_id=identifier, messages=[], activity=[])
        self._publish_messages()

    def _perform(self, action, value):
        try:
            if action == "session":
                self._open(value)
            else:
                if self.agent is None:
                    self._open()
                self.agent.followup(value.strip())
                try:
                    self.agent.run(max_turns=10000)
                finally:
                    self.store.save(self.agent.session)
                    self._publish_messages()
        except Exception as error:
            with self.lock:
                self.state["error"] = f"{type(error).__name__}: {error}"
        finally:
            with self.lock:
                self.state["busy"] = False

    def close(self):
        try:
            self.pool.submit(self._release).result()
        finally:
            self.pool.shutdown(wait=True)


def make_server(workspace, port=8765):
    token = secrets.token_urlsafe(32)
    template = Path(__file__).with_name("ui.html").read_text(encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, status, data, content_type="application/json; charset=utf-8"):
            payload = data.encode("utf-8") if isinstance(data, str) else json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def allowed(self, api=False):
            port = self.server.server_port
            if self.headers.get("Host") not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
                self.send(403, {"error": "仅允许本机访问"})
                return False
            if api and not secrets.compare_digest(self.headers.get("X-UI-Token", ""), token):
                self.send(403, {"error": "请刷新本地页面后重试"})
                return False
            return True

        def do_GET(self):
            if not self.allowed(api=self.path.startswith("/api/")):
                return
            if self.path == "/":
                self.send(200, template.replace("__UI_TOKEN__", token), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self.send(200, workspace.snapshot())
            elif self.path == "/api/sessions":
                self.send(200, workspace.sessions())
            else:
                self.send(404, {"error": "未找到页面"})

        def do_POST(self):
            if not self.allowed(api=True):
                return
            if self.path not in {"/api/chat", "/api/session"}:
                self.send(404, {"error": "未知接口"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 150000:
                    raise ValueError("请求过大或为空")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("请求必须为 JSON 对象")
                action = self.path.rsplit("/", 1)[-1]
                workspace.submit(action, data.get("message") if action == "chat" else data.get("id"))
                self.send(202, {"accepted": True})
            except RuntimeError as error:
                self.send(409, {"error": str(error)})
            except (ValueError, UnicodeError) as error:
                self.send(400, {"error": str(error)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description="mini-harness 本地聊天界面")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    workspace = Workspace(Path(".mini-harness/sessions"))
    try:
        server = make_server(workspace, args.port)
    except OSError as error:
        workspace.close()
        parser.exit(1, f"启动失败：{error}。可通过 --port 指定其他端口。\n")
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"mini-harness UI: {url}\n关闭服务请按 Ctrl+C；正在执行的任务会先完成并保存。", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("正在保存会话并关闭…", flush=True)
    finally:
        server.server_close()
        workspace.close()


if __name__ == "__main__":
    main()
