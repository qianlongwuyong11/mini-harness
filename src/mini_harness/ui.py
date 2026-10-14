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
from .telemetry import turn_metrics, step_metrics
from urllib.parse import urlsplit, parse_qs


class Workspace:
    def __init__(self, root: Path, factory=None, papers=None):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.factory = factory
        self.papers = papers
        self.evaluation = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ui-agent")
        self.lock = threading.Lock()
        self.agent = None
        self.store = None
        self.unsubscribe = None
        self.state = dict(busy=False, session_id=None, messages=[], activity=[], error=None, paper_ids=[], turn_metrics=[], step_metrics=[])

    def evaluations(self):
        with self.lock:
            if self.evaluation is None:
                from .evaluation.ui_service import EvaluationUI
                self.evaluation = EvaluationUI(self.root.parent / 'evals', self.papers.runtime if self.papers else None)
            return self.evaluation

    def snapshot(self):
        with self.lock:
            if self.papers:
                self.state['paper_ids'] = [i for i in self.state['paper_ids']
                    if self.papers.runtime.catalog.get_active_version(i) is not None]
            return copy.deepcopy(self.state)

    def sessions(self):
        result = []
        for path in sorted(self.root.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            if re.fullmatch(r"[0-9a-f]{32}", path.stem):
                result.append(dict(id=path.stem, modified=path.stat().st_mtime))
        return result

    def select_papers(self, ids):
        if not isinstance(ids,list) or len(ids)>20 or any(not isinstance(i,str) for i in ids):
            raise ValueError("最多选择 20 篇论文")
        if self.papers is None: raise ValueError("论文服务未启用")
        if any(self.papers.runtime.catalog.get_active_version(i) is None for i in ids):
            raise ValueError("只能选择已就绪的论文")
        with self.lock:
            if self.state['busy']: raise RuntimeError("当前任务仍在执行")
            self.state['paper_ids']=list(dict.fromkeys(ids))

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
            selected = tuple(self.state['paper_ids'])
        self.pool.submit(self._perform, action, (value,selected) if action=='chat' else value)

    def _publish_messages(self):
        if self.agent is None:
            return
        evidence = {}
        for event in self.agent.session.events:
            if event.type == 'rag/citations':
                for item in event.data.get('evidence',()):
                    evidence[item['evidence_id']] = json.loads(json.dumps(item))
        messages = []
        for event in self.agent.session.events:
            # The chat transcript keeps original replies; model-context replacements
            # must not move historical replies past their execution statistics.
            if event.data.get("surface_op"): continue
            if event.type not in {'user/message', 'assistant/message'}: continue
            content = event.data.get('content')
            if not content: continue
            role = event.type.split('/')[0]
            row = dict(role=role, content=content, event_id=event.id)
            if role == 'assistant':
                ids = dict.fromkeys(re.findall(r'\[cite:([^\]\s]+)\]', content))
                row['citations'] = [evidence[i] for i in ids if i in evidence]
            messages.append(row)
        with self.lock:
            self.state["messages"] = messages
            self.state["turn_metrics"] = json.loads(json.dumps(turn_metrics(self.agent.session.events)))
            self.state["step_metrics"] = json.loads(json.dumps(step_metrics(self.agent.session.events)))

    def _event(self, event):
        labels = {"tool/call": "调用工具", "tool/result": "工具返回", "step/start": "分析与执行", "turn/end": "本轮结束"}
        if event.type in labels:
            name = event.data.get("name", "")
            with self.lock:
                self.state["activity"].append(labels[event.type] + (f" · {name}" if name else ""))
                self.state["activity"] = self.state["activity"][-30:]
        if event.type in {"user/message", "assistant/message", "turn/end", "metrics/model", "metrics/tool", "rag/trace", "step/start", "step/end"}:
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
        options=dict(session=session,checkpoint_flush=store.save,enable_console_questions=False)
        if self.papers is not None and self.factory is None:
            options['rag_runtime']=self.papers.runtime
        agent = factory(**options)
        self.agent, self.store = agent, store
        self.unsubscribe = session.subscribe(self._event)
        with self.lock:
            selections=[list(e.data['paper_ids']) for e in session.events if e.type=='rag/scope']
            selected=selections[-1] if selections else []
            if self.papers:
                selected=[i for i in selected if self.papers.runtime.catalog.get_active_version(i)]
            self.state.update(session_id=identifier,messages=[],activity=[],paper_ids=selected)
        self._publish_messages()

    def _perform(self, action, value):
        try:
            if action == "session":
                self._open(value)
            else:
                value, selected = value
                if self.agent is None:
                    self._open()
                if self.papers and hasattr(self.agent,'context'):
                    self.agent.context.require('paper_scope')['paper_ids']=selected
                    self.agent.session.append('rag/scope',{'paper_ids':list(selected)})
                    with self.lock: self.state['paper_ids']=list(selected)
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
            if self.evaluation: self.evaluation.close()
            if self.papers: self.papers.close()


def make_server(workspace, port=8765):
    token = secrets.token_urlsafe(32)
    template = Path(__file__).with_name("ui.html").read_text(encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, status, data, content_type="application/json; charset=utf-8"):
            payload = data if isinstance(data,bytes) else data.encode("utf-8") if isinstance(data,str) else json.dumps(data,ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'self' blob:; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
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
            elif self.path in ('/evaluation','/evaluation.js'):
                file = 'evaluation.html' if self.path=='/evaluation' else 'evaluation.js'
                content=Path(__file__).with_name(file).read_text(encoding='utf-8').replace('__UI_TOKEN__',token)
                self.send(200,content,'text/html; charset=utf-8' if file.endswith('.html') else 'application/javascript; charset=utf-8')
            elif urlsplit(self.path).path.startswith('/api/evaluation/'):
                try:
                    service=workspace.evaluations()
                    action=urlsplit(self.path).path.rsplit('/',1)[-1]
                    params=parse_qs(urlsplit(self.path).query)
                    get=lambda key: params[key][0]
                    if action=='state': result=service.state()
                    elif action=='run': result=service.reports.run(get('run_id'))
                    elif action=='case': result=service.reports.case(get('run_id'),get('attempt_id'))
                    elif action=='compare': result=service.reports.compare(get('baseline'),get('run_id'))
                    elif action=='export':
                        content,kind=service.reports.export(get('run_id'),get('format'))
                        self.send(200,content,kind)
                        return
                    else: raise ValueError('未知评测接口')
                    self.send(200,result)
                except (ValueError,KeyError,StopIteration,OSError,TypeError):
                    self.send(400,{'error':'评测数据不可用，请检查运行、用例或本地配置。'})
            elif self.path == "/api/state":
                self.send(200, workspace.snapshot())
            elif self.path == "/api/sessions":
                self.send(200, workspace.sessions())
            elif self.path == '/api/papers':
                self.send(200,workspace.papers.snapshot() if workspace.papers else {'papers':[],'status':{'error':'论文服务未启用'}})
            elif urlsplit(self.path).path in ('/api/papers/source','/api/papers/page'):
                try:
                    if workspace.papers is None: raise ValueError('论文服务未启用')
                    params=parse_qs(urlsplit(self.path).query,strict_parsing=True)
                    if set(params)!={'paper_id','version','page'} or any(len(v)!=1 for v in params.values()):
                        raise ValueError('原文请求参数无效')
                    is_image=urlsplit(self.path).path.endswith('/page')
                    reader=workspace.papers.page_image if is_image else workspace.papers.source
                    data=reader(params['paper_id'][0],params['version'][0],int(params['page'][0]))
                    self.send(200,data,'image/png' if is_image else 'application/pdf')
                except (ValueError,OSError,KeyError):
                    self.send(400,{'error':'原文不可用：请检查版本、页码，论文可能已删除或更新。'})
            else:
                self.send(404, {"error": "未找到页面"})

        def do_POST(self):
            if not self.allowed(api=True):
                return
            if self.path.startswith('/api/evaluation/'):
                try:
                    self.connection.settimeout(30)
                    size=int(self.headers.get('Content-Length','0'))
                    if not 0<size<=10*1024*1024: raise ValueError('请求为空或超过 10 MiB')
                    data=json.loads(self.rfile.read(size))
                    if not isinstance(data,dict): raise ValueError('需要 JSON 对象')
                    result=workspace.evaluations().action(self.path.rsplit('/',1)[-1],data)
                    self.send(200,result)
                except RuntimeError as error: self.send(409,{'error':str(error)})
                except (ValueError,KeyError,StopIteration,OSError,TypeError) as error:
                    self.send(400,{'error':str(error) if isinstance(error,ValueError) else '评测数据不可用'})
                return
            if self.path == '/api/papers/upload':
                try:
                    if workspace.papers is None: raise ValueError('论文服务未启用')
                    self.connection.settimeout(30)
                    size=int(self.headers.get('Content-Length','0'))
                    result=workspace.papers.upload(self.rfile,size,self.headers.get('X-Filename',''),self.headers.get('Content-Type',''))
                    self.send(202,result)
                except (ValueError,OSError,TimeoutError) as error:
                    self.close_connection=True
                    self.send(400,{'error':str(error)})
                except Exception:
                    self.close_connection=True
                    self.send(503,{'error':'无法创建导入任务，请检查本地模型文件与服务配置。'})
                return
            if self.path not in {"/api/chat", "/api/session", '/api/scope','/api/papers/retry','/api/papers/cancel','/api/papers/delete'}:
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
                if self.path.startswith('/api/papers/'):
                    if workspace.papers is None: raise ValueError('论文服务未启用')
                    self.send(200,workspace.papers.action(action,data.get('id')))
                    return
                if action=='scope':
                    workspace.select_papers(data.get('paper_ids'))
                    self.send(200,{'selected':True})
                    return
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
    parser.add_argument('--rag-catalog',default='.mini-harness/rag/vector-catalog.sqlite3')
    parser.add_argument('--rag-root',default='.mini-harness/rag/ingestion')
    parser.add_argument('--rag-index',default=None)
    parser.add_argument('--no-rag',action='store_true')
    args = parser.parse_args()
    papers=None
    if not args.no_rag:
        from .rag.runtime import PaperRuntime,RagConfig
        from .rag.ui_service import PaperUI
        papers=PaperUI(PaperRuntime(RagConfig(catalog_path=args.rag_catalog,root=args.rag_root,index_name=args.rag_index)))
    workspace = Workspace(Path(".mini-harness/sessions"),papers=papers)
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
