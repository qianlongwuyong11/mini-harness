import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from mini_harness.ui import Workspace, make_server


class FakeAgent:
    def __init__(self, session, **kwargs):
        self.session = session

    def followup(self, task):
        self.task = task

    def run(self, **kwargs):
        self.session.append('user/message', {'content': self.task})
        self.session.append('assistant/message', {'content': '回答：' + self.task})

    def close(self):
        pass


class UiTests(unittest.TestCase):
    def test_metrics_survive_ui_session_restore(self):
        class MeasuredAgent(FakeAgent):
            def run(self, **kwargs):
                self.session.append('turn/start', {'metrics_version': 1})
                super().run(**kwargs)
                self.session.append('metrics/model', dict(step=1, model='fixture', elapsed_ms=12,
                    status='ok', usage=dict(prompt_tokens=10, completion_tokens=2, total_tokens=12)))
                self.session.append('llm/retry', {'delay_ms': 100})
                self.session.append('turn/end', {'reason': 'completed', 'elapsed_ms': 20})
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(Path(directory), MeasuredAgent)
            try:
                workspace.submit('chat', 'test metrics')
                workspace.pool.submit(lambda: None).result(timeout=10)
                state = json.loads(json.dumps(workspace.snapshot()))
                metrics = state['turn_metrics']
                self.assertEqual(metrics[0]['tokens']['total_tokens'], 12)
                self.assertEqual(metrics[0]['retry_wait_ms'], 100)
                workspace.submit('session', state['session_id'])
                workspace.pool.submit(lambda: None).result(timeout=10)
                self.assertEqual(workspace.snapshot()['turn_metrics'], metrics)
            finally:
                workspace.close()

    def test_roundtrip_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(Path(directory), FakeAgent)
            try:
                workspace.submit('chat', '测试中文')
                workspace.pool.submit(lambda: None).result(timeout=10)
                state = workspace.snapshot()
                self.assertIsNone(state['error'])
                self.assertEqual(state['messages'][-1]['content'], '回答：测试中文')
                identifier = state['session_id']
                self.assertTrue((Path(directory) / f'{identifier}.jsonl').exists())
                workspace.submit('session', None)
                workspace.pool.submit(lambda: None).result(timeout=10)
                workspace.submit('session', identifier)
                workspace.pool.submit(lambda: None).result(timeout=10)
                self.assertEqual(workspace.snapshot()['messages'], state['messages'])
                with self.assertRaises(ValueError):
                    workspace.submit('session', '../../escape')
            finally:
                workspace.close()

    def test_turn_metrics_interleave_with_repeated_messages_and_resume(self):
        class RepeatedAgent(FakeAgent):
            def run(self, **kwargs):
                self.session.append('turn/start', {'metrics_version': 1, 'turn': 1})
                self.session.append('user/message', {'content': self.task})
                self.session.append('assistant/message', {'content': '正在查询'})
                self.session.append('assistant/message', {'content': '相同回答'})
                self.session.append('turn/end', {'reason': 'completed', 'elapsed_ms': 20})
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(Path(directory), RepeatedAgent)
            try:
                for _ in range(2):
                    workspace.submit('chat', '相同问题')
                    workspace.pool.submit(lambda: None).result(timeout=10)
                state = workspace.snapshot()
                timeline = sorted(
                    [(m['event_id'], m['content']) for m in state['messages']] +
                    [(r['end_event_id'], '本轮统计') for r in state['turn_metrics']])
                self.assertEqual([text for _, text in timeline],
                    ['相同问题', '正在查询', '相同回答', '本轮统计'] * 2)
                workspace.submit('session', state['session_id'])
                workspace.pool.submit(lambda: None).result(timeout=10)
                restored = workspace.snapshot()
                self.assertEqual(restored['messages'], state['messages'])
                self.assertEqual(restored['turn_metrics'], state['turn_metrics'])
                # Even a failed turn with no assistant message keeps its statistics
                # next to that turn's user message, before the next turn.
                session = workspace.agent.session
                session.append('turn/start', {'metrics_version': 1})
                message = session.append('user/message', {'content': '失败的问题'})
                session.append('turn/end', {'reason': 'error', 'elapsed_ms': 1})
                workspace._publish_messages()
                row = workspace.snapshot()['turn_metrics'][-1]
                self.assertEqual(row['status'], 'error')
                self.assertGreater(row['end_event_id'], message.id)
            finally:
                workspace.close()

    def test_step_timeline_keeps_original_replies_after_context_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(Path(directory), FakeAgent)
            try:
                workspace._open()
                s = workspace.agent.session
                s.append('turn/start', {'turn': 1, 'metrics_version': 2})
                first = s.append('step/start', {'step': 1})
                user = s.append('user/message', {'content': '原始问题'})
                reply = s.append('assistant/message', {'content': '查到的原始回答'})
                end = s.append('step/end', {'elapsed_ms': 10, 'status': 'completed'})
                s.append('turn/end', {'reason': 'completed'})
                s.append('assistant/message', {'content': '模型上下文摘要',
                    'surface_op': {'op': 'replace', 'start': user.id, 'end': reply.id}})
                workspace._publish_messages()
                state = workspace.snapshot()
                self.assertEqual([m['content'] for m in state['messages']], ['原始问题', '查到的原始回答'])
                self.assertEqual(state['step_metrics'][0]['id'], first.id)
                self.assertEqual(state['step_metrics'][0]['end_event_id'], end.id)
                self.assertLess(state['messages'][-1]['event_id'], end.id)
            finally:
                workspace.close()

    def test_busy_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            started, release = threading.Event(), threading.Event()
            def factory(**kwargs):
                started.set()
                release.wait(5)
                raise RuntimeError('模拟启动错误')
            workspace = Workspace(Path(directory), factory)
            try:
                workspace.submit('chat', '测试')
                self.assertTrue(started.wait(5))
                with self.assertRaises(RuntimeError):
                    workspace.submit('chat', '第二条')
                release.set()
                workspace.pool.submit(lambda: None).result(timeout=10)
                self.assertFalse(workspace.snapshot()['busy'])
                self.assertIn('模拟启动错误', workspace.snapshot()['error'])
            finally:
                release.set()
                workspace.close()

    def test_http_token_and_real_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(Path(directory), FakeAgent)
            server = make_server(workspace, 0)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            try:
                page = urlopen(base).read().decode()
                token = page.split("const token='")[1].split("'")[0]
                with self.assertRaises(HTTPError) as error:
                    urlopen(base + '/api/state')
                self.assertEqual(error.exception.code, 403)
                request = Request(base + '/api/chat', data=json.dumps({'message': 'hello'}).encode(), headers={'X-UI-Token': token})
                self.assertEqual(urlopen(request).status, 202)
                workspace.pool.submit(lambda: None).result(timeout=10)
                state = json.load(urlopen(Request(base+'/api/state', headers={'X-UI-Token': token})))
                self.assertEqual(state['messages'][-1]['content'], '回答：hello')
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
                workspace.close()


if __name__ == '__main__':
    unittest.main()
