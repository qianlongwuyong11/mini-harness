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
