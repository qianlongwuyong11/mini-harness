import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import httpx
from mini_harness.rag.es_service import ESConfig, ESService

class ESServiceTests(unittest.TestCase):
    def test_reject_remote_plaintext_and_url_credentials(self):
        for url in ('http://localhost:9200','https://example.com:9200','https://u:p@localhost:9200','https://localhost:9200/path'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                ESConfig(url, 'ca.pem', 'user', 'secret')
        self.assertNotIn('secret', repr(ESConfig('https://127.0.0.1:9200','ca.pem','user','secret')))

    def test_load_and_password_environment_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'connection.json'
            path.write_text(json.dumps(dict(url='https://localhost:9200',ca_file='ca.pem',username='user',password='local')))
            with patch.dict('os.environ', {'MINI_HARNESS_ES_PASSWORD':'override'}):
                self.assertEqual(ESConfig.load(path).password, 'override')

    def service(self, status=200, data=None):
        service = ESService.__new__(ESService)
        service.client = httpx.Client(base_url='https://localhost:9200',transport=httpx.MockTransport(
            lambda request: httpx.Response(status, json=data or {})))
        self.addCleanup(service.close)
        return service

    def test_health_rejects_wrong_cluster_red_and_timeout(self):
        for data in (dict(cluster_name='other',status='green'),
                     dict(cluster_name='mini-harness-rag',status='red'),
                     dict(cluster_name='mini-harness-rag',status='yellow',timed_out=True)):
            with self.subTest(data=data), self.assertRaises(RuntimeError):
                self.service(data=data).health()
        self.assertEqual(self.service(data=dict(cluster_name='mini-harness-rag',status='green')).health()['status'],'green')

    def test_authentication_error_is_not_healthy(self):
        with self.assertRaises(httpx.HTTPStatusError):
            self.service(status=401).health()

class ESLauncherOwnershipTests(unittest.TestCase):
    def test_reusing_existing_instance_never_launches_or_kills(self):
        import importlib.util
        from unittest.mock import MagicMock
        spec = importlib.util.spec_from_file_location('local_es_test', Path(__file__).resolve().parents[1] / 'scripts/local_es.py')
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        socket_mock = MagicMock()
        socket_mock.__enter__.return_value.connect_ex.return_value = 0
        with patch.object(launcher.ESConfig, 'load'), patch.object(launcher, 'ESService') as service, \
             patch.object(launcher.socket, 'socket', return_value=socket_mock), \
             patch.object(launcher.subprocess, 'Popen') as popen, patch.object(launcher.subprocess, 'run') as run:
            launcher.serve()
            service.return_value.health.assert_called_once()
            service.return_value.close.assert_called_once()
            popen.assert_not_called()
            run.assert_not_called()
