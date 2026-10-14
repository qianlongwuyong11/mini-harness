import json
from pathlib import Path
import re
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from mini_harness.client import Message
from mini_harness.evaluation.ui_service import EvaluationUI
from mini_harness.evaluation.models import EvaluationDataset
from mini_harness.ui import Workspace, make_server


def dataset():
    return dict(dataset_id='ui-fixture',cases=[dict(case_id='c1',category='smoke',split='regression',scenario='fixture',expected_behavior='Answer',turns=[dict(input='<script>alert(1)</script>',selected_papers=[])])])


class Client:
    MODEL='local-fixture'
    BASE_URL='local-only'
    def chat(self,messages,tools):
        return Message(role='assistant',content='fixture answer',usage=dict(prompt_tokens=3,completion_tokens=2,total_tokens=5))


class EvaluationUITests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.service=EvaluationUI(Path(self.tmp.name)/'evals',client_factory=Client)
        self.addCleanup(self.service.close)

    def prepared(self):
        return self.service.validate(dict(filename='fixture.json',content=json.dumps(dataset())))

    def start(self):
        preview=self.prepared()
        run=self.service.start(dict(dataset_id=preview['id'],config_sha256=preview['config_sha256'],allow_model_api=True))['run_id']
        self.service.pool.submit(lambda:None).result(timeout=20)
        return run

    def test_validation_consent_and_scoped_queue(self):
        preview=self.prepared()
        unrelated=self.service.store.enqueue(EvaluationDataset.model_validate(dataset()))
        with self.assertRaises(ValueError): self.service.start(dict(dataset_id=preview['id']))
        with self.assertRaises(ValueError): self.service.start(dict(dataset_id=preview['id'],allow_model_api=True,config_sha256='changed'))
        run=self.start()
        self.assertEqual(self.service.store.get_run(run)['status'],'completed')
        self.assertEqual(self.service.store.get_run(unrelated)['status'],'queued')
        self.assertEqual(self.service.store.attempts(unrelated),[])

    def test_report_step_artifact_compare_export_and_review(self):
        run=self.start()
        report=self.service.reports.run(run)
        self.assertEqual(report['finished'],1)
        self.assertEqual(len(report['attempts']),1)
        attempt=report['cases'][0]['attempt']['attempt_id']
        case=self.service.reports.case(run,attempt)
        self.assertEqual(len(case['steps']),1)
        self.assertEqual(case['steps'][0]['tokens']['total_tokens'],5)
        ref=case['steps'][0]['model_calls'][0]['input_ref']
        payload=self.service.reports.artifact(run,attempt,ref)
        self.assertIn('<script>',json.dumps(payload))
        bad=dict(ref,relative_path='../outside.json')
        with self.assertRaises((ValueError,OSError)): self.service.reports.artifact(run,attempt,bad)
        with self.assertRaises(ValueError): self.service.reports.artifact(run,attempt,dict(ref,sha256='0'*64))
        body,_=self.service.reports.export(run,'html')
        self.assertNotIn('<script>alert',body)
        self.assertIn('&lt;script&gt;',body)
        self.assertIn('metric',self.service.reports.export(run,'csv')[0])
        self.service.action('review',dict(run_id=run,attempt_id=attempt,verdict='failed',note='needs review'))
        reviewed=self.service.reports.case(run,attempt)
        self.assertEqual(reviewed['reviews'][0]['verdict'],'failed')
        self.assertEqual(reviewed['verdict'],case['verdict'])
        draft=self.service.action('draft',dict(run_id=run,case_ids=['c1']))
        self.assertEqual(draft['cases'][0]['split'],'regression')
        other=self.start()
        comparison=self.service.reports.compare(run,other)
        self.assertEqual(comparison['cases'][0]['delta']['total_tokens'],0)
        self.assertEqual(comparison['warnings'],[])
        calls=next(m for m in comparison['metrics'] if m['metric_id']=='model_calls')
        self.assertEqual(calls['delta'],0)
        self.assertEqual(calls['before_samples'],1)

    def test_cancel_active_at_safe_boundary(self):
        entered,release=threading.Event(),threading.Event()
        class Slow(Client):
            def chat(self,messages,tools):
                entered.set();release.wait(5)
                return super().chat(messages,tools)
        self.service.client_factory=Slow
        preview=self.prepared()
        run=self.service.start(dict(dataset_id=preview['id'],config_sha256=preview['config_sha256'],allow_model_api=True))['run_id']
        self.assertTrue(entered.wait(5))
        self.service.action('cancel',dict(run_id=run))
        release.set()
        self.service.pool.submit(lambda:None).result(timeout=10)
        self.assertEqual(self.service.store.get_run(run)['status'],'interrupted')

    def test_http_auth_static_and_bad_requests(self):
        workspace=Workspace(Path(self.tmp.name)/'sessions')
        workspace.evaluation=self.service
        server=make_server(workspace,0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.addCleanup(workspace.close)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base=f'http://127.0.0.1:{server.server_port}'
        html=urlopen(base+'/evaluation').read().decode()
        token=re.search("const UI_TOKEN='([^']+)'",html).group(1)
        self.assertIn('评测中心',html)
        self.assertIn('/evaluation',urlopen(base+'/').read().decode())
        with self.assertRaises(HTTPError) as error: urlopen(base+'/api/evaluation/state')
        self.assertEqual(error.exception.code,403)
        request=Request(base+'/api/evaluation/state',headers={'X-UI-Token':token})
        self.assertIn('runs',json.load(urlopen(request)))
        body=json.dumps(dict(filename='bad.json',content='{}')).encode()
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(base+'/api/evaluation/validate',data=body,headers={'X-UI-Token':token,'Content-Type':'application/json'}))
        self.assertEqual(error.exception.code,400)


if __name__=='__main__': unittest.main()
