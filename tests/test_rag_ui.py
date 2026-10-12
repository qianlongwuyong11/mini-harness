import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from dataclasses import replace
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from unittest.mock import Mock
from pypdf import PdfWriter

from mini_harness.rag.runtime import PaperRuntime,RagConfig
from mini_harness.rag.ui_service import PaperUI,MAX_PDF_BYTES
from mini_harness.rag.models import PaperVersion,Paper,VersionState
from mini_harness.ui import Workspace,make_server
from test_ui import FakeAgent


class PaperUITests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.runtime=PaperRuntime(RagConfig(catalog_path=str(self.root/'catalog.sqlite3'),root=str(self.root/'ingestion')))
        self.papers=PaperUI(self.runtime,monitor=False)
        self.workspace=Workspace(self.root/'sessions',FakeAgent,self.papers)
        self.addCleanup(self.workspace.close)
        self.server=make_server(self.workspace,0)
        thread=threading.Thread(target=self.server.serve_forever)
        thread.start()
        self.addCleanup(lambda:(self.server.shutdown(),self.server.server_close(),thread.join()))
        self.base=f'http://127.0.0.1:{self.server.server_port}'
        self.token=urlopen(self.base).read().decode().split("const token='")[1].split("'")[0]
        writer=PdfWriter()
        writer.add_blank_page(width=100,height=100)
        buffer=io.BytesIO();writer.write(buffer)
        self.pdf=buffer.getvalue()

    def request(self,path,data=None,headers=None):
        return urlopen(Request(self.base+path,data=data,headers={'X-UI-Token':self.token,**(headers or {})}))

    def test_upload_validates_bytes_size_and_names(self):
        self.runtime.enqueue=Mock(return_value=Mock(job_id='job',paper_id='p',status=Mock(value='queued')))
        with self.request('/api/papers/upload',self.pdf,{'Content-Type':'application/pdf','X-Filename':'paper.pdf'}) as r:
            self.assertEqual(r.status,202)
        self.runtime.enqueue.assert_called_once()
        self.assertFalse(list((self.root/'ingestion/uploads').glob('*.pdf')))
        for payload,name,size in [(b'not pdf','a.pdf',7),(self.pdf,'../escape.pdf',len(self.pdf)),(self.pdf,'a.txt',len(self.pdf)),(b'x','a.pdf',MAX_PDF_BYTES+1)]:
            with self.assertRaises(HTTPError) as e:
                self.request('/api/papers/upload',payload,{'Content-Type':'application/pdf','X-Filename':name,'Content-Length':str(size)})
            self.assertEqual(e.exception.code,400)
        self.assertEqual(self.runtime.enqueue.call_count,1)

    def ready(self):
        source=self.root/'source.pdf';source.write_bytes(self.pdf)
        c=self.runtime.catalog
        c.create_paper(Paper('p',str(source),'h','Paper'))
        v=PaperVersion('p','v','h','parser','chunker')
        c.save_version(v)
        ready=replace(v,state=VersionState.READY,chunk_count=1)
        with c._connection(write=True) as db:db.execute('UPDATE versions SET data=?',(ready.to_json(),))
        c.activate_version('p','v')

    def test_source_endpoint_and_scope_are_restricted(self):
        self.ready()
        with self.request('/api/papers/source?paper_id=p&version=v&page=1') as r:
            self.assertEqual(r.headers['Content-Type'],'application/pdf')
            self.assertEqual(r.read(),self.pdf)
        for query in ['paper_id=p&version=v&page=2','paper_id=p&version=old&page=1','path=../../secret','paper_id=missing&version=v&page=1']:
            with self.assertRaises(HTTPError) as e:self.request('/api/papers/source?'+query)
            self.assertEqual(e.exception.code,400)
        self.workspace.select_papers(['p'])
        with self.assertRaises(ValueError):self.workspace.select_papers(['unknown'])
        self.runtime.catalog.soft_delete_paper('p')
        self.assertEqual(self.workspace.snapshot()['paper_ids'],[])
        with self.assertRaises(HTTPError):self.request('/api/papers/source?paper_id=p&version=v&page=1')

    def test_retry_cancel_and_status_keep_source_paths_private(self):
        c=self.runtime.catalog
        job=c.enqueue_import('private.pdf','hash',parser_version='p',chunker_version='c')
        owned=c.claim_job('test')
        c.fail_job(owned,'parse failed')
        data=json.load(self.request('/api/papers'))
        self.assertNotIn('source_uri',json.dumps(data))
        self.assertEqual(data['papers'][0]['job']['stage'],'parsing')
        self.request('/api/papers/retry',json.dumps({'id':job.job_id}).encode()).close()
        self.assertEqual(c.get_job(job.job_id).status.value,'queued')
        self.request('/api/papers/cancel',json.dumps({'id':job.job_id}).encode()).close()
        self.assertEqual(c.get_job(job.job_id).status.value,'cancelled')

    def test_upload_and_source_require_ui_token(self):
        for path in ['/api/papers','/api/papers/source?paper_id=p&version=v&page=1']:
            with self.assertRaises(HTTPError) as e:urlopen(self.base+path)
            self.assertEqual(e.exception.code,403)

    def test_nested_citations_survive_snapshot_and_session_resume(self):
        self.workspace._open()
        session=self.workspace.agent.session
        session.append('rag/citations',{'evidence':[{'evidence_id':'e1','locations':[{'page':1,'bbox':[0,0,1,1]}]}]})
        session.append('assistant/message',{'content':'Evidence [cite:e1]'})
        before=self.workspace.snapshot()
        self.assertEqual(before['messages'][-1]['citations'][0]['locations'][0]['page'],1)
        self.workspace._open(before['session_id'])
        self.assertEqual(self.workspace.snapshot()['messages'],before['messages'])

    def test_es_unavailable_is_visible_without_loading_model(self):
        from unittest.mock import patch
        with patch('mini_harness.rag.es_service.ESConfig.load',side_effect=OSError('offline')), patch.object(self.papers._stop,'wait',side_effect=lambda _:self.papers._stop.set()):
            self.papers._monitor()
        status=self.papers.snapshot()['status']
        self.assertEqual(status['es'],'unavailable')
        self.assertTrue(status['error'])
        self.assertIsNone(self.runtime._embedding)

    def test_page_preview_renders_exact_page_and_rejects_invalid_page(self):
        self.ready()
        with self.request('/api/papers/page?paper_id=p&version=v&page=1') as r:
            self.assertEqual(r.headers['Content-Type'],'image/png')
            self.assertTrue(r.read().startswith(bytes([137,80,78,71])))
        with self.assertRaises(HTTPError):self.request('/api/papers/page?paper_id=p&version=v&page=2')
