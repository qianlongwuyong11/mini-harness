import json,tempfile,unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock,patch
from mini_harness.client import Message,Tool,ToolCall,DeepSeekClient,normalize_usage
from mini_harness.bundle import BundleConfig,headless_bundle
from mini_harness.cordis import Context
from mini_harness.session import Session
from mini_harness.persistence import JsonlStore
from mini_harness.telemetry import metric_scope,turn_metrics,step_metrics,record_metric
from mini_harness.rag.adapters.llamaindex import LlamaIndexSearchAdapter
from mini_harness.rag.models import SearchQuery,SearchResult,RetrievalMode
from llama_index.core.callbacks.schema import CBEventType

class TelemetryTests(unittest.TestCase):
    def test_provider_usage_parsed_without_echo_to_wire(self):
        response=Mock();response.json.return_value={'choices':[{'message':{'content':'ok'}}],'usage':{'prompt_tokens':20,'completion_tokens':3,'total_tokens':23,'prompt_cache_hit_tokens':10,'secret':'not retained'}}
        with patch('mini_harness.client.httpx.Client') as factory:
            factory.return_value.__enter__.return_value.post.return_value=response
            message=DeepSeekClient('fixture').chat([Message('user','q')])
        self.assertEqual(message.usage['total_tokens'],23)
        self.assertNotIn('secret',message.usage)
        self.assertNotIn('usage',DeepSeekClient._wire_message(message))
        self.assertIsNone(normalize_usage({'prompt_tokens':True,'total_tokens':-2}))

    def test_real_agent_tool_calls_usage_and_roundtrip(self):
        class Client:
            MODEL='fixture'
            def __init__(self):self.count=0
            def chat(self,messages,tools):
                self.count+=1
                if self.count==1:return Message('assistant',tool_calls=(ToolCall('c','metric_echo','{}'),),usage={'prompt_tokens':10,'completion_tokens':2,'total_tokens':12})
                return Message('assistant','done',usage={'prompt_tokens':20,'completion_tokens':3,'total_tokens':23})
        ctx=Context();ctx.plugin(headless_bundle,BundleConfig(settings_document={},client=Client(),enable_knowledge_base=False))
        ctx.require('tools').register(Tool('metric_echo','echo',{},lambda a:'ok'))
        try:
            agent=ctx.require('agent');agent.followup('hello');agent.run()
            row=turn_metrics(agent.session.events)[0]
            self.assertEqual((row['model_count'],row['tool_count']),(2,1));self.assertEqual(row['tokens']['total_tokens'],35)
            self.assertTrue(row['usage_complete']);self.assertGreater(row['model_ms'],0)
            steps=step_metrics(agent.session.events)
            self.assertEqual(len(steps),2)
            self.assertEqual([r['tokens']['total_tokens'] for r in steps],[12,23])
            self.assertEqual([r['outcome'] for r in steps],['tool_results','answer'])
            self.assertEqual([r['tool_count'] for r in steps],[1,0])
            self.assertEqual(steps[0]['tool_calls'][0]['result_preview'],'ok')
            self.assertEqual(steps[0]['tools'][0]['step_event_id'],steps[0]['id'])
            self.assertEqual(steps[1]['model_calls'][0]['step_event_id'],steps[1]['id'])
            self.assertTrue(all(r['timing_source']=='monotonic' for r in steps))
            with tempfile.TemporaryDirectory() as tmp:
                store=JsonlStore(Path(tmp)/'s.jsonl');store.save(agent.session)
                self.assertEqual(turn_metrics(store.load().events),turn_metrics(agent.session.events))
                self.assertEqual(step_metrics(store.load().events),steps)
        finally:ctx.dispose()

    def test_error_model_is_counted_and_unknown_usage_is_not_zero(self):
        class Client:
            MODEL='fixture'
            def chat(self,messages,tools):raise ValueError('bad request')
        ctx=Context();ctx.plugin(headless_bundle,BundleConfig(settings_document={},client=Client(),enable_knowledge_base=False))
        try:
            agent=ctx.require('agent');agent.followup('hello')
            with self.assertRaises(ValueError):agent.run()
            row=turn_metrics(agent.session.events)[0]
            self.assertEqual(row['model_errors'],1);self.assertIsNone(row['tokens']['total_tokens']);self.assertEqual(row['status'],'error')
            self.assertEqual(step_metrics(agent.session.events)[0]['status'],'error')
        finally:ctx.dispose()

    def test_scoped_llamaindex_spans_and_empty_results(self):
        def worker(fail):
            session=Session()
            def search(q,callback_manager):
                with callback_manager.event(CBEventType.EMBEDDING,payload={'stage':'query_embedding','secret':'do not persist'}):
                    if fail:raise RuntimeError('sensitive error text')
                return SearchResult(q,(),RetrievalMode.HYBRID,0),{}
            with metric_scope(session):
                adapter=LlamaIndexSearchAdapter(Mock(),trace_search=search)
                if fail:
                    with self.assertRaises(RuntimeError):adapter.search(SearchQuery('private query'))
                else:adapter.search(SearchQuery('private query'))
            return session.events[-1].data
        with ThreadPoolExecutor(2) as pool:rows=list(pool.map(worker,[False,True]))
        self.assertNotEqual(rows[0]['trace_id'],rows[1]['trace_id'])
        self.assertEqual(rows[0]['returned_count'],0);self.assertEqual(rows[1]['status'],'error')
        self.assertTrue(any(s['name']=='query_embedding' for s in rows[0]['spans']))
        self.assertTrue(any(s['status']=='error' for s in rows[1]['spans']))
        self.assertNotIn('private query',json.dumps(rows));self.assertNotIn('sensitive error',json.dumps(rows))

    def test_restored_turn_numbers_do_not_merge(self):
        s=Session()
        for _ in range(2):
            s.append('turn/start',{'turn':1});s.append('turn/end',{'reason':'completed'})
        rows=turn_metrics(s.events)
        self.assertEqual(len(rows),2);self.assertNotEqual(rows[0]['id'],rows[1]['id']);self.assertFalse(rows[0]['instrumented'])

    def test_step_retries_multiple_tools_and_rag_stay_in_one_step(self):
        s=Session()
        s.append('turn/start',{'turn':1,'metrics_version':2})
        start=s.append('step/start',{'step':1})
        s.append('metrics/model',dict(step=1,elapsed_ms=2,status='error',usage=None))
        s.append('llm/retry',{'delay_ms':10})
        s.append('metrics/model',dict(step=1,elapsed_ms=3,status='ok',usage=dict(prompt_tokens=5,completion_tokens=2,total_tokens=7)))
        for cid in ('a','b'):
            s.append('tool/call',dict(call_id=cid,name='search_papers',arguments='{}'))
            with metric_scope(s,step_event_id=start.id,call_id=cid):
                record_metric('rag/trace',dict(spans=[],status='ok'))
            s.append('tool/result',dict(call_id=cid,content='x'*1500,is_error=cid=='b'))
            s.append('metrics/tool',dict(call_id=cid,name='search_papers',elapsed_ms=1,status='error' if cid=='b' else 'ok'))
        s.append('step/end',dict(status='completed',elapsed_ms=20,outcome='tool_results'))
        s.append('turn/end',{'reason':'completed'})
        r=step_metrics(s.events)[0]
        self.assertEqual((r['model_count'],r['tool_count'],r['retries']),(2,2,1))
        self.assertEqual(r['status'],'completed_with_errors')
        self.assertFalse(r['usage_complete'])
        self.assertEqual([t['call_id'] for t in r['rag']],['a','b'])
        self.assertEqual(len(r['tool_calls'][0]['result_preview']),1200)
        self.assertEqual(r['tool_calls'][0]['result_chars'],1500)
        self.assertEqual(r['tokens']['total_tokens'],7)

    def test_legacy_steps_reset_numbers_and_crash(self):
        s=Session()
        for reason in ('completed','crashed'):
            s.append('turn/start',{'turn':1})
            s.append('step/start',{'step':1})
            s.append('step/end',{'step':1,'reason':reason})
            s.append('turn/end',{'reason':reason})
        rows=step_metrics(s.events)
        self.assertEqual([r['turn_number'] for r in rows],[1,2])
        self.assertNotEqual(rows[0]['id'],rows[1]['id'])
        self.assertEqual(rows[1]['status'],'interrupted')
        self.assertFalse(rows[0]['instrumented'])
