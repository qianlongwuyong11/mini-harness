import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import httpx
from mini_harness.bundle import BundleConfig, headless_bundle
from mini_harness.cordis import Context
from mini_harness.client import Message, Tool, ToolCall, DeepSeekClient
from mini_harness.evaluation.trace import TraceStore, wire_capture

class TraceTests(unittest.TestCase):
    def test_store_redacts_and_content_addressed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=TraceStore(tmp,secrets=('fixture-private-key',))
            value={'messages':[{'content':'fixture-private-key','reasoning_content':'hidden'}],
                   'arguments':'{"api_key":"nested-private"}'}
            ref=store.put(value);self.assertEqual(ref,store.put(value))
            text=(Path(tmp)/ref['relative_path']).read_text(encoding='utf-8')
            self.assertNotIn('fixture-private-key',text);self.assertNotIn('nested-private',text)
            self.assertNotIn('hidden',text)
            self.assertEqual(len(list(Path(tmp).rglob('*.json'))),1)

    def test_wire_capture_is_actual_request(self):
        response=Mock();response.json.return_value={'id':'provider-1','choices':[{'message':{'content':'ok'}}]}
        payloads=[]
        with patch('mini_harness.client.httpx.Client') as factory:
            factory.return_value.__enter__.return_value.post.return_value=response
            with wire_capture(payloads.append):
                reply=DeepSeekClient('fixture-key').chat([Message('user','q')])
            self.assertEqual(payloads[0],factory.return_value.__enter__.return_value.post.call_args.kwargs['json'])
            self.assertNotIn('Authorization',payloads[0])
            self.assertEqual(reply.provider_request_id,'provider-1')

    def test_retry_and_tool_error_linked_and_final_context_saved(self):
        class Client:
            MODEL='fixture'
            def __init__(self):self.n=0
            def chat(self,messages,tools):
                self.n+=1
                if self.n==1:raise httpx.ReadTimeout('fixture timeout')
                if self.n==2:return Message('assistant',tool_calls=(ToolCall('call1','broken','{}'),))
                return Message('assistant','done')
        ctx=Context();ctx.plugin(headless_bundle,BundleConfig(settings_document={},client=Client(),enable_knowledge_base=False))
        def broken(args):raise ValueError('fixture failure')
        ctx.require('tools').register(Tool('broken','broken',{},broken))
        with tempfile.TemporaryDirectory() as tmp:
            agent=ctx.require('agent');agent._trace_store=TraceStore(tmp)
            def inject(request,next_fn):
                from dataclasses import replace
                return next_fn(replace(request,messages=[*request.messages,Message('user','injected-evidence')]))
            ctx.on('llm/request',inject)
            try:
                with patch('mini_harness.retry.time.sleep'):
                    agent.followup('question');agent.run()
                events=agent.session.events
                models=[e.data for e in events if e.type=='metrics/model']
                self.assertEqual(len({m['model_request_id'] for m in models}),3)
                self.assertEqual(models[1]['retry_of'],models[0]['model_request_id'])
                self.assertEqual(models[1]['purpose'],'retry')
                self.assertEqual(models[0]['step_event_id'],models[1]['step_event_id'])
                snapshot=json.loads((Path(tmp)/models[1]['input_ref']['relative_path']).read_text(encoding='utf-8'))
                self.assertEqual(snapshot['messages'][-1]['content'],'injected-evidence')
                tool=next(e.data for e in events if e.type=='metrics/tool')
                self.assertEqual(tool['model_request_id'],models[1]['model_request_id'])
                self.assertEqual(tool['error_type'],'ValueError')
                self.assertEqual(tool['error_code'],'TOOL_EXECUTION_ERROR')
                self.assertTrue(tool['raw_output_ref'])
                self.assertNotIn('trace/model-start',[m.content for m in agent.session.derive_messages()])
            finally:ctx.dispose()

    def test_trace_write_failure_does_not_repeat_model(self):
        client=Mock(MODEL='fixture',api_key=None);client.chat.return_value=Message('assistant','done')
        ctx=Context();ctx.plugin(headless_bundle,BundleConfig(settings_document={},client=client,enable_knowledge_base=False))
        try:
            agent=ctx.require('agent');agent._trace_store=Mock();agent._trace_store.put.side_effect=OSError('disk full')
            agent.followup('q');agent.run()
            self.assertEqual(client.chat.call_count,1)
            self.assertTrue(any(e.type=='trace/error' for e in agent.session.events))
        finally:ctx.dispose()

if __name__=='__main__':unittest.main()
