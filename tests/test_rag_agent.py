import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from mini_harness.bundle import BundleConfig,headless_bundle
from mini_harness.client import Message,ToolCall
from mini_harness.cordis import Context
from mini_harness.rag.runtime import PaperRuntime,RagConfig
from mini_harness.rag.citations import EvidenceLedger,FALLBACK
from mini_harness.rag.models import Evidence,PageSpan,SourceKind,SourceBlock,BlockKind,Paper,PaperVersion,VersionState,SearchResult,RetrievalMode


class PaperAgentTests(unittest.TestCase):
    def test_request_projection_deduplicates_and_expires_without_rewriting_session(self):
        from mini_harness.rag.projection import project_results
        ctx=Context()
        owner=self
        class Client:
            MODEL='fixture'
            calls=0
            def chat(inner,messages,tools):
                inner.calls+=1
                if inner.calls==1:
                    return Message('assistant',tool_calls=(ToolCall('search','search_papers',json.dumps(dict(query='scale',paper_ids=['p']))),))
                results=[m for m in messages if m.role=='tool']
                owner.assertEqual(len(results),1)
                owner.assertEqual(results[0].tool_call_id,'search')
                if inner.calls==2:
                    payloads=[m for m in messages if (m.content or '').startswith('本轮证据快照')]
                    owner.assertEqual(len(payloads),1)
                    owner.assertEqual(sum((m.content or '').count(owner.e.excerpt) for m in messages),1)
                    item=json.loads(payloads[0].content.split('\n',1)[1])[0]
                    summary=json.loads(results[0].content)
                    owner.assertEqual(summary['evidence_ids'],[item['evidence_id']])
                    return Message('assistant','Answer [cite:'+item['evidence_id']+']')
                owner.assertEqual(json.loads(results[0].content)['status'],'historical')
                owner.assertFalse(any((m.content or '').startswith('本轮证据快照') for m in messages))
                owner.assertFalse(any(owner.e.excerpt in (m.content or '') for m in messages))
                owner.assertTrue(any('：[]。' in (m.content or '') for m in messages))
                return Message('assistant','请在论文库重新勾选论文。')
        ctx.plugin(headless_bundle,BundleConfig(settings_document={},client=Client(),enable_rag=True,
            rag_runtime=self.runtime,enable_knowledge_base=False))
        self.addCleanup(ctx.dispose)
        agent=ctx.require('agent')
        ctx.require('paper_scope')['paper_ids']=('p',)
        agent.followup('Read method');agent.run(max_turns=1)
        original=next(e for e in agent.session.events if e.type=='tool/result')
        self.assertEqual(json.loads(original.data['content'])['evidence'][0]['excerpt'],self.e.excerpt)
        ctx.require('paper_scope')['paper_ids']=()
        agent.followup('取消勾选');agent.run(max_turns=2)
        self.assertEqual(original,next(e for e in agent.session.events if e.type=='tool/result'))
        self.assertFalse(any((e.data.get('content') or '').startswith('本轮证据快照') for e in agent.session.events))
        # A restored session has no live ledger/cache; old tool bodies still expire.
        from mini_harness.session import Session
        restored=Session.from_log(agent.session.events)
        projected=project_results(restored.derive_messages(),{})
        self.assertFalse(any(self.e.excerpt in (m.content or '') for m in projected))

    def test_projection_handles_spilled_results_errors_and_non_rag_tools(self):
        from mini_harness.rag.projection import project_results
        msgs=[Message('assistant',tool_calls=(ToolCall('x','search_papers','{}'),)),
              Message('tool','large truncated preview and spill path',tool_call_id='x'),
              Message('assistant',tool_calls=(ToolCall('other','calculator','{}'),)),
              Message('tool','42',tool_call_id='other')]
        summary=dict(status='ok',evidence_ids=['e'],count=1)
        projected=project_results(msgs,{'x':summary})
        self.assertEqual(json.loads(projected[1].content),summary)
        self.assertEqual(projected[2:],msgs[2:])
        self.assertEqual(msgs[1].content,'large truncated preview and spill path')
        self.assertEqual(project_results(msgs,{'x':None}),msgs)
        # Reused call IDs do not make the old result current again.
        repeated=msgs[:2]+msgs[:2]
        projected=project_results(repeated,{'x':summary})
        self.assertEqual(json.loads(projected[1].content)['status'],'historical')
        self.assertEqual(json.loads(projected[3].content),summary)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runtime=PaperRuntime(RagConfig(catalog_path=str(Path(self.tmp.name)/'catalog.sqlite3')))
        self.addCleanup(self.runtime.close)
        c=self.runtime.catalog
        c.create_paper(Paper('p','fixture://pdf','h','Paper'))
        v=PaperVersion('p','v','h','parser','chunker')
        c.save_version(v)
        c.save_blocks((SourceBlock('b','p','v','A scale parameter controls the receptive field.',BlockKind.TEXT,SourceKind.PAPER,('Method',),(PageSpan(2,2),)),
                       SourceBlock('a','p','v','My annotation',BlockKind.ANNOTATION,SourceKind.USER_ANNOTATION,('PDF annotation','FreeText'),(PageSpan(2,2),))))
        ready=replace(v,state=VersionState.READY,chunk_count=1)
        with c._connection(write=True) as db:
            db.execute('UPDATE versions SET data=?',(ready.to_json(),))
        c.activate_version('p','v')
        self.e=Evidence('old','p','v','c','Paper','A scale parameter controls the receptive field.',SourceKind.PAPER,('Method',),(PageSpan(2,2),))
        self.runtime.search=lambda q: SearchResult(q,(self.e,),RetrievalMode.HYBRID,1)

    def test_ledger_budget_old_ids_and_deleted_version(self):
        ledger=EvidenceLedger(self.runtime,256)
        first=ledger.add((self.e,self.e))
        self.assertEqual(len(first),1)
        ledger.validate_answer('[cite:'+first[0].evidence_id+']')
        self.assertFalse(ledger.add((replace(self.e,excerpt='large '*3000),)))
        ledger.begin()
        with self.assertRaises(ValueError): ledger.validate_answer('[cite:'+first[0].evidence_id+']')
        e=ledger.add((self.e,))[0]
        self.runtime.catalog.soft_delete_paper('p')
        with self.assertRaises(ValueError): ledger.validate_answer('[cite:'+e.evidence_id+']')

    def test_tools_lifecycle_and_annotation_isolation(self):
        ctx=Context()
        ctx.plugin(headless_bundle,BundleConfig(settings_document={},client=Mock(),enable_knowledge_base=False,enable_rag=True,rag_runtime=self.runtime))
        tools=ctx.require('tools')
        self.assertTrue({'list_papers','search_papers','read_section'} <= set(ctx.require('agent').tool_names))
        result=json.loads(tools.get('read_section').execute(dict(paper_id='p',version='v',section_path=['PDF annotation'],source_kind='user_annotation')))
        self.assertEqual(result['evidence'][0]['source_kind'],'user_annotation')
        self.assertEqual(result['evidence'][0]['excerpt'],'My annotation')
        with self.assertRaises(ValueError): tools.get('search_papers').execute({'query':'q','paper_ids':[]})
        ctx.dispose()
        self.assertFalse(tools.get('search_papers'))
        self.assertFalse(self.runtime._closed)

    def test_agent_tool_answer_and_spill_prune_protection(self):
        ctx=Context()
        class Client:
            MODEL='fake'
            def __init__(inner): inner.calls=0
            def chat(inner,messages,tools):
                inner.calls+=1
                if inner.calls==1:
                    return Message(role='assistant',tool_calls=(ToolCall('s','search_papers',json.dumps({'query':'scale','paper_ids':['p']})),))
                snapshots=[m.content for m in messages if m.content and m.content.startswith('本轮证据快照')]
                assert snapshots
                data=json.loads(snapshots[-1].split('\n',1)[1])
                return Message(role='assistant',content='【作者原文依据】尺度参数控制感受野。[cite:'+data[0]['evidence_id']+']')
        ctx.plugin(headless_bundle,BundleConfig(settings_document={},client=Client(),enable_rag=True,rag_runtime=self.runtime,enable_knowledge_base=False,spill_root=str(Path(self.tmp.name)/'spill')))
        # Simulate history projection losing all tool bodies after prune/compact.
        ctx.on('agent/prepare-request',lambda request: setattr(request,'messages',[m for m in request.messages if m.role!='tool']))
        agent=ctx.require('agent')
        try:
            agent.followup('Explain scale')
            agent.run()
            final=agent.session.derive_messages()[-1].content
            self.assertIn('第 2 页',final)
            self.assertIn('作者原文',final)
        finally: ctx.dispose()

    def test_forged_citation_repair_is_bounded(self):
        ctx=Context()
        class Client:
            MODEL='fake'
            def chat(inner,messages,tools):
                return Message(role='assistant',content='invented [cite:forged]')
        ctx.plugin(headless_bundle,BundleConfig(settings_document={},client=Client(),enable_rag=True,rag_runtime=self.runtime,enable_knowledge_base=False))
        agent=ctx.require('agent')
        try:
            agent.followup('question')
            agent.run()
            self.assertEqual(agent.session.derive_messages()[-1].content,FALLBACK)
        finally: ctx.dispose()

    def test_actual_spill_prune_and_summary_replacement_keep_evidence(self):
        from mini_harness.pruner import ToolResultPruner
        self.e=replace(self.e,excerpt='A scale parameter controls the receptive field. '*180)
        ctx=Context()
        class Client:
            MODEL='fake'
            def __init__(inner):inner.calls=0
            def chat(inner,messages,tools):
                inner.calls+=1
                if inner.calls==1:
                    return Message(role='assistant',tool_calls=(ToolCall('q','search_papers',json.dumps({'query':'scale','paper_ids':['p']})),))
                payload=next(m.content for m in messages if m.content and m.content.startswith('本轮证据快照'))
                evidence=json.loads(payload.split('\n',1)[1])[0]
                assert evidence['excerpt']==self.e.excerpt
                return Message(role='assistant',content='Scale controls field [cite:'+evidence['evidence_id']+']')
        ctx.plugin(headless_bundle,BundleConfig(settings_document={'agent':{'spill_max_inline_bytes':512}},
            client=Client(),enable_rag=True,rag_runtime=self.runtime,enable_knowledge_base=False,
            spill_root=str(Path(self.tmp.name)/'spill')))
        def compact(execution):
            s=execution.session
            ToolResultPruner(threshold_chars=150,head_chars=30,tail_chars=10).prune_session(s)
            visible=[e for e in s.events if e.type in ('user/message','assistant/message','tool/result')]
            s.append('user/message',{'content':'Summary of earlier actions.', 'surface_op':{
                'op':'replace','start':visible[0].id,'end':visible[-1].id}})
        ctx.on('tools/result',compact)
        try:
            agent=ctx.require('agent')
            agent.followup('scale')
            agent.run()
            self.assertTrue(list((Path(self.tmp.name)/'spill').rglob('*.txt')))
            self.assertTrue(any(e.type=='compaction/prune' for e in agent.session.events))
            self.assertIn('第 2 页',agent.session.derive_messages()[-1].content)
        finally:ctx.dispose()

    def test_shared_runtime_has_session_local_evidence(self):
        contexts=[]
        try:
            for _ in range(2):
                ctx=Context()
                ctx.plugin(headless_bundle,BundleConfig(settings_document={},client=Mock(),enable_rag=True,rag_runtime=self.runtime,enable_knowledge_base=False))
                contexts.append(ctx)
            a,b=(ctx.require('paper_evidence') for ctx in contexts)
            identity=a.add((self.e,))[0].evidence_id
            self.assertIsNone(b.get_evidence(identity))
            contexts[0].dispose()
            self.assertFalse(self.runtime._closed)
            self.assertEqual(contexts[1].require('paper_catalog'),self.runtime.catalog)
        finally:
            for ctx in contexts:ctx.dispose()

    def test_partial_evidence_cannot_claim_global_absence(self):
        ledger=EvidenceLedger(self.runtime)
        e=ledger.add((self.e,))[0]
        with self.assertRaises(ValueError):
            ledger.validate_answer('论文中也没有记录设备编号。[cite:'+e.evidence_id+']')
        ledger.validate_answer('本轮检索证据未提供设备编号。[cite:'+e.evidence_id+']')

    def test_ui_scope_cannot_be_overridden_by_model_arguments(self):
        ctx=Context()
        try:
            ctx.plugin(headless_bundle,BundleConfig(settings_document={},client=Mock(),enable_rag=True,rag_runtime=self.runtime,enable_knowledge_base=False))
            ctx.require('paper_scope')['paper_ids']=()
            tools=ctx.require('tools')
            with self.assertRaises(ValueError):tools.get('search_papers').execute({'query':'q','paper_ids':['p']})
            with self.assertRaises(ValueError):tools.get('read_section').execute({'paper_id':'p','version':'v','section_path':['Method']})
            listing=json.loads(tools.get('list_papers').execute({}))
            self.assertEqual(listing['status'],'no_paper_selected')
            self.assertEqual(listing['next_action'],'check_library_inventory')
            self.assertEqual(listing['papers'],[])
            self.assertEqual(listing['library_status'],'nonempty')
            self.assertEqual([p['paper_id'] for p in listing['library_papers']],['p'])
            self.assertEqual(set(listing['library_papers'][0]),{'paper_id','title','ready'})
            ctx.require('paper_scope')['paper_ids']=('p',)
            listing=json.loads(tools.get('list_papers').execute({}))
            self.assertEqual(listing['status'],'ok')
            self.assertEqual([p['paper_id'] for p in listing['papers']],['p'])
            ctx.require('paper_scope')['paper_ids']=('unavailable',)
            self.assertEqual(json.loads(tools.get('list_papers').execute({}))['status'],'selected_papers_unavailable')
            ctx.require('paper_scope')['paper_ids']=None
            self.runtime.list_papers=lambda: []
            self.assertEqual(json.loads(tools.get('list_papers').execute({}))['status'],'library_empty')
            ctx.require('paper_scope')['paper_ids']=()
            empty=json.loads(tools.get('list_papers').execute({}))
            self.assertEqual(empty['status'],'no_paper_selected')
            self.assertEqual(empty['library_status'],'empty')
            self.assertEqual(empty['library_papers'],[])
        finally:ctx.dispose()

    def test_unimported_named_paper_is_not_invented_by_empty_scope(self):
        ctx=Context()
        self.runtime.list_papers=lambda: [
            dict(paper_id='p',title='SparseBEV',ready=True,version='v',sections=[])]
        try:
            ctx.plugin(headless_bundle,BundleConfig(settings_document={},client=Mock(),
                enable_rag=True,rag_runtime=self.runtime,enable_knowledge_base=False))
            ctx.require('paper_scope')['paper_ids']=()
            tools=ctx.require('tools')
            result=json.loads(tools.get('list_papers').execute({}))
            self.assertEqual(result['papers'],[])
            self.assertEqual([p['title'] for p in result['library_papers']],['SparseBEV'])
            self.assertNotIn('FlashOCC',repr(result['library_papers']))
            self.assertEqual(result['next_action'],'check_library_inventory')
            # Visible inventory metadata still does not authorize retrieval.
            with self.assertRaises(ValueError):
                tools.get('search_papers').execute({'query':'method','paper_ids':['p']})
        finally:ctx.dispose()

    def test_stage2_decision_reasons_are_distinct(self):
        ledger=EvidenceLedger(self.runtime,256)
        ledger.add((self.e,self.e,replace(self.e,excerpt='large '*3000)))
        self.assertEqual([d['reason'] for d in ledger.last_decisions],['accepted','duplicate','budget'])
        self.assertEqual(ledger.last_decisions[0]['evidence_id'],next(iter(ledger.items)))

    def test_stage2_section_to_context_to_citation(self):
        from mini_harness.evaluation.trace import TraceStore
        from mini_harness.telemetry import step_metrics
        ctx=Context()
        class Client:
            MODEL='fixture'
            def __init__(inner):inner.count=0
            def chat(inner,messages,tools):
                inner.count+=1
                if inner.count==1:
                    return Message('assistant',tool_calls=(ToolCall('section-call','read_section',json.dumps(dict(paper_id='p',version='v',section_path=['Method']))),))
                evidence=json.loads(messages[-1].content.split('\n',1)[1])
                return Message('assistant','Answer [cite:'+evidence[0]['evidence_id']+']')
        ctx.plugin(headless_bundle,BundleConfig(settings_document={},client=Client(),enable_rag=True,rag_runtime=self.runtime,enable_knowledge_base=False))
        agent=ctx.require('agent');root=Path(self.tmp.name)/'trace';agent._trace_store=TraceStore(root)
        try:
            agent.followup('Read method');agent.run()
            events=agent.session.events
            section=next(e.data for e in events if e.type=='rag/section')
            flow=next(e.data for e in events if e.type=='rag/evidence-flow')
            injected=next(e.data for e in events if e.type=='rag/context')
            cited=next(e.data for e in events if e.type=='rag/citations')
            self.assertEqual(section['call_id'],'section-call')
            self.assertEqual(section['read_count'],1)
            self.assertEqual(section['score_semantics'],'not_applicable')
            decisions=json.loads((root/flow['detail_ref']['relative_path']).read_text(encoding='utf-8'))['decisions']
            self.assertEqual(decisions[0]['evidence_id'],injected['evidence_ids'][0])
            self.assertEqual(list(cited['ids']),list(injected['evidence_ids']))
            self.assertEqual(cited['model_request_id'],injected['model_request_id'])
            self.assertEqual(len(step_metrics(events)[0]['rag_details']),2)
        finally:ctx.dispose()
