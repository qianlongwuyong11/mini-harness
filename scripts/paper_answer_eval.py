"""Four-question development evaluation; sends selected evidence to DeepSeek."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mini_harness.bundle import BundleConfig,headless_bundle
from mini_harness.client import DeepSeekClient
from mini_harness.cordis import Context
from mini_harness.rag.runtime import PaperRuntime,RagConfig

QUESTIONS=[
 ('method','SparseBEV 的三个主要模块分别是什么、各有什么作用？请引用作者原文。'),
 ('context','尺度参数如何影响自注意力的感受野？请补读 3.2 节，解释定义及相邻上下文。'),
 ('annotation','请读取第4页的 FreeText 用户批注，列举两条；明确这些是我的批注，不是作者结论。'),
 ('unanswerable','SparseBEV 作者第一次训练时的 GPU 序列号是什么？论文没有明确记录时请回答未找到依据。'),
]

def main():
 client=DeepSeekClient()
 runtime=PaperRuntime(RagConfig(catalog_path=str(ROOT/'.mini-harness/rag/ingestion-smoke.sqlite3'),index_name='mini-harness-papers-ingestion-step11'))
 report_path=ROOT/'.mini-harness/rag/answer-eval/report.json'
 report_path.parent.mkdir(parents=True,exist_ok=True)
 reports=json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else []
 try:
  for category,question in QUESTIONS:
   if any(row['category']==category for row in reports):
    continue
   ctx=Context()
   try:
    ctx.plugin(headless_bundle,BundleConfig(settings_document={},client=client,enable_rag=True,rag_runtime=runtime,enable_knowledge_base=False,enable_console_questions=False))
    agent=ctx.require('agent')
    agent.followup('仅使用 list_papers、search_papers、read_section 处理以下论文问题。回答简洁，不要调用其他工具。'+question)
    agent.run()
    answer=[m.content for m in agent.session.derive_messages() if m.role=='assistant' and not m.tool_calls][-1]
    evidence=[e.to_dict() for e in ctx.require('paper_evidence').items.values()]
    calls=[e.data['name'] for e in agent.session.events if e.type=='tool/call']
    reports.append(dict(category=category,question=question,answer=answer,tool_calls=calls,evidence=evidence))
    print(json.dumps({'category':category,'answer':answer},ensure_ascii=True),flush=True)
    report_path.write_text(json.dumps(reports,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
   finally: ctx.dispose()
 finally: runtime.close()

if __name__=='__main__':main()
