"""Exercise real local retrieval callbacks without a chat-model request."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mini_harness.session import Session
from mini_harness.telemetry import metric_scope
from mini_harness.rag.runtime import PaperRuntime
from mini_harness.rag.models import SearchQuery

def main():
    data=json.loads((ROOT/'docs/rag-eval-dataset.json').read_text(encoding='utf-8'))
    runtime=PaperRuntime();session=Session()
    try:
        with metric_scope(session):
            result=runtime.search(SearchQuery('自注意力如何调整感受野？',paper_ids=(data['paper_id'],)))
        rows=[dict(type=e.type,data=dict(e.data)) for e in session.events]
        trace=next(r['data'] for r in rows if r['type']=='rag/trace')
        assert {'retrieve','bm25','query_embedding','knn','fusion'} <= {s['name'] for s in trace['spans']}
        assert result.evidence and trace['status']=='ok'
        report=dict(validation='real local ES and E5; no remote model call',events=rows)
        (ROOT/'docs/rag-trace-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report))
    finally:runtime.close()
if __name__=='__main__':main()
