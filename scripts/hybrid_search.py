"""Step 9 query CLI and fixed-set comparison, without index writes."""
import argparse
from dataclasses import asdict
import json
import statistics

from vector_search import ROOT,CATALOG,CONFIG
from chunking_eval import atomic_write
from embedding_eval import metrics
from mini_harness.rag.catalog import SQLitePaperCatalog
from mini_harness.rag.embedding import LocalE5Embedding,load_default_config
from mini_harness.rag.es_service import ESService,ESConfig
from mini_harness.rag.models import SearchQuery
from mini_harness.rag.vector_index import EmbeddingProfile,ElasticsearchVectorIndex
from mini_harness.rag.hybrid import HybridConfig,HybridSearchService


def compact(trace):
    return {**trace,**{name:[dict(chunk_id=h['chunk']['chunk_id'],ordinal=h['chunk']['ordinal'],rank=h['rank'],score=h['score'])
                           for h in trace[name]] for name in ('bm25_candidates','vector_candidates')}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('query',nargs='?',default='自注意力如何调整感受野？')
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--top-k',type=int,default=5)
    parser.add_argument('--candidates',type=int,default=20)
    parser.add_argument('--rrf-k',type=int,default=60)
    parser.add_argument('--bm25-weight',type=float,default=1.)
    parser.add_argument('--vector-weight',type=float,default=1.)
    parser.add_argument('--no-expansion',action='store_true')
    args=parser.parse_args()
    cfg=load_default_config()
    embedding=LocalE5Embedding(cfg,ROOT/'.mini-harness/rag/models'/cfg.model_id.split('/')[-1])
    catalog=SQLitePaperCatalog(CATALOG)
    es=ESService(ESConfig.load(CONFIG))
    try:
        es.health()
        index=ElasticsearchVectorIndex(es.client,catalog,EmbeddingProfile.from_metadata(embedding.metadata()),analyzer='english')
        config=HybridConfig(args.candidates,args.rrf_k,args.bm25_weight,args.vector_weight,not args.no_expansion)
        service=HybridSearchService(index,embedding,config)
        if not args.evaluate:
            result,trace=service.search_with_trace(SearchQuery(args.query,top_k=args.top_k))
            print(json.dumps(dict(result=result.to_dict(),trace=compact(trace)),ensure_ascii=True,indent=2))
            return
        questions=json.loads((ROOT/'docs/embedding-queries.json').read_text(encoding='utf-8'))
        rankings={name:[] for name in ('bm25_original','bm25_expanded','vector','hybrid')}
        traces=[]
        for q in questions:
            query=SearchQuery(q['query'],top_k=5)
            original=index.lexical_search(query,limit=max(5,config.candidate_limit))
            result,trace=service.search_with_trace(query)
            ids={h['chunk']['chunk_id']:h['chunk']['ordinal'] for route in ('bm25_candidates','vector_candidates') for h in trace[route]}
            rankings['bm25_original'].append([h.chunk.ordinal for h in original][:5])
            rankings['bm25_expanded'].append([h['chunk']['ordinal'] for h in trace['bm25_candidates']][:5])
            rankings['vector'].append([h['chunk']['ordinal'] for h in trace['vector_candidates']][:5])
            rankings['hybrid'].append([ids[e.chunk_id] for e in result.evidence])
            traces.append(dict(question_id=q['id'],**compact(trace)))
        scores={name:metrics(rows,questions) for name,rows in rankings.items()}
        report=dict(config=asdict(config),index=index.index_name,questions=questions,quality=scores,rankings=rankings,traces=traces,
                    query_ms_median=statistics.median(t['total_ms'] for t in traces),
                    baseline_recall_at_5=0.7083333333333334,
                    acceptance_passed=scores['hybrid']['recall@5']>=max(0.7083333333333334,scores['vector']['recall@5'],scores['bm25_original']['recall@5']),
                    limitations='Single paper; glossary and questions are domain-specific; no held-out generalization claim')
        atomic_write(ROOT/'docs/hybrid-search-report.json',json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k not in ('traces','questions')},ensure_ascii=True),flush=True)
    finally:
        es.close()


if __name__=='__main__':main()
