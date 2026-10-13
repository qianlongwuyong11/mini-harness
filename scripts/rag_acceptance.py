"""Reproducible local-only retrieval acceptance. Never calls a chat API."""
import argparse, hashlib, json, math, statistics, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vector_search import ARTIFACT,CATALOG,CONFIG
from mini_harness.rag.catalog import SQLitePaperCatalog
from mini_harness.rag.embedding import LocalE5Embedding,load_default_config
from mini_harness.rag.es_service import ESService,ESConfig
from mini_harness.rag.models import SearchQuery
from mini_harness.rag.vector_index import EmbeddingProfile,ElasticsearchVectorIndex
from mini_harness.rag.hybrid import HybridSearchService


def metrics(rows):
    from mini_harness.rag.adapters.evaluation import evaluate_rankings
    return evaluate_rankings(rows)


def latency(values):
    values=sorted(values)
    return dict(n=len(values),median_ms=statistics.median(values),p95_ms=values[math.ceil(.95*len(values))-1],max_ms=max(values))


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--output',type=Path,default=ROOT/'docs/rag-acceptance-report.json');cli.add_argument('--replay-report',type=Path);args=cli.parse_args()
    if args.replay_report:
        raw=args.replay_report.read_bytes();previous=json.loads(raw)
        cases=previous['cases']
        scores={split:{route:metrics([(c['rankings'][route],c['gold']) for c in cases if c['split']==split]) for route in cases[0]['rankings']} for split in ('development','heldout')}
        report=dict(evaluator='LlamaIndex RetrieverEvaluator',mode='replay captured rankings; no retrieval or LLM calls',source_sha256=hashlib.sha256(raw).hexdigest(),quality=scores,agrees_with_previous=scores==previous['quality'],empty_result_policy='zero scores; installed LlamaIndex metrics reject empty retrieved_ids')
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report));return
    data_path=ROOT/'docs/rag-eval-dataset.json';data=json.loads(data_path.read_text(encoding='utf-8'))
    corpus=(ARTIFACT/'chunks.json').read_bytes()
    if hashlib.sha256(corpus).hexdigest()!=data['corpus_sha256']:raise ValueError('Corpus changed: reannotate dataset')
    cfg=load_default_config();started=time.perf_counter()
    embedding=LocalE5Embedding(cfg,ROOT/'.mini-harness/rag/models'/cfg.model_id.split('/')[-1])
    load_ms=(time.perf_counter()-started)*1000
    catalog=SQLitePaperCatalog(CATALOG);es=ESService(ESConfig.load(CONFIG))
    try:
        es.health();v=catalog.get_active_version(data['paper_id'])
        if not v or v.version!=data['version']:raise ValueError('Active version differs from annotation')
        index=ElasticsearchVectorIndex(es.client,catalog,EmbeddingProfile.from_metadata(embedding.metadata()),analyzer='english')
        service=HybridSearchService(index,embedding);cases=[]
        # Warm query is outside the fixed scoring set; first-use latency reported separately.
        t=time.perf_counter();service.search(SearchQuery('SparseBEV',paper_ids=(data['paper_id'],)));first_ms=(time.perf_counter()-t)*1000
        for q in data['questions']:
            query=SearchQuery(q['query'],paper_ids=(data['paper_id'],),top_k=5)
            t=time.perf_counter();plain=index.lexical_search(query,limit=20);plain_ms=(time.perf_counter()-t)*1000
            result,trace=service.search_with_trace(query)
            ranked=dict(bm25=[h.chunk.chunk_id for h in plain[:5]],bm25_expanded=[h['chunk']['chunk_id'] for h in trace['bm25_candidates'][:5]],vector=[h['chunk']['chunk_id'] for h in trace['vector_candidates'][:5]],hybrid=[e.chunk_id for e in result.evidence])
            cases.append(dict(id=q['id'],split=q['split'],rankings=ranked,gold=[e['chunk_id'] for e in q['evidence']],timings={k:trace[k] for k in ('encoding_ms','bm25_ms','knn_ms','total_ms')},bm25_original_ms=plain_ms))
        quality={split:{route:metrics([(c['rankings'][route],c['gold']) for c in cases if c['split']==split]) for route in cases[0]['rankings']} for split in ('development','heldout')}
        stats=es.client.get('/'+index.index_name+'/_stats/store,docs,translog');stats.raise_for_status()
        held=quality['heldout'];gates=dict(heldout_hybrid_hit_at_5_ge_075=held['hybrid']['hit@5']>=.75,heldout_hybrid_recall_not_below_vector=held['hybrid']['recall@5']>=held['vector']['recall@5'])
        report=dict(evaluator='LlamaIndex RetrieverEvaluator; recall/hit_rate/mrr; replay captured rankings; empty retrieval=0',dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),config=service.config.__dict__,embedding=embedding.metadata(),model_load_ms=load_ms,first_query_ms=first_ms,quality=quality,latency={k:latency([c['timings'][k] for c in cases]) for k in cases[0]['timings']},index=index.index_name,index_stats=stats.json()['_all'],cases=cases,gates=gates,passed=all(gates.values()),limitations=['Single paper, 8 newly authored heldout questions; not independent human labels or multi-paper validation','Latency is one run per query, p95 from 20 samples','OS file cache not cleared; process cold is not disk cold','Gold annotation is a selected relevant set, not exhaustive relevance judgments'])
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(quality=quality,latency=report['latency'],gates=gates),ensure_ascii=True))
    finally:es.close()
if __name__=='__main__':main()
