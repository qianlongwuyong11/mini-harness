"""Read-only queries against the published corpus; compare fresh-process snapshots across ES restart."""
import argparse
import json
import statistics
import time

from vector_search import ROOT,ARTIFACT,CATALOG,CONFIG,load_artifacts
from chunking_eval import atomic_write
from embedding_eval import metrics
from mini_harness.rag.catalog import SQLitePaperCatalog
from mini_harness.rag.embedding import LocalE5Embedding,load_default_config
from mini_harness.rag.es_service import ESService,ESConfig
from mini_harness.rag.models import SearchQuery,SourceKind
from mini_harness.rag.vector_index import EmbeddingProfile,ElasticsearchVectorIndex,VectorSearchService


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--compare',action='store_true')
    args=parser.parse_args()
    path=ROOT/'docs/vector-index-report.json'
    previous=json.loads(path.read_text(encoding='utf-8')) if args.compare else None
    cfg=load_default_config()
    embedding=LocalE5Embedding(cfg,ROOT/'.mini-harness/rag/models'/cfg.model_id.split('/')[-1])
    catalog=SQLitePaperCatalog(CATALOG)
    service=ESService(ESConfig.load(CONFIG))
    try:
        service.health()
        index=ElasticsearchVectorIndex(service.client,catalog,EmbeddingProfile.from_metadata(embedding.metadata()),analyzer='english')
        index.ensure_index()
        _,version,_,chunks=load_artifacts(ARTIFACT)
        assert index.verify_version(version,chunks)==106
        query_service=VectorSearchService(index,embedding)
        questions=json.loads((ROOT/'docs/embedding-queries.json').read_text(encoding='utf-8'))
        latencies,ranks=[],[]
        for repeat in range(3):
            for question in questions:
                started=time.perf_counter()
                hits=query_service.search(SearchQuery(question['query'],top_k=5))
                latencies.append((time.perf_counter()-started)*1000)
                assert all(h.chunk.version==version.version and h.chunk.source_kind is SourceKind.PAPER for h in hits)
                if repeat==0:ranks.append([h.chunk.ordinal for h in hits])
        vector=embedding.embed_query(questions[0]['query'])
        assert not index.vector_search(SearchQuery('q',paper_ids=('absent',)),vector,limit=5)
        assert not index.vector_search(SearchQuery('q',source_kinds=(SourceKind.USER_ANNOTATION,)),vector,limit=5)
        report=dict(index=index.index_name,count=106,quality=metrics(ranks,questions),rankings=ranks,
            warm_query_encode_es_catalog_ms_median=statistics.median(latencies),
            warm_query_encode_es_catalog_ms_p95=sorted(latencies)[int(.95*(len(latencies)-1))],
            excludes='Model startup and LLM generation', restart_verified=False)
        if previous:
            assert previous['rankings']==ranks, 'Ranking changed after restart'
            report['restart_verified']=True
            report['before_restart_query_ms_median']=previous['warm_query_encode_es_catalog_ms_median']
        atomic_write(path,json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(report,ensure_ascii=True))
    finally:
        service.close()


if __name__=='__main__':main()
