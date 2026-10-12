"""Compare live hybrid retrieval against the optional framework adapter (read-only)."""
import json
from hybrid_search import ROOT, CATALOG, CONFIG
from mini_harness.rag.adapters.llamaindex import LlamaIndexSearchAdapter
from mini_harness.rag.catalog import SQLitePaperCatalog
from mini_harness.rag.embedding import LocalE5Embedding, load_default_config
from mini_harness.rag.es_service import ESService, ESConfig
from mini_harness.rag.hybrid import HybridSearchService
from mini_harness.rag.models import SearchQuery
from mini_harness.rag.vector_index import EmbeddingProfile, ElasticsearchVectorIndex


def main():
    cfg = load_default_config()
    embedding = LocalE5Embedding(cfg, ROOT/'.mini-harness/rag/models'/cfg.model_id.split('/')[-1])
    catalog = SQLitePaperCatalog(CATALOG)
    es = ESService(ESConfig.load(CONFIG))
    try:
        es.health()
        index = ElasticsearchVectorIndex(es.client,catalog,EmbeddingProfile.from_metadata(embedding.metadata()),analyzer='english')
        backend = HybridSearchService(index,embedding)
        adapter = LlamaIndexSearchAdapter(backend)
        rows = []
        questions = json.loads((ROOT/'docs/embedding-queries.json').read_text(encoding='utf-8'))
        for question in questions:
            query = SearchQuery(question['query'],top_k=5)
            expected = backend.search(query)
            actual = adapter.search(query)
            if not expected.evidence:
                raise RuntimeError('Empty live index cannot validate retrieval equivalence')
            if expected != actual:
                raise AssertionError(question['id'])
            rows.append({'id':question['id'],'equal':True,'chunk_ids':[e.chunk_id for e in actual.evidence]})
        report = {'index':index.index_name,'queries':rows,'all_equal':True}
        (ROOT/'docs/llamaindex-adapter-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,ensure_ascii=True))
    finally:
        es.close()


if __name__ == '__main__':
    main()
