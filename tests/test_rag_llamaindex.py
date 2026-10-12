import asyncio
from dataclasses import replace
import importlib.util
import unittest
from unittest.mock import Mock, patch

from mini_harness.rag.models import (
    BlockKind, Evidence, PageSpan, RetrievalMode, SearchQuery, SearchResult,
    SourceBlock, SourceKind, PaperVersion,
)
from test_rag_elasticsearch import chunk

AVAILABLE = importlib.util.find_spec('llama_index') is not None


@unittest.skipUnless(AVAILABLE, 'Install rag-adapter extra')
class AdapterTests(unittest.TestCase):
    def test_structural_nodes_preserve_text_and_metadata(self):
        from mini_harness.rag.adapters.llamaindex import to_node
        from llama_index.core.schema import MetadataMode
        block = SourceBlock('b','p','v','  equation: x = y\n|a|b|',BlockKind.TEXT,
                            SourceKind.PAPER,('section',),(PageSpan(1,2),))
        for value in (block, chunk(PaperVersion('p','v','h','p','c'))):
            node = to_node(value)
            self.assertEqual(node.get_content(metadata_mode=MetadataMode.EMBED),value.text)
            self.assertEqual(node.get_content(metadata_mode=MetadataMode.LLM),value.text)
            self.assertEqual(node.metadata['page_spans'],value.to_dict()['page_spans'])

    def test_search_scope_scores_and_empty_results(self):
        from mini_harness.rag.adapters.llamaindex import LlamaIndexSearchAdapter
        query = SearchQuery('中文问题',('p',),(SourceKind.USER_ANNOTATION,),2)
        evidence = Evidence('e','p','v','c','title','exact\ntext',SourceKind.USER_ANNOTATION,
                            ('section',),(PageSpan(2,2),),retrieval_score=0.03,bm25_rank=2,vector_rank=1)
        for items in ((evidence,),()):
            result = SearchResult(query,items,RetrievalMode.HYBRID,10)
            backend = Mock()
            backend.search.return_value = result
            with patch.dict('os.environ',{},clear=True):
                actual = LlamaIndexSearchAdapter(backend).search(query)
            self.assertEqual(actual,result)
            backend.search.assert_called_once_with(query)

    def test_embedding_uses_injected_encoder_once_without_prefix_changes(self):
        from mini_harness.rag.adapters.llamaindex import LocalEmbeddingAdapter
        service = Mock(model_id='local-e5')
        service.embed_documents.return_value = ((1.,0.),(0.,1.))
        service.embed_query.return_value = (1.,0.)
        adapter = LocalEmbeddingAdapter(service)
        self.assertEqual(adapter.get_text_embedding_batch(['a','b']),[[1.,0.],[0.,1.]])
        service.embed_documents.assert_called_once_with(['a','b'])
        self.assertEqual(asyncio.run(adapter.aget_query_embedding('中文')), [1.,0.])
        service.embed_query.assert_called_once_with('中文')

    def test_backend_failure_propagates(self):
        from mini_harness.rag.adapters.llamaindex import LlamaIndexSearchAdapter
        backend = Mock()
        backend.search.side_effect = RuntimeError('ES unavailable')
        with self.assertRaisesRegex(RuntimeError,'ES unavailable'):
            LlamaIndexSearchAdapter(backend).search(SearchQuery('q'))
