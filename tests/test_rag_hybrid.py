from dataclasses import asdict,replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from mini_harness.rag.hybrid import HybridConfig,HybridSearchService,expand_query,reciprocal_rank_fusion
from mini_harness.rag.models import PaperVersion,RetrievalChannel,RetrievalHit,SearchQuery
from mini_harness.rag.vector_index import EmbeddingProfile
from test_rag_elasticsearch import chunk


class HybridTests(unittest.TestCase):
    def test_rrf_uses_ranks_not_scores_and_deduplicates(self):
        v=PaperVersion('p','v','h','parser','chunker')
        a,b=chunk(v,0),chunk(v,1)
        lexical=[RetrievalHit(a,RetrievalChannel.BM25,1,100000),RetrievalHit(a,RetrievalChannel.BM25,1,100000)]
        vector=[RetrievalHit(b,RetrievalChannel.VECTOR,1,1),RetrievalHit(a,RetrievalChannel.VECTOR,2,.5)]
        rows=reciprocal_rank_fusion(lexical,vector,HybridConfig())
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['chunk'],a)
        self.assertAlmostEqual(rows[0]['score'],1/61+1/62)
        self.assertEqual(rows[0]['vector_rank'],2)

    def test_expansion_is_visible_and_can_be_disabled(self):
        text='自注意力的感受野'
        rewritten,terms=expand_query(text)
        self.assertTrue(rewritten.startswith(text))
        self.assertEqual(terms['感受野'],'receptive field')
        self.assertEqual(expand_query(text,False),(text,{}))
        self.assertEqual(expand_query('unknown term'),('unknown term',{}))

    def test_invalid_config_and_conflicting_payload(self):
        for kwargs in ({'candidate_limit':0},{'rrf_k':0},{'bm25_weight':float('nan')}):
            with self.assertRaises(ValueError):HybridConfig(**kwargs)
        v=PaperVersion('p','v','h','p','c')
        a=chunk(v)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([RetrievalHit(a,RetrievalChannel.BM25,1,1)],
                [RetrievalHit(replace(a,text='different'),RetrievalChannel.VECTOR,1,1)],HybridConfig())

    def test_same_scope_and_activation_during_routes_is_removed(self):
        profile=EmbeddingProfile('model','a'*40,2,'b'*64)
        v=PaperVersion('p','v','h','p','c',embedding_model='model',embedding_revision='a'*40)
        a=chunk(v)
        catalog=Mock()
        catalog.list_active_versions.side_effect=[(v,),()]
        embedding=Mock()
        embedding.metadata.return_value=asdict(profile)
        index=SimpleNamespace(profile=profile,catalog=catalog,client=Mock(),index_name='index',analyzer='english')
        observed=[]
        class FrozenIndex:
            def __init__(self,client,catalog,profile,*args,**kwargs):self.catalog,self.profile=catalog,profile
            def lexical_search(self,q,**kwargs):
                observed.append((self.catalog.list_active_versions(),q.paper_ids,q.source_kinds))
                return (RetrievalHit(a,RetrievalChannel.BM25,1,1),)
            def vector_search(self,q,vec,**kwargs):
                observed.append((self.catalog.list_active_versions(),q.paper_ids,q.source_kinds))
                return (RetrievalHit(a,RetrievalChannel.VECTOR,1,1),)
        with patch('mini_harness.rag.hybrid.ElasticsearchVectorIndex',FrozenIndex):
            result,trace=HybridSearchService(index,embedding).search_with_trace(SearchQuery('q',paper_ids=('p',)))
        self.assertEqual(observed[0],observed[1])
        self.assertEqual(result.evidence,())
        self.assertEqual(trace['fusion'],[])
