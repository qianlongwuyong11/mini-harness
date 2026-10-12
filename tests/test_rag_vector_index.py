from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import httpx

from mini_harness.rag.catalog import SQLitePaperCatalog
from mini_harness.rag.elasticsearch import BulkWriteError
from mini_harness.rag.models import IngestionStage, IngestionStatus, SearchQuery, VersionState
from mini_harness.rag.vector_index import EmbeddingProfile,ElasticsearchVectorIndex,VectorSearchService
from mini_harness.rag.vector_pipeline import encode_index_publish
from test_rag_elasticsearch import chunk


class VectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.catalog=SQLitePaperCatalog(Path(self.tmp.name)/'catalog.sqlite3')
        self.profile=EmbeddingProfile('test-model','a'*40,2,'b'*64)
        self.job=self.catalog.enqueue_import('fixture://paper','hash',parser_version='p',chunker_version='c',
            embedding_model=self.profile.model_id,embedding_revision=self.profile.revision)
        self.version=self.catalog.get_version(self.job.paper_id,self.job.target_version)
        self.chunk=chunk(self.version)
        self.requests=[]

    def index(self,handler,catalog=None):
        def dispatch(request):
            self.requests.append(request)
            return handler(request)
        client=httpx.Client(base_url='https://localhost:9200',transport=httpx.MockTransport(dispatch))
        self.addCleanup(client.close)
        return ElasticsearchVectorIndex(client,catalog or self.catalog,self.profile,batch_size=1)

    def test_bad_vectors_and_identity_fail_before_network(self):
        index=self.index(lambda _:self.fail('Unexpected write'))
        for vector in ((1.,), (0.,0.), (float('nan'),0.), (True,0.)):
            with self.assertRaises(ValueError):
                index.upsert_chunks(self.version,[self.chunk],{self.chunk.chunk_id:vector})
        with self.assertRaises(ValueError):index.upsert_chunks(self.version,[self.chunk],{})
        with self.assertRaises(ValueError):index.upsert_chunks(replace(self.version,embedding_revision='c'*40),[self.chunk],{self.chunk.chunk_id:(1.,0.)})
        self.assertFalse(self.requests)

    def test_bulk_has_vector_and_partial_failure_does_not_publish(self):
        def handler(request):
            rows=request.content.decode().splitlines()
            doc=json.loads(rows[1])
            self.assertEqual(doc['embedding'],[1.,0.])
            self.assertEqual(doc['embedding_profile'],self.profile.fingerprint)
            self.assertIn('chunk',doc)
            return httpx.Response(200,json={'errors':True,'items':[{'index':{'_id':self.chunk.chunk_id,'status':429,'error':{'type':'rejected'}}}]})
        index=self.index(handler)
        with self.assertRaises(BulkWriteError):index.upsert_chunks(self.version,[self.chunk],{self.chunk.chunk_id:(1.,0.)})
        self.assertIsNone(self.catalog.get_active_version(self.version.paper_id))

    def test_prefilter_contains_version_source_and_profile_and_race_is_rechecked(self):
        ready=replace(self.version,state=VersionState.READY,chunk_count=1)
        catalog=Mock()
        catalog.list_active_versions.side_effect=[(ready,),()]
        def handler(request):
            body=json.loads(request.content)
            filters=body['knn']['filter']
            self.assertEqual(filters[0]['bool']['should'][0]['bool']['filter'],[
                {'term':{'paper_id':ready.paper_id}},{'term':{'version':ready.version}}])
            self.assertEqual(filters[1],{'terms':{'source_kind':['paper']}})
            self.assertEqual(filters[2],{'term':{'embedding_profile':self.profile.fingerprint}})
            return httpx.Response(200,json={'hits':{'hits':[{'_score':1,'_source':{'chunk':self.chunk.to_dict()}}]}})
        self.assertEqual(self.index(handler,catalog).vector_search(SearchQuery('query'),(1.,0.),limit=5),())

    def test_staging_hidden_and_query_encoder_identity_checked(self):
        index=self.index(lambda _:self.fail('Unfiltered query'))
        self.assertEqual(index.vector_search(SearchQuery('q'),(1.,0.),limit=5),())
        encoder=Mock()
        from dataclasses import asdict
        encoder.metadata.return_value=asdict(replace(self.profile,tokenizer_sha256='c'*64))
        with self.assertRaises(ValueError):VectorSearchService(index,encoder)

    def test_mapping_model_revision_change_requires_new_index(self):
        index=self.index(lambda _:httpx.Response(400,json={'error':{'type':'resource_already_exists_exception'}}))
        desired=index.mapping()
        from dataclasses import asdict
        self.assertEqual(desired['properties']['embedding']['dims'],2)
        changed=replace(self.profile,revision='c'*40)
        self.assertNotEqual(changed.fingerprint,self.profile.fingerprint)
        desired['properties']['embedding'].pop('element_type')
        self.assertTrue(index._mapping_matches(desired))
        desired['_meta']['embedding']['revision']='c'*40
        self.assertFalse(index._mapping_matches(desired))

    def test_targeted_claim_and_pipeline_failure_record(self):
        later=self.catalog.enqueue_import('fixture://other','other',parser_version='p',chunker_version='c')
        picked=self.catalog.claim_job('other-worker',job_id=later.job_id)
        self.assertEqual(picked.job_id,later.job_id)
        owned=self.catalog.claim_job('worker',job_id=self.job.job_id)
        for stage in (IngestionStage.CHUNKING,IngestionStage.EMBEDDING):
            owned=self.catalog.advance_job(owned,stage)
        encoder=Mock()
        from dataclasses import asdict
        encoder.metadata.return_value=asdict(self.profile)
        encoder.config=SimpleNamespace(batch_size=4)
        encoder.embed_documents.return_value=((1.,0.),)
        index=Mock(profile=self.profile)
        index.upsert_chunks.side_effect=RuntimeError('simulated bulk failure')
        with self.assertRaisesRegex(RuntimeError,'bulk failure'):
            encode_index_publish(self.catalog,index,encoder,owned,[self.chunk])
        self.assertEqual(self.catalog.get_job(self.job.job_id).status,IngestionStatus.FAILED)
        self.assertIsNone(self.catalog.get_active_version(self.version.paper_id))
        index.verify_version.assert_not_called()
