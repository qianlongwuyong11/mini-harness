from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import httpx

from mini_harness.rag.catalog import SQLitePaperCatalog
from mini_harness.rag.elasticsearch import BulkWriteError, ElasticsearchPaperIndex
from mini_harness.rag.interfaces import PaperIndex
from mini_harness.rag.models import (Chunk, PageSpan, PaperVersion, SearchQuery,
    SourceKind, VersionState, content_hash, stable_chunk_id)


def chunk(version, ordinal=0):
    text = 'SparseBEV uses scale-adaptive self attention.'
    return Chunk(stable_chunk_id(version.paper_id,version.version,version.chunker_version,ordinal),
        version.paper_id,version.version,ordinal,text,content_hash(text),8,
        SourceKind.PAPER,('Method',),(PageSpan(3,3),),('block-1',))


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.catalog=SQLitePaperCatalog(Path(self.tmp.name)/'catalog.sqlite3')
        job=self.catalog.enqueue_import('fixture://paper','hash',parser_version='p1',chunker_version='c1',title='Fixture')
        self.version=self.catalog.get_version(job.paper_id,job.target_version)
        self.chunk=chunk(self.version)
        self.requests=[]

    def index(self,handler,catalog=None,**kwargs):
        def transport(request):
            self.requests.append(request)
            return handler(request)
        client=httpx.Client(base_url='https://localhost:9200',transport=httpx.MockTransport(transport))
        self.addCleanup(client.close)
        return ElasticsearchPaperIndex(client,catalog or self.catalog,**kwargs)

    def test_partial_bulk_failure_is_reported_even_with_http_200(self):
        def handler(request):
            rows=request.content.decode().splitlines()
            ids=[json.loads(rows[i])['index']['_id'] for i in range(0,len(rows),2)]
            self.assertEqual(request.url.params['refresh'],'wait_for')
            self.assertTrue(request.content.endswith(b'\n'))
            return httpx.Response(200,json={'errors':True,'items':[
                {'index':{'_id':ids[0],'status':201}},
                {'index':{'_id':ids[1],'status':429,'error':{'type':'es_rejected_execution_exception'}}}]})
        index=self.index(handler)
        with self.assertRaises(BulkWriteError) as exc:
            index.upsert_chunks(self.version,[self.chunk,chunk(self.version,1)],{})
        self.assertEqual(exc.exception.failures[0][1],429)
        self.assertIsNone(self.catalog.get_active_version(self.version.paper_id))
        self.assertIsInstance(index,PaperIndex)

    def test_batching_stable_ids_and_missing_acknowledgement(self):
        ids=[]
        def handler(request):
            rows=request.content.decode().splitlines()
            self.assertLessEqual(len(rows),2)
            key=json.loads(rows[0])['index']['_id']
            ids.append(key)
            return httpx.Response(200,json={'errors':False,'items':[{'index':{'_id':key,'status':200}}]})
        index=self.index(handler,batch_size=1)
        for _ in range(2):
            index.upsert_chunks(self.version,[self.chunk,chunk(self.version,1)],{})
        self.assertEqual(ids[:2],ids[2:])
        broken=self.index(lambda _: httpx.Response(200,json={'errors':False,'items':[]}))
        with self.assertRaises(RuntimeError): broken.upsert_chunks(self.version,[self.chunk],{})

    def test_invalid_batch_is_rejected_before_network(self):
        index=self.index(lambda _: self.fail('Unexpected network write'))
        for bad in (replace(self.chunk,content_hash='bad'),replace(self.chunk,version='other'),replace(self.chunk,chunk_id='arbitrary')):
            with self.assertRaises(ValueError):index.upsert_chunks(self.version,[self.chunk,bad],{})
        with self.assertRaises(ValueError):index.upsert_chunks(self.version,[self.chunk],{'x':(1.0,)})
        self.assertFalse(self.requests)

    def test_no_active_versions_means_no_search(self):
        index=self.index(lambda _: self.fail('Unscoped query sent'))
        self.assertEqual(index.lexical_search(SearchQuery('SparseBEV'),limit=5),())
        self.assertEqual(self.catalog.list_active_versions(),())

    def test_version_pairs_prefilter_and_racing_delete(self):
        catalog=Mock()
        v1=replace(self.version,state=VersionState.READY,chunk_count=1)
        v2=replace(v1,paper_id='paper-b',version='different')
        catalog.list_active_versions.side_effect=[(v1,v2),()]
        def handler(request):
            body=json.loads(request.content)
            filters=body['query']['bool']['filter']
            self.assertEqual(filters[1],{'terms':{'source_kind':['paper']}})
            pairs=filters[0]['bool']['should']
            self.assertEqual(len(pairs),2)
            self.assertTrue(all(len(p['bool']['filter'])==2 for p in pairs))
            return httpx.Response(200,json={'hits':{'hits':[{'_score':1.0,'_source':{'chunk':self.chunk.to_dict()}}]}})
        index=self.index(handler,catalog)
        self.assertEqual(index.lexical_search(SearchQuery('SparseBEV'),limit=3),())

    def test_hit_roundtrip_and_explicit_paper_filter(self):
        catalog=Mock()
        v=replace(self.version,state=VersionState.READY,chunk_count=1)
        catalog.list_active_versions.return_value=(v,)
        index=self.index(lambda _:httpx.Response(200,json={'hits':{'hits':[{'_score':2.0,'_source':{'chunk':self.chunk.to_dict()}}]}}),catalog)
        hits=index.lexical_search(SearchQuery('SparseBEV',paper_ids=(v.paper_id,)),limit=3)
        self.assertEqual(hits[0].chunk,self.chunk)
        self.assertEqual(hits[0].rank,1)
        self.assertEqual(index.lexical_search(SearchQuery('SparseBEV',paper_ids=('other',)),limit=3),())

    def test_mapping_mismatch_rejected(self):
        def handler(request):
            if request.method=='PUT':
                return httpx.Response(400,json={'error':{'type':'resource_already_exists_exception'}})
            return httpx.Response(200,json={'mini-harness-papers-bm25-v1':{'mappings':{}}})
        with self.assertRaises(ValueError):self.index(handler).ensure_index()

    def test_partial_search_and_failed_cleanup_rejected(self):
        catalog=Mock()
        catalog.list_active_versions.return_value=(replace(self.version,state=VersionState.READY,chunk_count=1),)
        index=self.index(lambda _:httpx.Response(200,json={'timed_out':True}),catalog)
        with self.assertRaises(RuntimeError):index.lexical_search(SearchQuery('SparseBEV'),limit=3)
        catalog.get_active_version.return_value=self.version
        with self.assertRaises(ValueError):index.delete_version(self.version.paper_id,self.version.version)
        catalog.get_active_version.return_value=None
        with self.assertRaises(RuntimeError):index.delete_version(self.version.paper_id,self.version.version)

if __name__=='__main__':unittest.main()
