"""Synthetic BM25 baseline, deliberately not presented as extracted paper text."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from local_es import BASE, CONFIG, HOME
from mini_harness.rag.catalog import SQLitePaperCatalog
from mini_harness.rag.elasticsearch import ElasticsearchPaperIndex
from mini_harness.rag.es_service import ESConfig, ESService
from mini_harness.rag.models import (BlockKind, Chunk, IngestionStage, PageSpan,
    SearchQuery, SourceBlock, SourceKind, content_hash, stable_chunk_id)

# Small hand-written fixtures test plumbing, not paper factual accuracy/Recall@K.
CORPUS = [
    ('Overview', 'SparseBEV is a synthetic keyword fixture for multi-camera 3D object detection.'),
    ('Scale-adaptive Self Attention', 'Scale-adaptive self attention adjusts interactions between object queries.'),
    ('Experiments', 'Experiments evaluate detection on the nuScenes dataset with NDS and mAP metrics.'),
    ('Sampling', 'Adaptive sampling selects image features for object queries.'),
    ('Terminology', 'Detectors use multi-camera images; detection and detecting are related word forms.'),
]


def import_fixture(catalog, index, source_hash='fixture-a', parser='fixture-v1', annotation=False):
    job = catalog.enqueue_import('synthetic://fixture', source_hash, parser_version=parser,
                                 chunker_version='fixture-chunker-v1', title='Synthetic perception fixture')
    version = catalog.get_version(job.paper_id, job.target_version)
    chunks, blocks = [], []
    for ordinal, (section, text) in enumerate(CORPUS):
        kind = SourceKind.USER_ANNOTATION if annotation else SourceKind.PAPER
        block = SourceBlock(f'block-{ordinal}', version.paper_id, version.version, text,
            BlockKind.TEXT, kind, (section,), (PageSpan(ordinal+1,ordinal+1),))
        chunk = Chunk(stable_chunk_id(version.paper_id,version.version,version.chunker_version,ordinal),
            version.paper_id,version.version,ordinal,text,content_hash(text),len(text.split()),
            kind,block.section_path,block.page_spans,(block.block_id,))
        blocks.append(block)
        chunks.append(chunk)
    catalog.save_blocks(blocks)
    index.upsert_chunks(version,chunks,{})
    index.upsert_chunks(version,chunks,{})
    assert index.count_version(version.paper_id,version.version) == len(chunks), 'Retry duplicated chunks'
    return job, version, chunks


def publish(catalog, index, job, chunks):
    owned = catalog.claim_job('bm25-baseline')
    assert owned.job_id == job.job_id
    for stage in (IngestionStage.CHUNKING, IngestionStage.EMBEDDING, IngestionStage.INDEXING, IngestionStage.PUBLISHING):
        owned = catalog.advance_job(owned,stage)
    catalog.complete_job(owned,indexed_chunk_count=index.count_version(job.paper_id,job.target_version),
                         expected_chunk_count=len(chunks),index_searchable=True)


def baseline(service):
    report = {'fixture_type':'hand-written synthetic snippets; not PDF extraction or quality benchmark',
              'es_version':service.client.get('/').json()['version']['number'], 'analyzers':{}}
    for analyzer in ('standard','english'):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = SQLitePaperCatalog(Path(tmp)/'catalog.sqlite3')
            name = 'mini-harness-bm25-test-' + secrets.token_hex(8)
            index = ElasticsearchPaperIndex(service.client,catalog,name,analyzer=analyzer,batch_size=2)
            index.ensure_index()
            try:
                index.ensure_index()  # Existing mapping must remain compatible.
                job, version, chunks = import_fixture(catalog,index)
                assert not index.lexical_search(SearchQuery('SparseBEV'),limit=3), 'Staging version leaked'
                publish(catalog,index,job,chunks)
                tokens = {}
                queries = ('SparseBEV','nuScenes','scale-adaptive self attention','multi-camera','detection')
                ranked = {}
                for query in queries:
                    response = service.client.post('/'+name+'/_analyze',json={'analyzer':analyzer,'text':query})
                    response.raise_for_status()
                    tokens[query]=[item['token'] for item in response.json()['tokens']]
                    hits=index.lexical_search(SearchQuery(query),limit=3)
                    assert hits, query
                    ranked[query]=[dict(section=h.chunk.section_path,score=h.score,rank=h.rank) for h in hits]
                assert ranked['SparseBEV'][0]['section'] == ('Overview',)
                assert ranked['nuScenes'][0]['section'] == ('Experiments',)
                assert ranked['scale-adaptive self attention'][0]['section'] == ('Scale-adaptive Self Attention',)
                assert not index.lexical_search(SearchQuery('SparseBEV',paper_ids=('absent',)),limit=3)
                # A second version is indexed but cannot replace the published version until publication.
                next_job, next_version, next_chunks = import_fixture(catalog,index,parser='fixture-v2')
                assert all(h.chunk.version == version.version for h in index.lexical_search(SearchQuery('SparseBEV'),limit=3))
                publish(catalog,index,next_job,next_chunks)
                assert all(h.chunk.version == next_version.version for h in index.lexical_search(SearchQuery('SparseBEV'),limit=3))
                index.delete_version(version.paper_id,version.version)
                assert index.count_version(version.paper_id,version.version)==0
                annotation_job, annotation_version, annotation_chunks=import_fixture(catalog,index,'fixture-annotation',annotation=True)
                publish(catalog,index,annotation_job,annotation_chunks)
                assert all(h.chunk.source_kind is SourceKind.PAPER for h in index.lexical_search(SearchQuery('SparseBEV'),limit=3))
                assert index.lexical_search(SearchQuery('SparseBEV',source_kinds=(SourceKind.USER_ANNOTATION,)),limit=3)
                catalog.soft_delete_paper(version.paper_id)
                assert not index.lexical_search(SearchQuery('SparseBEV'),limit=3), 'Soft-deleted paper leaked'
                report['analyzers'][analyzer]={'tokens':tokens,'top_k':ranked,
                    'checks':['retry_idempotency','staging_hidden','paper_scope','activation_switch','old_version_cleanup','annotation_isolation','soft_delete']}
            finally:
                service.client.delete('/'+name).raise_for_status()
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--start-es',action='store_true',help='Start local ES only if port is unused; stop owned process afterward')
    parser.add_argument('--report',default=str(ROOT/'docs/bm25-baseline.json'))
    args=parser.parse_args()
    service=ESService(ESConfig.load(CONFIG))
    process=None
    log=None
    try:
        with socket.socket() as sock:
            occupied=sock.connect_ex(('127.0.0.1',9200))==0
        if not occupied and args.start_es:
            log=(BASE/'baseline-launcher.log').open('ab')
            process=subprocess.Popen([str(HOME/'bin/elasticsearch.bat')],cwd=HOME,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
            for _ in range(180):
                if process.poll() is not None:
                    raise RuntimeError('ES exited; check baseline-launcher.log')
                try:
                    service.health()
                    break
                except Exception:
                    time.sleep(1)
        service.health()
        report=baseline(service)
        Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('PASS: both analyzers, exact-term top hits, lifecycle filters and idempotent bulk. Report: '+args.report)
    finally:
        if process is not None and process.poll() is None:
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],check=True,capture_output=True)
            process.wait(timeout=30)
        if log: log.close()
        service.close()

if __name__=='__main__':
    main()
