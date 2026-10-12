"""Real ES lifecycle/partial-bulk checks in a temporary index and SQLite catalog."""
from dataclasses import replace
import json
from pathlib import Path
import secrets
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from chunking_eval import atomic_write
from local_es import CONFIG
from mini_harness.rag.catalog import SQLitePaperCatalog
from mini_harness.rag.es_service import ESConfig,ESService
from mini_harness.rag.elasticsearch import BulkWriteError
from mini_harness.rag.models import (Chunk,SourceBlock,BlockKind,SourceKind,PageSpan,
    SearchQuery,IngestionStage,content_hash,stable_chunk_id)
from mini_harness.rag.vector_index import EmbeddingProfile,ElasticsearchVectorIndex


def main():
    service=ESService(ESConfig.load(CONFIG))
    name='mini-harness-vector-lifecycle-'+secrets.token_hex(8)
    created=False
    checks=[]
    try:
        service.health()
        with tempfile.TemporaryDirectory() as tmp:
            catalog=SQLitePaperCatalog(Path(tmp)/'catalog.sqlite3')
            profile=EmbeddingProfile('fixture','a'*40,2,'b'*64)
            index=ElasticsearchVectorIndex(service.client,catalog,profile,name)
            index.ensure_index()
            created=True
            def prepare(parser):
                job=catalog.enqueue_import('fixture://source','hash',parser_version=parser,chunker_version='c',
                    embedding_model=profile.model_id,embedding_revision=profile.revision)
                version=catalog.get_version(job.paper_id,job.target_version)
                blocks=[]
                chunks=[]
                for ordinal,kind in enumerate((SourceKind.PAPER,SourceKind.USER_ANNOTATION)):
                    block=SourceBlock('block-'+str(ordinal),version.paper_id,version.version,'fixture content',BlockKind.TEXT,kind,('Method',),(PageSpan(1,1),))
                    blocks.append(block)
                    chunks.append(Chunk(stable_chunk_id(version.paper_id,version.version,'c',ordinal),version.paper_id,version.version,ordinal,
                        block.text,content_hash(block.text),3,kind,block.section_path,block.page_spans,(block.block_id,)))
                catalog.save_blocks(blocks)
                return job,version,chunks
            def publish(job,version,chunks):
                owned=catalog.claim_job('lifecycle',job_id=job.job_id)
                for stage in (IngestionStage.CHUNKING,IngestionStage.EMBEDDING,IngestionStage.INDEXING,IngestionStage.PUBLISHING):
                    owned=catalog.advance_job(owned,stage)
                count=index.verify_version(version,chunks)
                catalog.complete_job(owned,indexed_chunk_count=count,expected_chunk_count=len(chunks),index_searchable=True)
            job,v,chunks=prepare('p1')
            vectors={c.chunk_id:(1.,0.) for c in chunks}
            for _ in range(2):index.upsert_chunks(v,chunks,vectors)
            assert index.count_version(v.paper_id,v.version)==2
            assert not index.vector_search(SearchQuery('q'),(1.,0.),limit=5)
            checks+=['idempotent_retry','staging_hidden']
            publish(job,v,chunks)
            assert len(index.vector_search(SearchQuery('q'),(1.,0.),limit=5))==1
            annotations=index.vector_search(SearchQuery('q',source_kinds=(SourceKind.USER_ANNOTATION,)),(1.,0.),limit=5)
            assert len(annotations)==1 and annotations[0].chunk.source_kind is SourceKind.USER_ANNOTATION
            assert not index.vector_search(SearchQuery('q',paper_ids=('absent',)),(1.,0.),limit=5)
            checks+=['annotation_prefilter','paper_prefilter']
            job2,v2,chunks2=prepare('p2')
            # Send an intentional malformed vector only to this disposable fixture index.
            docs=[]
            for ordinal,c in enumerate(chunks2):
                document=dict(paper_id=c.paper_id,version=c.version,chunk_id=c.chunk_id,source_kind=c.source_kind.value,
                    content=c.text,chunk=c.to_dict(),embedding_profile=profile.fingerprint,
                    embedding=[1.,0.] if ordinal==0 else [1.,0.,0.])
                raw=(json.dumps({'index':{'_id':c.chunk_id}})+'\n'+json.dumps(document)+'\n').encode()
                docs.append((c.chunk_id,raw))
            try:
                index._bulk(docs)
                raise AssertionError('Expected partial bulk error')
            except BulkWriteError as error:
                assert len(error.failures)==1
            assert index.count_version(v2.paper_id,v2.version)==1
            hits=index.vector_search(SearchQuery('q'),(1.,0.),limit=5)
            assert hits and all(h.chunk.version==v.version for h in hits)
            checks+=['actual_http200_partial_bulk_error','partial_staging_does_not_replace_active']
            index.upsert_chunks(v2,chunks2,{c.chunk_id:(1.,0.) for c in chunks2})
            publish(job2,v2,chunks2)
            hits=index.vector_search(SearchQuery('q'),(1.,0.),limit=5)
            assert hits and all(h.chunk.version==v2.version for h in hits)
            checks+=['repair_retry_and_version_switch']
            changed=ElasticsearchVectorIndex(service.client,catalog,replace(profile,revision='c'*40),name)
            try:
                changed.ensure_index()
                raise AssertionError('Expected model identity mismatch')
            except ValueError:
                pass
            checks+=['existing_index_model_mismatch_rejected']
            catalog.soft_delete_paper(v.paper_id)
            assert not index.vector_search(SearchQuery('q'),(1.,0.),limit=5)
            checks+=['deleted_paper_hidden']
        atomic_write(ROOT/'docs/vector-lifecycle-report.json',json.dumps(dict(checks=checks,passed=True),indent=2))
        print('PASS '+', '.join(checks))
    finally:
        if created:service.client.delete('/'+name).raise_for_status()
        service.close()


if __name__=='__main__':main()
