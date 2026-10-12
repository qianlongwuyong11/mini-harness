"""Step 8 CLI: import saved paper artifacts, then query the durable ES vector index."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from local_es import CONFIG
from mini_harness.rag.catalog import SQLitePaperCatalog, _decode
from mini_harness.rag.elasticsearch import chunk_from_dict
from mini_harness.rag.embedding import LocalE5Embedding,load_default_config
from mini_harness.rag.es_service import ESService,ESConfig
from mini_harness.rag.models import (Paper,PaperVersion,SourceBlock,IngestionJob,IngestionStage,
    IngestionStatus,SearchQuery,SourceKind,VersionState)
from mini_harness.rag.vector_index import EmbeddingProfile,ElasticsearchVectorIndex,VectorSearchService
from mini_harness.rag.vector_pipeline import encode_index_publish

ARTIFACT=ROOT/'.mini-harness/rag/selected/557c3224b3e11dd5d1372e20'
CATALOG=ROOT/'.mini-harness/rag/vector-catalog.sqlite3'


def load_artifacts(path):
    manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    version=_decode(PaperVersion,json.dumps(manifest['version']))
    blocks=tuple(_decode(SourceBlock,json.dumps(b)) for b in json.loads((path/'source-blocks.json').read_text(encoding='utf-8')))
    chunks=tuple(chunk_from_dict(c) for c in json.loads((path/'chunks.json').read_text(encoding='utf-8')) if c['source_kind']=='paper')
    if set(manifest['chunk_ids'])!={c.chunk_id for c in chunks}:
        raise ValueError('Artifact manifest/chunk IDs differ')
    if any((b.paper_id,b.version)!=(version.paper_id,version.version) for b in blocks):
        raise ValueError('Source block version mismatch')
    return manifest,version,blocks,chunks


def import_artifacts(catalog,index,embedding,path):
    manifest,version,blocks,chunks=load_artifacts(path)
    if EmbeddingProfile.from_metadata(manifest['embedding'])!=index.profile:
        raise ValueError('Artifact embedding profile differs from index')
    current=catalog.get_version(version.paper_id,version.version)
    if current is not None and current.state is VersionState.READY:
        index.verify_version(current,chunks)
        return {'already_ready':True,'chunks':len(chunks)}
    if catalog.get_paper(version.paper_id) is None:
        candidates=[path/'source.pdf',*sorted((ROOT/'.mini-harness/rag/parse-eval').glob('*/source.pdf'))]
        source_pdf=next((p for p in candidates if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==version.source_hash),None)
        if source_pdf is None:
            raise ValueError('Original PDF with matching source hash is required')
        catalog.create_paper(Paper(version.paper_id,str(source_pdf.resolve()),version.source_hash,title='SparseBEV'))
    if current is None:
        catalog.save_version(version)
    job_id='vector-import-'+version.version
    job=catalog.get_job(job_id)
    if job is None:
        catalog.save_job(IngestionJob(job_id,version.paper_id,version.version,catalog.get_paper(version.paper_id).source_uri,version.source_hash))
    elif job.status is IngestionStatus.FAILED:
        catalog.retry_job(job_id)
    catalog.recover_expired_jobs()
    owned=catalog.claim_job('vector-cli',job_id=job_id)
    if owned is None:
        raise RuntimeError('Job is already owned or cancelled; wait for its lease/recovery')
    try:
        catalog.save_blocks(blocks)
        owned=catalog.advance_job(owned,IngestionStage.CHUNKING)
        owned=catalog.advance_job(owned,IngestionStage.EMBEDDING)
    except Exception as exc:
        catalog.fail_job(owned,str(exc))
        raise
    started=time.perf_counter()
    completed=encode_index_publish(catalog,index,embedding,owned,chunks)
    return dict(job_id=completed.job_id,chunks=len(chunks),encode_index_publish_seconds=time.perf_counter()-started)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['import','query'])
    parser.add_argument('query',nargs='?',default='自注意力如何调整感受野？')
    parser.add_argument('--artifacts',type=Path,default=ARTIFACT)
    parser.add_argument('--catalog',type=Path,default=CATALOG)
    parser.add_argument('--top-k',type=int,default=5)
    args=parser.parse_args()
    cfg=load_default_config()
    embedding=LocalE5Embedding(cfg,ROOT/'.mini-harness/rag/models'/cfg.model_id.split('/')[-1])
    catalog=SQLitePaperCatalog(args.catalog)
    service=ESService(ESConfig.load(CONFIG))
    try:
        service.health()
        index=ElasticsearchVectorIndex(service.client,catalog,EmbeddingProfile.from_metadata(embedding.metadata()),analyzer='english',batch_size=32)
        index.ensure_index()
        if args.action=='import':
            result=import_artifacts(catalog,index,embedding,args.artifacts)
        else:
            started=time.perf_counter()
            hits=VectorSearchService(index,embedding).search(SearchQuery(args.query,top_k=args.top_k))
            result=dict(query=args.query,query_encoding_and_es_ms=(time.perf_counter()-started)*1000,
                        hits=[h.to_dict() for h in hits])
        sys.stdout.buffer.write((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    finally:
        service.close()


if __name__=='__main__':
    main()
