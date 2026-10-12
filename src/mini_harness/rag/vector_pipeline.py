"""Complete vector ingestion from durable parsed/chunked artifacts."""
from .models import IngestionStage
from .vector_index import EmbeddingProfile


def encode_index_publish(catalog,index,embedding,job,chunks):
    """Input job is leased at EMBEDDING; all failures leave the version unpublished."""
    if job.stage is not IngestionStage.EMBEDDING:
        raise ValueError("Job must be leased at EMBEDDING")
    owned=job
    try:
        if EmbeddingProfile.from_metadata(embedding.metadata()) != index.profile:
            raise ValueError("Embedding service differs from target index")
        if not chunks or len({c.chunk_id for c in chunks})!=len(chunks):
            raise ValueError("Nonempty unique chunk list required")
        version=catalog.get_version(job.paper_id,job.target_version)
        vectors={}
        batch_size=embedding.config.batch_size
        for start in range(0,len(chunks),batch_size):
            owned=catalog.heartbeat(owned)
            batch=chunks[start:start+batch_size]
            rows=embedding.embed_documents([c.text for c in batch])
            if len(rows)!=len(batch):
                raise ValueError("Embedding result count differs from batch")
            vectors.update((c.chunk_id,v) for c,v in zip(batch,rows))
        owned=catalog.advance_job(owned,IngestionStage.INDEXING)
        # Small local papers fit in memory; ES bulk remains byte- and count-bounded.
        index.upsert_chunks(version,chunks,vectors)
        owned=catalog.heartbeat(owned)
        count=index.verify_version(version,chunks)
        owned=catalog.advance_job(owned,IngestionStage.PUBLISHING)
        return catalog.complete_job(owned,indexed_chunk_count=count,
            expected_chunk_count=len(chunks),index_searchable=True)
    except Exception as exc:
        try:
            catalog.fail_job(owned,str(exc))
        except Exception:
            # A stale worker cannot alter the newer owner's state; preserve original failure.
            pass
        raise
