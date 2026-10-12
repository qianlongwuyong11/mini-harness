"""Single-worker durable PDF ingestion. Files precede state transitions."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import logging
from pathlib import Path
import threading
import uuid

from .models import IngestionStage, IngestionStatus, SourceKind
from .parsers.docling import _atomic_copy, _atomic_text, _sha256_file
from .vector_index import EmbeddingProfile


@contextmanager
def worker_lock(path):
    """One worker per catalog, including separate CLI processes; OS releases on exit."""
    with Path(path).open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        import os
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class _Lease:
    def __init__(self, catalog, job, seconds):
        self.catalog, self.job, self.seconds = catalog, job, seconds
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.error = None

    def __enter__(self):
        def renew():
            while not self.stop.wait(self.seconds / 3):
                try:
                    self.call('heartbeat', lease_seconds=self.seconds)
                except Exception as exc:
                    self.error = exc
                    return
        self.thread = threading.Thread(target=renew, name='rag-lease', daemon=True)
        self.thread.start()
        return self

    def call(self, method, *args, **kwargs):
        with self.lock:
            if self.error:
                raise self.error
            self.job = getattr(self.catalog, method)(self.job, *args, **kwargs)
            return self.job

    def check(self):
        self.call('heartbeat', lease_seconds=self.seconds)

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


class IngestionWorker:
    def __init__(self, catalog, parser, chunker, embedding, index, root, *, lease_seconds=300):
        if lease_seconds < 3:
            raise ValueError('Lease must be at least 3 seconds')
        self.catalog, self.parser, self.chunker = catalog, parser, chunker
        self.embedding, self.index = embedding, index
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lease_seconds = lease_seconds
        self.worker_id = 'ingest-' + uuid.uuid4().hex
        if EmbeddingProfile.from_metadata(embedding.metadata()) != index.profile:
            raise ValueError('Encoder and target index differ')

    def enqueue(self, source, *, title=None):
        source = Path(source).resolve()
        if source.suffix.lower() != '.pdf' or not source.is_file():
            raise ValueError('Existing PDF required')
        digest = _sha256_file(source)
        target = self.root / 'sources' / (digest + '.pdf')
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_copy(source, target, digest)
        return self.catalog.enqueue_import(str(target), digest,
            parser_version=self.parser.parser_version, chunker_version=self.chunker.chunker_version,
            embedding_model=self.index.profile.model_id, embedding_revision=self.index.profile.revision,
            title=title or source.stem)

    def run_once(self, *, job_id=None):
        with worker_lock(str(self.catalog.path) + '.worker.lock'):
            self.catalog.recover_expired_jobs()
            job = self.catalog.claim_job(self.worker_id, self.lease_seconds, job_id=job_id)
            if job is None:
                return None
            with _Lease(self.catalog, job, self.lease_seconds) as lease:
                try:
                    return self._process(lease)
                except Exception as exc:
                    try:
                        lease.call('fail_job', str(exc))
                    except Exception:
                        pass  # Cancellation or lost ownership must not be overwritten.
                    raise

    def serve(self, stop: threading.Event, *, poll_seconds=1):
        """Call in a managed background thread; shutdown finishes the current job."""
        while not stop.is_set():
            try:
                result = self.run_once()
            except Exception:
                # The failed job remains durable for explicit retry; continue other jobs.
                logging.getLogger(__name__).exception('RAG worker iteration failed')
                result = None
            if result is None:
                stop.wait(poll_seconds)

    def _process(self, lease):
        job = lease.job
        version = self.catalog.get_version(job.paper_id, job.target_version)
        if (version.parser_version, version.chunker_version, version.embedding_model, version.embedding_revision) != (
            self.parser.parser_version, self.chunker.chunker_version,
            self.index.profile.model_id, self.index.profile.revision):
            raise ValueError('Queued job configuration differs from this worker')
        paper = self.catalog.get_paper(job.paper_id)
        if _sha256_file(Path(paper.source_uri)) != paper.source_hash:
            raise ValueError('Stored original PDF checksum mismatch')
        blocks = self.catalog.get_blocks(job.paper_id, job.target_version)
        if not blocks:
            blocks = self.parser.parse(paper, version)
            lease.check()
            self.catalog.save_blocks(blocks)
        lease.call('advance_job', IngestionStage.CHUNKING)
        chunks = tuple(c for c in self.chunker.split(version, blocks) if c.source_kind is SourceKind.PAPER)
        if not chunks:
            raise ValueError('No paper chunks to index')
        directory = self.root / 'artifacts' / hashlib.sha256(job.job_id.encode()).hexdigest()
        directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, sort_keys=True)
        _atomic_text(directory / 'chunks.json', payload)
        lease.call('advance_job', IngestionStage.EMBEDDING)
        cache_key = hashlib.sha256((payload + self.index.profile.fingerprint).encode()).hexdigest()
        cache = directory / (cache_key + '.vectors.json')
        vectors = {}
        if cache.exists():
            vectors = json.loads(cache.read_text(encoding='utf-8'))
        else:
            size = self.embedding.config.batch_size
            for start in range(0, len(chunks), size):
                lease.check()
                batch = chunks[start:start+size]
                rows = self.embedding.embed_documents([c.text for c in batch])
                if len(rows) != len(batch):
                    raise ValueError('Embedding result count differs from batch')
                vectors.update((c.chunk_id, list(v)) for c, v in zip(batch, rows))
                lease.call('update_progress', min(start+size,len(chunks)), len(chunks))
            for vector in vectors.values():
                self.index.profile.check_vector(vector)
            _atomic_text(cache, json.dumps(vectors, allow_nan=False))
        if set(vectors) != {c.chunk_id for c in chunks}:
            raise ValueError('Cached vectors differ from chunk IDs')
        for vector in vectors.values():
            self.index.profile.check_vector(vector)
        lease.call('advance_job', IngestionStage.INDEXING)
        self.index.ensure_index()
        self.index.upsert_chunks(version, chunks, vectors)
        lease.check()
        count = self.index.verify_version(version, chunks)
        lease.call('advance_job', IngestionStage.PUBLISHING)
        return lease.call('complete_job', indexed_chunk_count=count,
                          expected_chunk_count=len(chunks), index_searchable=True)
