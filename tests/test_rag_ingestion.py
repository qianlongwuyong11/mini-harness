from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mini_harness.rag.catalog import SQLitePaperCatalog
from mini_harness.rag.ingestion import IngestionWorker, worker_lock
from mini_harness.rag.models import BlockKind, SourceBlock, SourceKind, PageSpan, IngestionStatus, IngestionStage
from mini_harness.rag.vector_index import EmbeddingProfile
from test_rag_elasticsearch import chunk


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.catalog = SQLitePaperCatalog(self.root/'catalog.sqlite3')
        self.source = self.root/'input.pdf'
        self.source.write_bytes(b'%PDF-test')
        self.parser = Mock(parser_version='parser')
        self.parser.parse.side_effect = lambda p,v: (SourceBlock('block-'+v.version,p.paper_id,v.version,'text',BlockKind.TEXT,SourceKind.PAPER,('S',),(PageSpan(1,1),)),)
        self.chunker = Mock(chunker_version='chunker')
        self.chunker.split.side_effect = lambda v,b: (chunk(v),)
        self.profile = EmbeddingProfile('model','a'*40,2,'b'*64)
        self.embedding = Mock(config=SimpleNamespace(batch_size=2))
        self.embedding.metadata.return_value = asdict(self.profile)
        self.embedding.embed_documents.return_value = ((1.,0.),)
        self.index = Mock(profile=self.profile)
        self.index.verify_version.return_value = 1
        self.worker = IngestionWorker(self.catalog,self.parser,self.chunker,self.embedding,self.index,self.root/'files')

    def test_durable_source_and_idempotent_restart(self):
        job = self.worker.enqueue(self.source)
        self.assertEqual(self.worker.enqueue(self.source).job_id,job.job_id)
        self.source.unlink()
        ready = self.worker.run_once()
        self.assertEqual(ready.status,IngestionStatus.READY)
        reopened = SQLitePaperCatalog(self.catalog.path)
        self.assertEqual(len(reopened.get_blocks(job.paper_id,job.target_version)),1)
        self.assertEqual(reopened.get_active_version(job.paper_id).chunk_count,1)
        self.assertIsNone(self.worker.run_once())
        self.assertTrue(list((self.root/'files/artifacts').glob('*/chunks.json')))

    def test_failure_stages_retry_and_old_version_survives(self):
        old = self.worker.enqueue(self.source)
        self.worker.run_once()
        self.parser.parser_version = 'parser2'
        job = self.worker.enqueue(self.source)
        self.index.upsert_chunks.side_effect = RuntimeError('partial bulk failure')
        with self.assertRaisesRegex(RuntimeError,'partial bulk'):
            self.worker.run_once()
        failed = self.catalog.get_job(job.job_id)
        self.assertEqual(failed.stage,IngestionStage.INDEXING)
        self.assertEqual(failed.status,IngestionStatus.FAILED)
        self.assertEqual(self.catalog.get_active_version(job.paper_id).version,old.target_version)
        self.catalog.retry_job(job.job_id)
        parse_calls = self.parser.parse.call_count
        embedding_calls = self.embedding.embed_documents.call_count
        self.index.upsert_chunks.side_effect = None
        self.worker.run_once()
        self.assertEqual(self.parser.parse.call_count,parse_calls)
        self.assertEqual(self.embedding.embed_documents.call_count,embedding_calls)
        self.assertEqual(self.catalog.get_active_version(job.paper_id).version,job.target_version)

    def test_parser_and_embedding_failure_keep_stage(self):
        job = self.worker.enqueue(self.source)
        self.parser.parse.side_effect = RuntimeError('parse failure')
        with self.assertRaises(RuntimeError): self.worker.run_once()
        self.assertEqual(self.catalog.get_job(job.job_id).stage,IngestionStage.PARSING)
        self.catalog.retry_job(job.job_id)
        self.parser.parse.side_effect = lambda p,v: (SourceBlock('b',p.paper_id,v.version,'text',BlockKind.TEXT,SourceKind.PAPER,(),(PageSpan(1,1),)),)
        self.embedding.embed_documents.side_effect = RuntimeError('encode failure')
        with self.assertRaises(RuntimeError): self.worker.run_once()
        self.assertEqual(self.catalog.get_job(job.job_id).stage,IngestionStage.EMBEDDING)
        self.assertIsNone(self.catalog.get_active_version(job.paper_id))

    def test_cancel_during_index_does_not_publish(self):
        job = self.worker.enqueue(self.source)
        self.index.upsert_chunks.side_effect = lambda *a: self.catalog.cancel_job(job.job_id)
        with self.assertRaises(Exception): self.worker.run_once()
        self.assertEqual(self.catalog.get_job(job.job_id).status,IngestionStatus.CANCELLED)
        self.assertIsNone(self.catalog.get_active_version(job.paper_id))

    def test_crash_before_publish_recovers_after_lease_expiry(self):
        job = self.worker.enqueue(self.source)
        with patch.object(self.catalog,'complete_job',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt): self.worker.run_once()
        running = self.catalog.get_job(job.job_id)
        self.assertEqual(running.stage,IngestionStage.PUBLISHING)
        self.assertIsNone(self.catalog.get_active_version(job.paper_id))
        with self.catalog._connection(write=True) as db:
            self.catalog._put_job(db,replace(running,lease_expires_at=datetime.now(timezone.utc)-timedelta(seconds=1)))
        self.assertEqual(self.worker.run_once().status,IngestionStatus.READY)

    def test_second_worker_lock_is_rejected(self):
        with worker_lock(str(self.catalog.path)+'.worker.lock'):
            with self.assertRaises(OSError): self.worker.run_once()

    def test_verify_failure_does_not_publish(self):
        job = self.worker.enqueue(self.source)
        self.index.verify_version.side_effect = RuntimeError('refresh failed')
        with self.assertRaisesRegex(RuntimeError,'refresh failed'):
            self.worker.run_once()
        self.assertEqual(self.catalog.get_job(job.job_id).status,IngestionStatus.FAILED)
        self.assertIsNone(self.catalog.get_active_version(job.paper_id))

    def test_long_operation_renews_lease(self):
        from mini_harness.rag.ingestion import _Lease
        import threading
        job = self.worker.enqueue(self.source)
        owned = self.catalog.claim_job('test',lease_seconds=3)
        renewed = threading.Event()
        original = self.catalog.heartbeat
        def observe(*args,**kwargs):
            result = original(*args,**kwargs)
            renewed.set()
            return result
        with patch.object(self.catalog,'heartbeat',side_effect=observe):
            with _Lease(self.catalog,owned,3) as lease:
                self.assertTrue(renewed.wait(4))
                self.assertGreater(lease.job.lease_expires_at,owned.lease_expires_at)
                lease.call('advance_job',IngestionStage.CHUNKING)
