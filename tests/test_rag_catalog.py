from __future__ import annotations

import multiprocessing
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from mini_harness.rag.catalog import CatalogConflict, SQLitePaperCatalog, _now
from mini_harness.rag.interfaces import PaperCatalog
from mini_harness.rag.models import (
    BlockKind, BoundingBox, IngestionStage as Stage, IngestionStatus as Status,
    PageSpan, SourceBlock, SourceKind, VersionState, content_hash,
)


def claim_in_process(path, start, output):
    try:
        catalog = SQLitePaperCatalog(path)
        start.wait(10)
        job = catalog.claim_job(str(multiprocessing.current_process().pid))
        output.put(job.job_id if job else None)
    except BaseException as exc:
        output.put("ERROR:" + repr(exc))


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "catalog.sqlite3"
        self.catalog = SQLitePaperCatalog(self.path)

    def enqueue(self, chunker="c1"):
        return self.catalog.enqueue_import(
            "file:///papers/sample.pdf", content_hash("pdf bytes"),
            parser_version="p1", chunker_version=chunker, title="Sample",
        )

    def block(self, job, block_id="b1"):
        return SourceBlock(
            block_id, job.paper_id, job.target_version, "unique source body",
            BlockKind.TEXT, SourceKind.PAPER, ("Method",), (PageSpan(3, 4),),
            (BoundingBox(3, 10, 20, 100, 200),),
        )

    def publishing(self, chunker="c1"):
        job = self.enqueue(chunker)
        self.catalog.save_blocks((self.block(job),))
        job = self.catalog.claim_job("worker")
        for stage in (Stage.CHUNKING, Stage.EMBEDDING, Stage.INDEXING, Stage.PUBLISHING):
            job = self.catalog.advance_job(job, stage)
        return job

    def complete(self, job):
        return self.catalog.complete_job(job, indexed_chunk_count=2,
                                        expected_chunk_count=2, index_searchable=True)

    def test_schema_reopen_and_protocol(self):
        self.assertIsInstance(self.catalog, PaperCatalog)
        self.enqueue()
        reopened = SQLitePaperCatalog(self.path)
        self.assertEqual(len(reopened.list_unfinished_jobs()), 1)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 1)
            db.execute("PRAGMA user_version=99")
        with self.assertRaises(CatalogConflict):
            SQLitePaperCatalog(self.path)

    def test_duplicate_import_reuses_job_and_config_change_creates_version(self):
        first = self.enqueue()
        second = self.catalog.enqueue_import(
            "file:///other/same.pdf", content_hash("pdf bytes"),
            parser_version="p1", chunker_version="c1",
        )
        self.assertEqual(first, second)
        changed = self.enqueue("c2")
        self.assertEqual(changed.paper_id, first.paper_id)
        self.assertNotEqual(changed.target_version, first.target_version)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM papers").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT count(*) FROM jobs").fetchone()[0], 2)

    def test_blocks_roundtrip_and_retry_without_database_body(self):
        job = self.enqueue()
        blocks = (self.block(job, "z"), self.block(job, "a"))
        self.catalog.save_blocks(blocks)
        self.catalog.save_blocks(blocks)
        self.assertEqual(SQLitePaperCatalog(self.path).get_blocks(job.paper_id, job.target_version), blocks)
        with closing(sqlite3.connect(self.path)) as db:
            dump = "\n".join(db.iterdump())
            self.assertNotIn("unique source body", dump)
            self.assertEqual(db.execute("SELECT count(*) FROM blocks").fetchone()[0], 2)
        with self.assertRaises(CatalogConflict):
            self.catalog.save_blocks((replace(blocks[0], text="changed"),))

    def test_block_write_failure_does_not_publish_references(self):
        job = self.enqueue()
        with patch("mini_harness.rag.catalog.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                self.catalog.save_blocks((self.block(job),))
        self.assertEqual(self.catalog.get_blocks(job.paper_id, job.target_version), ())
        self.catalog.save_blocks((self.block(job),))
        payload = next(self.catalog.block_dir.glob("*.json"))
        payload.write_text("corrupted", encoding="utf-8")
        with self.assertRaises(CatalogConflict):
            self.catalog.get_blocks(job.paper_id, job.target_version)

    def test_invalid_stages_stale_snapshot_and_progress(self):
        self.enqueue()
        job = self.catalog.claim_job("worker")
        with self.assertRaises(CatalogConflict):
            self.catalog.advance_job(job, Stage.INDEXING)
        current = self.catalog.update_progress(job, 1, 3)
        with self.assertRaises(CatalogConflict):
            self.catalog.advance_job(job, Stage.CHUNKING)
        with self.assertRaises(CatalogConflict):
            self.catalog.update_progress(current, 0, 3)
        current = self.catalog.advance_job(current, Stage.CHUNKING)
        self.assertEqual(current.progress_current, 0)
        with self.assertRaises(CatalogConflict):
            self.catalog.save_job(replace(current, stage=Stage.INDEXING))

    def test_failure_restart_retry_preserves_stage_and_attempts(self):
        self.enqueue()
        job = self.catalog.claim_job("worker")
        job = self.catalog.advance_job(job, Stage.CHUNKING)
        failed = self.catalog.fail_job(job, "parser output rejected")
        reopened = SQLitePaperCatalog(self.path)
        self.assertEqual(reopened.get_job(job.job_id), failed)
        self.assertEqual(failed.stage, Stage.CHUNKING)
        reopened.retry_job(job.job_id)
        reclaimed = reopened.claim_job("worker")
        self.assertEqual(reclaimed.attempts, 2)
        self.assertEqual(reclaimed.stage, Stage.PARSING)
        with self.assertRaises(CatalogConflict):
            self.catalog.fail_job(job, "late result")

    def test_recover_only_expired_lease_and_fence_previous_attempt(self):
        self.enqueue()
        job = self.catalog.claim_job("same-worker", lease_seconds=60)
        reopened = SQLitePaperCatalog(self.path)
        self.assertEqual(reopened.recover_expired_jobs(), 0)
        renewed = reopened.heartbeat(job, 120)
        future = renewed.lease_expires_at + timedelta(seconds=1)
        with patch("mini_harness.rag.catalog._now", return_value=future):
            self.assertEqual(reopened.recover_expired_jobs(), 1)
            replacement = reopened.claim_job("same-worker")
            with self.assertRaises(CatalogConflict):
                reopened.advance_job(renewed, Stage.CHUNKING)
        self.assertEqual(replacement.attempts, 2)

    def test_publication_verification_atomicity_and_old_version_retained(self):
        first = self.publishing()
        self.complete(first)
        old = self.catalog.get_active_version(first.paper_id)
        second = self.publishing("c2")
        with self.assertRaises(CatalogConflict):
            self.catalog.complete_job(second, indexed_chunk_count=1,
                                      expected_chunk_count=2, index_searchable=True)
        with self.assertRaises(CatalogConflict):
            self.catalog.complete_job(second, indexed_chunk_count=2,
                                      expected_chunk_count=2, index_searchable=False)
        original = self.catalog._put_job
        def fail_finish(db, job):
            if job.status is Status.READY:
                raise OSError("crash before transaction commit")
            original(db, job)
        with patch.object(self.catalog, "_put_job", side_effect=fail_finish):
            with self.assertRaises(OSError):
                self.complete(second)
        self.assertEqual(self.catalog.get_active_version(first.paper_id), old)
        self.assertEqual(self.catalog.get_job(second.job_id), second)
        self.assertEqual(self.catalog.get_version(second.paper_id, second.target_version).state, VersionState.STAGING)
        self.complete(second)
        self.assertEqual(self.catalog.get_active_version(first.paper_id).version, second.target_version)
        self.assertEqual(self.catalog.get_version(old.paper_id, old.version), old)
        self.assertEqual(len(self.catalog.get_blocks(old.paper_id, old.version)), 1)
        self.catalog.activate_version(old.paper_id, old.version)
        self.assertEqual(self.catalog.get_active_version(old.paper_id), old)

    def test_empty_blocks_cannot_publish_or_activate_staging(self):
        queued = self.enqueue()
        with self.assertRaises(CatalogConflict):
            self.catalog.activate_version(queued.paper_id, queued.target_version)
        job = self.catalog.claim_job("w")
        for stage in (Stage.CHUNKING, Stage.EMBEDDING, Stage.INDEXING, Stage.PUBLISHING):
            job = self.catalog.advance_job(job, stage)
        with self.assertRaises(CatalogConflict):
            self.complete(job)
        self.assertIsNone(self.catalog.get_active_version(job.paper_id))

    def test_soft_delete_hides_active_version_and_fences_worker(self):
        first = self.publishing()
        self.complete(first)
        queued = self.enqueue("c2")
        running = self.catalog.claim_job("w")
        self.catalog.soft_delete_paper(first.paper_id)
        self.assertIsNone(self.catalog.get_paper(first.paper_id))
        self.assertIsNone(self.catalog.get_active_version(first.paper_id))
        self.assertEqual(self.catalog.get_job(queued.job_id).status, Status.CANCELLED)
        self.assertIsNotNone(self.catalog.get_version(first.paper_id, first.target_version))
        with self.assertRaises(CatalogConflict):
            self.catalog.advance_job(running, Stage.CHUNKING)
        with self.assertRaises(CatalogConflict):
            self.enqueue()
        self.assertEqual(self.catalog.list_unfinished_jobs(), ())

    def test_two_processes_cannot_claim_same_job(self):
        queued = self.enqueue()
        ctx = multiprocessing.get_context("spawn")
        start, output = ctx.Event(), ctx.Queue()
        processes = [ctx.Process(target=claim_in_process, args=(str(self.path), start, output)) for _ in range(2)]
        try:
            for proc in processes:
                proc.start()
            start.set()
            answers = [output.get(timeout=30) for _ in processes]
            for proc in processes:
                proc.join(30)
                self.assertEqual(proc.exitcode, 0)
            self.assertCountEqual(answers, [queued.job_id, None])
        finally:
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
                proc.join()
            output.close()
            output.join_thread()


if __name__ == "__main__":
    unittest.main()
