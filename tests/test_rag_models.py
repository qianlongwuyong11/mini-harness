from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from mini_harness.rag.interfaces import PaperIndex, PaperRetrievalService
from mini_harness.rag.models import (
    BlockKind,
    BoundingBox,
    Chunk,
    CoordinateSpace,
    Evidence,
    IngestionJob,
    IngestionStage,
    IngestionStatus,
    PageSpan,
    Paper,
    PaperVersion,
    RetrievalChannel,
    RetrievalHit,
    RetrievalMode,
    SearchQuery,
    SearchResult,
    SourceBlock,
    SourceKind,
    VersionState,
    content_hash,
    stable_chunk_id,
)


NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)


def make_block() -> SourceBlock:
    return SourceBlock(
        block_id="block-1",
        paper_id="paper-1",
        version="version-1",
        text="SparseBEV uses adaptive sampling for efficient multi-camera perception.",
        kind=BlockKind.TEXT,
        source_kind=SourceKind.PAPER,
        section_path=("3 Method", "3.2 Adaptive Sampling"),
        page_spans=(PageSpan(3, 3),),
        bboxes=(
            BoundingBox(
                page=3,
                x0=0.1,
                y0=0.2,
                x1=0.9,
                y1=0.4,
                coordinate_space=CoordinateSpace.NORMALIZED,
            ),
        ),
    )


def make_chunk(block: SourceBlock | None = None) -> Chunk:
    block = block or make_block()
    return Chunk(
        chunk_id=stable_chunk_id(
            block.paper_id,
            block.version,
            "paragraph-v1",
            0,
        ),
        paper_id=block.paper_id,
        version=block.version,
        ordinal=0,
        text=block.text,
        content_hash=content_hash(block.text),
        token_count=10,
        source_kind=block.source_kind,
        section_path=block.section_path,
        page_spans=block.page_spans,
        source_block_ids=(block.block_id,),
        bboxes=block.bboxes,
    )


class MemoryPaperIndex:
    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}
        self.vectors: dict[str, tuple[float, ...]] = {}

    def upsert_chunks(
        self,
        version: PaperVersion,
        chunks: tuple[Chunk, ...],
        embeddings: dict[str, tuple[float, ...]],
    ) -> None:
        del version
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            if chunk.chunk_id in embeddings:
                self.vectors[chunk.chunk_id] = embeddings[chunk.chunk_id]

    def lexical_search(
        self,
        query: SearchQuery,
        *,
        limit: int,
    ) -> tuple[RetrievalHit, ...]:
        terms = query.text.casefold().split()
        ranked: list[tuple[int, Chunk]] = []
        for chunk in self.chunks.values():
            if query.paper_ids and chunk.paper_id not in query.paper_ids:
                continue
            if chunk.source_kind not in query.source_kinds:
                continue
            score = sum(chunk.text.casefold().count(term) for term in terms)
            if score:
                ranked.append((score, chunk))
        ranked.sort(key=lambda item: (-item[0], item[1].ordinal))
        return tuple(
            RetrievalHit(
                chunk=chunk,
                channel=RetrievalChannel.BM25,
                rank=rank,
                score=float(score),
            )
            for rank, (score, chunk) in enumerate(ranked[:limit], start=1)
        )

    def vector_search(
        self,
        query: SearchQuery,
        query_vector: tuple[float, ...],
        *,
        limit: int,
    ) -> tuple[RetrievalHit, ...]:
        ranked: list[tuple[float, Chunk]] = []
        for chunk_id, stored in self.vectors.items():
            chunk = self.chunks[chunk_id]
            if query.paper_ids and chunk.paper_id not in query.paper_ids:
                continue
            if chunk.source_kind not in query.source_kinds:
                continue
            score = sum(a * b for a, b in zip(stored, query_vector))
            ranked.append((score, chunk))
        ranked.sort(key=lambda item: (-item[0], item[1].ordinal))
        return tuple(
            RetrievalHit(
                chunk=chunk,
                channel=RetrievalChannel.VECTOR,
                rank=rank,
                score=score,
            )
            for rank, (score, chunk) in enumerate(ranked[:limit], start=1)
        )

    def delete_version(self, paper_id: str, version: str) -> None:
        removed = [
            chunk_id
            for chunk_id, chunk in self.chunks.items()
            if chunk.paper_id == paper_id and chunk.version == version
        ]
        for chunk_id in removed:
            self.chunks.pop(chunk_id, None)
            self.vectors.pop(chunk_id, None)


class MemoryRetrievalService:
    def __init__(self, index: MemoryPaperIndex, blocks: tuple[SourceBlock, ...]) -> None:
        self.index = index
        self.blocks = blocks
        self.evidence: dict[str, Evidence] = {}

    def search(self, query: SearchQuery) -> SearchResult:
        hits = self.index.lexical_search(query, limit=query.top_k)
        evidence_items: list[Evidence] = []
        for ordinal, hit in enumerate(hits, start=1):
            chunk = hit.chunk
            item = Evidence(
                evidence_id=f"evidence-{ordinal}",
                paper_id=chunk.paper_id,
                version=chunk.version,
                chunk_id=chunk.chunk_id,
                title="SparseBEV",
                excerpt=chunk.text,
                source_kind=chunk.source_kind,
                section_path=chunk.section_path,
                page_spans=chunk.page_spans,
                bboxes=chunk.bboxes,
                retrieval_score=hit.score,
                bm25_rank=hit.rank,
            )
            self.evidence[item.evidence_id] = item
            evidence_items.append(item)
        return SearchResult(
            query=query,
            evidence=tuple(evidence_items),
            mode=RetrievalMode.LEXICAL,
            candidate_count=len(hits),
        )

    def read_section(
        self,
        paper_id: str,
        version: str,
        section_path: tuple[str, ...],
    ) -> tuple[SourceBlock, ...]:
        return tuple(
            block
            for block in self.blocks
            if block.paper_id == paper_id
            and block.version == version
            and block.section_path[: len(section_path)] == section_path
        )

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        return self.evidence.get(evidence_id)


class RagModelTests(unittest.TestCase):
    def test_stable_chunk_id_tracks_version_chunker_and_ordinal(self) -> None:
        first = stable_chunk_id("paper-1", "v1", "chunker-v1", 2)
        self.assertEqual(first, stable_chunk_id("paper-1", "v1", "chunker-v1", 2))
        self.assertNotEqual(first, stable_chunk_id("paper-1", "v2", "chunker-v1", 2))
        self.assertNotEqual(first, stable_chunk_id("paper-1", "v1", "chunker-v2", 2))
        self.assertNotEqual(first, stable_chunk_id("paper-1", "v1", "chunker-v1", 3))

    def test_source_location_rejects_invalid_page_and_box_data(self) -> None:
        with self.assertRaises(ValueError):
            PageSpan(0, 1)
        with self.assertRaises(ValueError):
            PageSpan(4, 3)
        with self.assertRaises(ValueError):
            BoundingBox(
                1,
                -0.1,
                0.0,
                0.5,
                0.5,
                CoordinateSpace.NORMALIZED,
            )
        with self.assertRaises(ValueError):
            BoundingBox(1, 0.5, 0.5, 0.4, 0.8)
        with self.assertRaises(ValueError):
            SourceBlock(
                block_id="b1",
                paper_id="p1",
                version="v1",
                text="body",
                kind=BlockKind.TEXT,
                source_kind=SourceKind.PAPER,
                section_path=(),
                page_spans=(),
            )
        with self.assertRaises(ValueError):
            SourceBlock(
                block_id="b1",
                paper_id="p1",
                version="v1",
                text="body",
                kind=BlockKind.TEXT,
                source_kind=SourceKind.PAPER,
                section_path=(),
                page_spans=(PageSpan(1, 1),),
                bboxes=(BoundingBox(2, 0.0, 0.0, 1.0, 1.0),),
            )

    def test_generated_summary_may_have_no_physical_page(self) -> None:
        block = SourceBlock(
            block_id="summary-1",
            paper_id="paper-1",
            version="version-1",
            text="A generated overview.",
            kind=BlockKind.TEXT,
            source_kind=SourceKind.GENERATED_SUMMARY,
            section_path=(),
            page_spans=(),
        )
        self.assertEqual(block.page_spans, ())

    def test_models_serialize_to_plain_json_values(self) -> None:
        paper = Paper(
            paper_id="paper-1",
            source_uri="file:///papers/SparseBEV.pdf",
            source_hash=content_hash("pdf bytes"),
            title="SparseBEV",
            created_at=NOW,
        )
        query = SearchQuery(text="adaptive sampling")
        evidence = Evidence(
            evidence_id="e1",
            paper_id=paper.paper_id,
            version="version-1",
            chunk_id=make_chunk().chunk_id,
            title=paper.title or paper.paper_id,
            excerpt="adaptive sampling",
            source_kind=SourceKind.PAPER,
            section_path=(),
            page_spans=(PageSpan(3, 3),),
        )
        result = SearchResult(query, (evidence,), RetrievalMode.HYBRID, 1)
        payload = json.loads(result.to_json())
        self.assertEqual(payload["mode"], "hybrid")
        self.assertEqual(payload["query"]["source_kinds"], ["paper"])
        self.assertEqual(payload["evidence"][0]["page_spans"][0]["start"], 3)
        self.assertEqual(json.loads(paper.to_json())["created_at"], NOW.isoformat())

    def test_query_result_and_job_boundaries(self) -> None:
        with self.assertRaises(ValueError):
            SearchQuery(text=" ")
        with self.assertRaises(ValueError):
            SearchQuery(text="query", top_k=0)
        query = SearchQuery(text="query", top_k=1)
        evidence = Evidence(
            evidence_id="e1",
            paper_id="paper-1",
            version="version-1",
            chunk_id=make_chunk().chunk_id,
            title="Paper",
            excerpt="body",
            source_kind=SourceKind.PAPER,
            section_path=(),
            page_spans=(PageSpan(1, 1),),
        )
        with self.assertRaises(ValueError):
            SearchResult(query, (evidence, evidence), RetrievalMode.LEXICAL, 2)
        with self.assertRaises(ValueError):
            IngestionJob(
                job_id="job-1",
                paper_id="paper-1",
                target_version="version-1",
                source_uri="file:///paper.pdf",
                source_hash=content_hash("pdf"),
                status=IngestionStatus.READY,
                stage=IngestionStage.INDEXING,
                created_at=NOW,
                updated_at=NOW,
            )
        with self.assertRaises(ValueError):
            IngestionJob(
                job_id="job-1",
                paper_id="paper-1",
                target_version="version-1",
                source_uri="file:///paper.pdf",
                source_hash=content_hash("pdf"),
                status=IngestionStatus.RUNNING,
                stage=IngestionStage.PARSING,
                progress_current=2,
                progress_total=1,
                created_at=NOW,
                updated_at=NOW,
            )

    def test_in_memory_block_to_evidence_contract(self) -> None:
        block = make_block()
        chunk = make_chunk(block)
        version = PaperVersion(
            paper_id=block.paper_id,
            version=block.version,
            source_hash=content_hash("pdf bytes"),
            parser_version="fake-parser-v1",
            chunker_version="paragraph-v1",
            embedding_model="fake-embedding",
            embedding_revision="v1",
            state=VersionState.STAGING,
            chunk_count=1,
            created_at=NOW,
        )
        index = MemoryPaperIndex()
        index.upsert_chunks(version, (chunk,), {chunk.chunk_id: (1.0, 0.0)})
        service = MemoryRetrievalService(index, (block,))

        self.assertIsInstance(index, PaperIndex)
        self.assertIsInstance(service, PaperRetrievalService)
        result = service.search(SearchQuery(text="adaptive sampling", top_k=3))

        self.assertEqual(len(result.evidence), 1)
        item = result.evidence[0]
        self.assertEqual(item.chunk_id, chunk.chunk_id)
        self.assertEqual(item.page_spans, (PageSpan(3, 3),))
        self.assertEqual(item.bboxes, block.bboxes)
        self.assertEqual(service.get_evidence(item.evidence_id), item)
        self.assertEqual(
            service.read_section(block.paper_id, block.version, ("3 Method",)),
            (block,),
        )


if __name__ == "__main__":
    unittest.main()
