"""论文 RAG 的框架无关服务协议。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from .models import (
    Chunk,
    Evidence,
    IngestionJob,
    IngestionStage,
    Paper,
    PaperVersion,
    RetrievalHit,
    SearchQuery,
    SearchResult,
    SourceBlock,
    Vector,
)


@runtime_checkable
class PaperParser(Protocol):
    def parse(self, paper: Paper, version: PaperVersion) -> tuple[SourceBlock, ...]: ...


@runtime_checkable
class Chunker(Protocol):
    def split(
        self,
        version: PaperVersion,
        blocks: Sequence[SourceBlock],
    ) -> tuple[Chunk, ...]: ...


@runtime_checkable
class EmbeddingService(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> tuple[Vector, ...]: ...

    def embed_query(self, text: str) -> Vector: ...


@runtime_checkable
class PaperIndex(Protocol):
    def upsert_chunks(
        self,
        version: PaperVersion,
        chunks: Sequence[Chunk],
        embeddings: Mapping[str, Vector],
    ) -> None: ...

    def lexical_search(
        self,
        query: SearchQuery,
        *,
        limit: int,
    ) -> tuple[RetrievalHit, ...]: ...

    def vector_search(
        self,
        query: SearchQuery,
        query_vector: Vector,
        *,
        limit: int,
    ) -> tuple[RetrievalHit, ...]: ...

    def delete_version(self, paper_id: str, version: str) -> None: ...


@runtime_checkable
class PaperCatalog(Protocol):
    def create_paper(self, paper: Paper) -> None: ...

    def get_paper(self, paper_id: str) -> Paper | None: ...

    def save_version(self, version: PaperVersion) -> None: ...

    def get_version(self, paper_id: str, version: str) -> PaperVersion | None: ...

    def get_active_version(self, paper_id: str) -> PaperVersion | None: ...

    def list_active_versions(self) -> tuple[PaperVersion, ...]: ...

    def activate_version(self, paper_id: str, version: str) -> None: ...

    def save_blocks(self, blocks: Sequence[SourceBlock]) -> None: ...

    def get_blocks(self, paper_id: str, version: str) -> tuple[SourceBlock, ...]: ...

    def save_job(self, job: IngestionJob) -> None: ...

    def get_job(self, job_id: str) -> IngestionJob | None: ...

    def enqueue_import(
        self, source_uri: str, source_hash: str, *, parser_version: str,
        chunker_version: str, embedding_model: str | None = None,
        embedding_revision: str | None = None, title: str | None = None,
    ) -> IngestionJob: ...

    def list_unfinished_jobs(self) -> tuple[IngestionJob, ...]: ...

    def claim_job(self, worker_id: str, lease_seconds: int = 300, *, job_id: str | None = None) -> IngestionJob | None: ...

    def heartbeat(self, expected: IngestionJob, lease_seconds: int = 300) -> IngestionJob: ...

    def advance_job(self, expected: IngestionJob, stage: IngestionStage) -> IngestionJob: ...

    def update_progress(
        self, expected: IngestionJob, current: int, total: int | None,
    ) -> IngestionJob: ...

    def fail_job(self, expected: IngestionJob, error: str) -> IngestionJob: ...

    def cancel_job(self, job_id: str) -> IngestionJob: ...

    def retry_job(self, job_id: str) -> IngestionJob: ...

    def recover_expired_jobs(self) -> int: ...

    def complete_job(
        self, expected: IngestionJob, *, indexed_chunk_count: int,
        expected_chunk_count: int, index_searchable: bool,
    ) -> IngestionJob: ...

    def soft_delete_paper(self, paper_id: str) -> None: ...


@runtime_checkable
class PaperRetrievalService(Protocol):
    def search(self, query: SearchQuery) -> SearchResult: ...

    def read_section(
        self,
        paper_id: str,
        version: str,
        section_path: tuple[str, ...],
    ) -> tuple[SourceBlock, ...]: ...

    def get_evidence(self, evidence_id: str) -> Evidence | None: ...
