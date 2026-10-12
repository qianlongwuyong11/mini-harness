"""论文 RAG 的框架无关领域模型。

这些类型是 mini-harness 自己的契约。Docling、Elasticsearch 和 LlamaIndex 的
对象必须在各自 adapter 内转换，不能穿透到工具、Session 或上层业务。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any, Mapping, cast


class SourceKind(str, Enum):
    PAPER = "paper"
    USER_ANNOTATION = "user_annotation"
    MIXED = "mixed"
    GENERATED_SUMMARY = "generated_summary"


class BlockKind(str, Enum):
    HEADING = "heading"
    TEXT = "text"
    TABLE = "table"
    FORMULA = "formula"
    FIGURE_CAPTION = "figure_caption"
    ANNOTATION = "annotation"


class CoordinateSpace(str, Enum):
    PDF_POINTS = "pdf_points"
    NORMALIZED = "normalized"


class VersionState(str, Enum):
    STAGING = "staging"
    READY = "ready"
    FAILED = "failed"
    INACTIVE = "inactive"


class IngestionStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    FAILED = "failed"
    READY = "ready"
    CANCELLED = "cancelled"


class IngestionStage(str, Enum):
    WAITING = "waiting"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    PUBLISHING = "publishing"
    COMPLETED = "completed"


class RetrievalMode(str, Enum):
    LEXICAL = "lexical"
    VECTOR = "vector"
    HYBRID = "hybrid"


class RetrievalChannel(str, Enum):
    BM25 = "bm25"
    VECTOR = "vector"


Vector = tuple[float, ...]


class JsonModel:
    """为领域对象提供不依赖框架的 JSON 安全投影。"""

    def to_dict(self) -> dict[str, Any]:
        value = _json_value(self)
        return cast(dict[str, Any], value)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class PageSpan(JsonModel):
    """物理 PDF 页码范围，页码从 1 开始且两端都包含。"""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 1 or self.end < self.start:
            raise ValueError("PageSpan 必须是从 1 开始的有效闭区间")


@dataclass(frozen=True)
class BoundingBox(JsonModel):
    """一页内的来源位置；坐标可以是 PDF points 或归一化坐标。"""

    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    coordinate_space: CoordinateSpace = CoordinateSpace.PDF_POINTS

    def __post_init__(self) -> None:
        _enum_value("coordinate_space", self.coordinate_space, CoordinateSpace)
        coordinates = (self.x0, self.y0, self.x1, self.y1)
        if self.page < 1:
            raise ValueError("BoundingBox.page 必须从 1 开始")
        if not all(isfinite(value) for value in coordinates):
            raise ValueError("BoundingBox 坐标必须是有限数")
        if self.x0 < 0 or self.y0 < 0 or self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("BoundingBox 坐标范围无效")
        if self.coordinate_space is CoordinateSpace.NORMALIZED and (
            self.x1 > 1 or self.y1 > 1
        ):
            raise ValueError("归一化 BoundingBox 坐标不能超过 1")


@dataclass(frozen=True)
class Paper(JsonModel):
    paper_id: str
    source_uri: str
    source_hash: str
    title: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        _required("paper_id", self.paper_id)
        _required("source_uri", self.source_uri)
        _required("source_hash", self.source_hash)
        if self.title is not None:
            _required("title", self.title)
        _aware("created_at", self.created_at)


@dataclass(frozen=True)
class PaperVersion(JsonModel):
    paper_id: str
    version: str
    source_hash: str
    parser_version: str
    chunker_version: str
    embedding_model: str | None = None
    embedding_revision: str | None = None
    state: VersionState = VersionState.STAGING
    chunk_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        for name in (
            "paper_id",
            "version",
            "source_hash",
            "parser_version",
            "chunker_version",
        ):
            _required(name, cast(str, getattr(self, name)))
        if self.embedding_model is not None:
            _required("embedding_model", self.embedding_model)
        if self.embedding_revision is not None:
            _required("embedding_revision", self.embedding_revision)
        if self.embedding_revision is not None and self.embedding_model is None:
            raise ValueError("embedding_revision 需要同时提供 embedding_model")
        _enum_value("state", self.state, VersionState)
        if self.chunk_count < 0:
            raise ValueError("chunk_count 不能为负数")
        if self.state is VersionState.READY and self.chunk_count == 0:
            raise ValueError("READY 版本必须至少包含一个 chunk")
        _aware("created_at", self.created_at)


@dataclass(frozen=True)
class SourceBlock(JsonModel):
    block_id: str
    paper_id: str
    version: str
    text: str
    kind: BlockKind
    source_kind: SourceKind
    section_path: tuple[str, ...]
    page_spans: tuple[PageSpan, ...]
    bboxes: tuple[BoundingBox, ...] = ()
    parent_block_id: str | None = None
    previous_block_id: str | None = None
    next_block_id: str | None = None
    extraction_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("block_id", "paper_id", "version", "text"):
            _required(name, cast(str, getattr(self, name)))
        _enum_value("kind", self.kind, BlockKind)
        if not isinstance(self.extraction_warnings, tuple) or any(
            not isinstance(value, str) or not value.strip()
            for value in self.extraction_warnings
        ):
            raise ValueError("extraction_warnings must be a tuple of nonempty strings")
        _enum_value("source_kind", self.source_kind, SourceKind)
        _section_path(self.section_path)
        _source_location(self.source_kind, self.page_spans, self.bboxes)
        _relations(
            self.block_id,
            self.parent_block_id,
            self.previous_block_id,
            self.next_block_id,
        )


@dataclass(frozen=True)
class Chunk(JsonModel):
    chunk_id: str
    paper_id: str
    version: str
    ordinal: int
    text: str
    content_hash: str
    token_count: int
    source_kind: SourceKind
    section_path: tuple[str, ...]
    page_spans: tuple[PageSpan, ...]
    source_block_ids: tuple[str, ...]
    bboxes: tuple[BoundingBox, ...] = ()
    parent_chunk_id: str | None = None
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("chunk_id", "paper_id", "version", "text", "content_hash"):
            _required(name, cast(str, getattr(self, name)))
        if self.ordinal < 0:
            raise ValueError("Chunk.ordinal 不能为负数")
        if self.token_count <= 0:
            raise ValueError("Chunk.token_count 必须为正数")
        _enum_value("source_kind", self.source_kind, SourceKind)
        _section_path(self.section_path)
        _source_location(self.source_kind, self.page_spans, self.bboxes)
        _non_empty_ids("source_block_ids", self.source_block_ids)
        _relations(
            self.chunk_id,
            self.parent_chunk_id,
            self.previous_chunk_id,
            self.next_chunk_id,
        )


@dataclass(frozen=True)
class Evidence(JsonModel):
    evidence_id: str
    paper_id: str
    version: str
    chunk_id: str
    title: str
    excerpt: str
    source_kind: SourceKind
    section_path: tuple[str, ...]
    page_spans: tuple[PageSpan, ...]
    bboxes: tuple[BoundingBox, ...] = ()
    retrieval_score: float | None = None
    bm25_rank: int | None = None
    vector_rank: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "evidence_id",
            "paper_id",
            "version",
            "chunk_id",
            "title",
            "excerpt",
        ):
            _required(name, cast(str, getattr(self, name)))
        _enum_value("source_kind", self.source_kind, SourceKind)
        _section_path(self.section_path)
        _source_location(self.source_kind, self.page_spans, self.bboxes)
        if self.retrieval_score is not None and not isfinite(self.retrieval_score):
            raise ValueError("retrieval_score 必须是有限数")
        for name in ("bm25_rank", "vector_rank"):
            rank = getattr(self, name)
            if rank is not None and rank < 1:
                raise ValueError(f"{name} 必须从 1 开始")


@dataclass(frozen=True)
class SearchQuery(JsonModel):
    text: str
    paper_ids: tuple[str, ...] = ()
    source_kinds: tuple[SourceKind, ...] = (SourceKind.PAPER,)
    top_k: int = 8

    def __post_init__(self) -> None:
        _required("text", self.text)
        if self.top_k < 1 or self.top_k > 100:
            raise ValueError("top_k 必须在 1 到 100 之间")
        if self.paper_ids:
            _non_empty_ids("paper_ids", self.paper_ids)
        if any(not isinstance(item, SourceKind) for item in self.source_kinds):
            raise ValueError("source_kinds 只能包含 SourceKind")
        if not self.source_kinds or len(set(self.source_kinds)) != len(
            self.source_kinds
        ):
            raise ValueError("source_kinds 必须非空且不能重复")


@dataclass(frozen=True)
class SearchResult(JsonModel):
    query: SearchQuery
    evidence: tuple[Evidence, ...]
    mode: RetrievalMode
    candidate_count: int

    def __post_init__(self) -> None:
        _enum_value("mode", self.mode, RetrievalMode)
        if self.candidate_count < len(self.evidence):
            raise ValueError("candidate_count 不能小于返回的 evidence 数量")
        if len(self.evidence) > self.query.top_k:
            raise ValueError("evidence 数量不能超过 query.top_k")
        ids = [item.evidence_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("同一 SearchResult 中 evidence_id 不能重复")


@dataclass(frozen=True)
class RetrievalHit(JsonModel):
    chunk: Chunk
    channel: RetrievalChannel
    rank: int
    score: float

    def __post_init__(self) -> None:
        _enum_value("channel", self.channel, RetrievalChannel)
        if self.rank < 1:
            raise ValueError("RetrievalHit.rank 必须从 1 开始")
        if not isfinite(self.score):
            raise ValueError("RetrievalHit.score 必须是有限数")


@dataclass(frozen=True)
class IngestionJob(JsonModel):
    job_id: str
    paper_id: str
    target_version: str
    source_uri: str
    source_hash: str
    status: IngestionStatus = IngestionStatus.QUEUED
    stage: IngestionStage = IngestionStage.WAITING
    attempts: int = 0
    progress_current: int = 0
    progress_total: int | None = None
    error_message: str | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        for name in (
            "job_id",
            "paper_id",
            "target_version",
            "source_uri",
            "source_hash",
        ):
            _required(name, cast(str, getattr(self, name)))
        _enum_value("status", self.status, IngestionStatus)
        _enum_value("stage", self.stage, IngestionStage)
        if self.attempts < 0 or self.progress_current < 0:
            raise ValueError("attempts 和 progress_current 不能为负数")
        if self.progress_total is not None and (
            self.progress_total < 0 or self.progress_current > self.progress_total
        ):
            raise ValueError("任务进度范围无效")
        if self.status is IngestionStatus.QUEUED and self.stage is not IngestionStage.WAITING:
            raise ValueError("QUEUED 任务必须处于 WAITING 阶段")
        if self.status is IngestionStatus.READY and self.stage is not IngestionStage.COMPLETED:
            raise ValueError("READY 任务必须处于 COMPLETED 阶段")
        if self.lease_expires_at is not None and self.lease_owner is None:
            raise ValueError("lease_expires_at 需要同时提供 lease_owner")
        if self.lease_owner is not None:
            _required("lease_owner", self.lease_owner)
        if self.error_message is not None:
            _required("error_message", self.error_message)
        _aware("created_at", self.created_at)
        _aware("updated_at", self.updated_at)
        if self.updated_at < self.created_at:
            raise ValueError("updated_at 不能早于 created_at")
        if self.lease_expires_at is not None:
            _aware("lease_expires_at", self.lease_expires_at)


def content_hash(text: str) -> str:
    """返回稳定的 SHA-256 内容摘要。"""

    _required("text", text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_chunk_id(
    paper_id: str,
    version: str,
    chunker_version: str,
    ordinal: int,
) -> str:
    """同一文档版本与分块配置在失败重试时生成相同 chunk ID。"""

    for name, value in (
        ("paper_id", paper_id),
        ("version", version),
        ("chunker_version", chunker_version),
    ):
        _required(name, value)
    if ordinal < 0:
        raise ValueError("ordinal 不能为负数")
    canonical = "\x1f".join((paper_id, version, chunker_version, str(ordinal)))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
    return f"chunk-{digest}"


def _required(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 不能为空")


def _enum_value(name: str, value: Enum, enum_type: type[Enum]) -> None:
    if not isinstance(value, enum_type):
        raise ValueError(f"{name} 必须是 {enum_type.__name__}")


def _aware(name: str, value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} 必须是带时区的 datetime")


def _section_path(path: tuple[str, ...]) -> None:
    if not isinstance(path, tuple) or any(
        not isinstance(item, str) or not item.strip() for item in path
    ):
        raise ValueError("section_path 必须是只包含非空标题的 tuple")


def _non_empty_ids(name: str, values: tuple[str, ...]) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} 必须是非空 tuple")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} 不能包含空 ID")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} 不能包含重复 ID")


def _source_location(
    source_kind: SourceKind,
    spans: tuple[PageSpan, ...],
    boxes: tuple[BoundingBox, ...],
) -> None:
    if not isinstance(spans, tuple) or not isinstance(boxes, tuple):
        raise ValueError("page_spans 和 bboxes 必须使用 tuple")
    previous_end = 0
    for span in spans:
        if span.start <= previous_end:
            raise ValueError("page_spans 必须按页码升序且不能重叠")
        previous_end = span.end
    if source_kind is not SourceKind.GENERATED_SUMMARY and not spans:
        raise ValueError("原文或批注必须包含至少一个页码范围")
    for box in boxes:
        if not any(span.start <= box.page <= span.end for span in spans):
            raise ValueError("BoundingBox 页码必须位于 page_spans 内")


def _relations(item_id: str, *related_ids: str | None) -> None:
    present = [value for value in related_ids if value is not None]
    for value in present:
        _required("relation_id", cast(str, value))
    if item_id in present:
        raise ValueError("对象不能引用自身作为父级或相邻节点")
    if len(present) != len(set(present)):
        raise ValueError("父级和相邻节点引用不能重复")


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"无法将 {type(value).__name__} 转换为 JSON")
