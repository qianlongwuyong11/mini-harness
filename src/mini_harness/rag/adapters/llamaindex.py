"""Explicit LlamaIndex bridges. No splitting, global Settings or cloud defaults."""
from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from pydantic import PrivateAttr

from ..interfaces import EmbeddingService
from ...telemetry import record_metric, record_detail
from .tracing import LocalTraceHandler
from ..models import (
    BoundingBox, Chunk, CoordinateSpace, Evidence, PageSpan, SearchQuery,
    SearchResult, SourceBlock, SourceKind,
)


class SearchBackend(Protocol):
    def search(self, query: SearchQuery) -> SearchResult: ...


def to_node(value: SourceBlock | Chunk | Evidence) -> TextNode:
    """One existing structural unit becomes exactly one node, with raw text."""
    data = value.to_dict()
    if isinstance(value, Evidence):
        text = data.pop('excerpt')
        identity = value.chunk_id
    else:
        text = data.pop('text')
        identity = value.chunk_id if isinstance(value, Chunk) else value.block_id
    return TextNode(
        id_=identity, text=text, metadata=data,
        excluded_embed_metadata_keys=list(data),
        excluded_llm_metadata_keys=list(data),
        text_template='{content}',
    )


def to_evidence(hit: NodeWithScore) -> Evidence:
    """Restore the application boundary after retrieval/postprocessing."""
    data = dict(hit.node.metadata)
    data['excerpt'] = hit.node.get_text()
    data['retrieval_score'] = hit.score
    data['source_kind'] = SourceKind(data['source_kind'])
    data['section_path'] = tuple(data['section_path'])
    data['page_spans'] = tuple(PageSpan(**span) for span in data['page_spans'])
    data['bboxes'] = tuple(BoundingBox(**dict(
        box, coordinate_space=CoordinateSpace(box['coordinate_space'])
    )) for box in data['bboxes'])
    if data['chunk_id'] != hit.node.node_id:
        raise ValueError('Evidence node identity changed')
    return Evidence(**data)


class PaperRetriever(BaseRetriever):
    """Fixed request scope; backend owns ES filters, embedding and ranking."""
    def __init__(self, backend: SearchBackend, query: SearchQuery, callback_manager=None):
        super().__init__(callback_manager=callback_manager or CallbackManager([]))
        self.backend = backend
        self.query = query

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        result = self.backend.search(replace(self.query, text=query_bundle.query_str))
        return [NodeWithScore(node=to_node(e), score=e.retrieval_score)
                for e in result.evidence]


class LlamaIndexSearchAdapter:
    """Framework-free search input/output; per-call state supports concurrent queries."""
    def __init__(self, backend: SearchBackend, *, trace_search=None):
        self.backend = backend
        self.trace_search = trace_search

    def search(self, query: SearchQuery) -> SearchResult:
        results: list[SearchResult] = []
        backend = self.backend
        handler=LocalTraceHandler()
        manager=CallbackManager([handler])
        backend_trace={}
        trace_search=self.trace_search
        class Capture:
            def search(self, request: SearchQuery) -> SearchResult:
                if trace_search is not None:
                    result,trace=trace_search(request,callback_manager=manager)
                    backend_trace.update(trace)
                else:
                    result=backend.search(request)
                results.append(result)
                return result
        status='error'
        try:
            hits = PaperRetriever(Capture(), query, manager).retrieve(query.text)
            result=replace(results[0], evidence=tuple(to_evidence(hit) for hit in hits))
            status='ok'
            return result
        finally:
            record_detail('rag/retrieval',backend_trace,trace_id=handler.trace_id,status=status,
                score_semantics='retrieval_scores_not_probabilities',reranker_enabled=False)
            record_metric('rag/trace',dict(trace_id=handler.trace_id,engine='llamaindex_callbacks',
                status=status,top_k=query.top_k,spans=handler.spans,
                candidate_count=results[0].candidate_count if results else None,
                returned_count=len(results[0].evidence) if results else None,
                bm25_candidates=len(backend_trace.get('bm25_candidates',[])),
                vector_candidates=len(backend_trace.get('vector_candidates',[])),
                embedding_profile=backend_trace.get('embedding_profile'),
                active_versions=backend_trace.get('active_versions',[])))


class LocalEmbeddingAdapter(BaseEmbedding):
    """Reuse the injected encoder; it owns E5 prefixes and normalization."""
    _service: EmbeddingService = PrivateAttr()

    def __init__(self, service: EmbeddingService):
        super().__init__(model_name=service.model_id, callback_manager=CallbackManager([]))
        self._service = service

    def _get_query_embedding(self, query: str) -> list[float]:
        return list(self._service.embed_query(query))

    async def _aget_query_embedding(self, query: str) -> list[float]:
        # Same serialized CPU encoder semantics as the existing application.
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [list(vector) for vector in self._service.embed_documents(texts)]
