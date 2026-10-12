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
    def __init__(self, backend: SearchBackend, query: SearchQuery):
        super().__init__(callback_manager=CallbackManager([]))
        self.backend = backend
        self.query = query

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        result = self.backend.search(replace(self.query, text=query_bundle.query_str))
        return [NodeWithScore(node=to_node(e), score=e.retrieval_score)
                for e in result.evidence]


class LlamaIndexSearchAdapter:
    """Framework-free search input/output; per-call state supports concurrent queries."""
    def __init__(self, backend: SearchBackend):
        self.backend = backend

    def search(self, query: SearchQuery) -> SearchResult:
        # Capture this invocation's result only; never run the backend twice.
        results: list[SearchResult] = []
        backend = self.backend

        class Capture:
            def search(self, request: SearchQuery) -> SearchResult:
                result = backend.search(request)
                results.append(result)
                return result

        hits = PaperRetriever(Capture(), query).retrieve(query.text)
        return replace(results[0], evidence=tuple(to_evidence(hit) for hit in hits))


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
