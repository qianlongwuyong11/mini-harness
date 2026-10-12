"""Pure BM25 adapter. Catalog visibility is checked before and after search.

The injected HTTP client is owned by ESService. Bulk success never publishes a
catalog version; the ingestion coordinator must verify and publish separately.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import re

import httpx

from .interfaces import PaperCatalog
from .models import (BoundingBox, Chunk, CoordinateSpace, PageSpan, PaperVersion,
                     RetrievalChannel, RetrievalHit, SearchQuery, SourceKind,
                     Vector, VersionState, content_hash, stable_chunk_id)


class BulkWriteError(RuntimeError):
    def __init__(self, failures: tuple[tuple[str, int, str], ...]):
        self.failures = failures
        super().__init__(f'{len(failures)} bulk item(s) failed; version must remain unpublished: {failures}')


def chunk_from_dict(data: dict) -> Chunk:
    data = dict(data)
    data['source_kind'] = SourceKind(data['source_kind'])
    for key in ('section_path', 'source_block_ids'):
        data[key] = tuple(data[key])
    data['page_spans'] = tuple(PageSpan(**item) for item in data['page_spans'])
    data['bboxes'] = tuple(BoundingBox(**dict(item,
        coordinate_space=CoordinateSpace(item['coordinate_space']))) for item in data['bboxes'])
    return Chunk(**data)


class ElasticsearchPaperIndex:
    def __init__(self, client: httpx.Client, catalog: PaperCatalog,
                 index_name: str = 'mini-harness-papers-bm25-v1', *,
                 analyzer: str = 'standard', batch_size: int = 100,
                 max_batch_bytes: int = 4 * 1024 * 1024):
        if not re.fullmatch(r'[a-z][a-z0-9_-]{0,199}', index_name):
            raise ValueError('Use one concrete lowercase index name; aliases/wildcards are not accepted')
        if analyzer not in ('standard', 'english'):
            raise ValueError('Supported baseline analyzers: standard, english')
        if batch_size < 1 or max_batch_bytes < 1:
            raise ValueError('Batch limits must be positive')
        self.client, self.catalog, self.index_name = client, catalog, index_name
        self.analyzer, self.batch_size, self.max_batch_bytes = analyzer, batch_size, max_batch_bytes

    def mapping(self) -> dict:
        properties = {name: {'type': 'keyword'} for name in
                      ('paper_id', 'version', 'chunk_id', 'source_kind')}
        properties.update({name: {'type': 'text', 'analyzer': self.analyzer, 'similarity': 'BM25'}
                           for name in ('title', 'section', 'content')})
        properties.update({
            'page_start': {'type': 'integer'}, 'page_end': {'type': 'integer'},
            'chunk': {'type': 'object', 'enabled': False},
        })
        return {'dynamic': 'strict', '_meta': {'mini_harness_schema': 'bm25-v1',
                'analyzer': self.analyzer}, 'properties': properties}

    def ensure_index(self) -> None:
        response = self.client.put('/' + self.index_name, json={
            'settings': {'number_of_shards': 1, 'number_of_replicas': 0},
            'mappings': self.mapping()})
        if response.status_code == 400 and response.json().get('error', {}).get('type') == 'resource_already_exists_exception':
            response = self.client.get('/' + self.index_name + '/_mapping')
            response.raise_for_status()
            if not self._mapping_matches(response.json().get(self.index_name, {}).get('mappings')):
                raise ValueError('Existing mapping differs; use a new physical index')
        else:
            response.raise_for_status()
            if not response.json().get('acknowledged'):
                raise RuntimeError('Index creation not acknowledged')

    def _mapping_matches(self, actual):
        return actual == self.mapping()

    def _bulk(self, documents: list[tuple[str, bytes]]) -> None:
        response = self.client.post('/' + self.index_name + '/_bulk',
            params={'refresh': 'wait_for'}, content=b''.join(raw for _, raw in documents),
            headers={'Content-Type': 'application/x-ndjson'})
        response.raise_for_status()
        data = response.json()
        items = data.get('items', [])
        if len(items) != len(documents):
            raise RuntimeError('Incomplete bulk acknowledgement; retry is safe')
        failures = []
        for (chunk_id, _), item in zip(documents, items):
            result = item.get('index', {})
            status = result.get('status', 0)
            if result.get('_id') != chunk_id or not 200 <= status < 300 or 'error' in result:
                failures.append((chunk_id, status, result.get('error', {}).get('type', 'invalid_acknowledgement')))
        if failures:
            raise BulkWriteError(tuple(failures))
        if data.get('errors'):
            raise RuntimeError('Bulk reported errors without identifiable failed items')

    def upsert_chunks(self, version: PaperVersion, chunks: Sequence[Chunk],
                      embeddings: Mapping[str, Vector]) -> None:
        self._validate_embeddings(version, chunks, embeddings)
        if self.catalog.get_version(version.paper_id, version.version) != version or version.state is not VersionState.STAGING:
            raise ValueError('Only the registered staging version may be written')
        paper = self.catalog.get_paper(version.paper_id)
        if paper is None:
            raise ValueError('Paper missing or deleted')
        ids = set()
        # Validate the entire input before the first network write.
        for chunk in chunks:
            if ((chunk.paper_id, chunk.version) != (version.paper_id, version.version)
                    or chunk.chunk_id != stable_chunk_id(version.paper_id, version.version, version.chunker_version, chunk.ordinal)
                    or chunk.content_hash != content_hash(chunk.text) or chunk.chunk_id in ids):
                raise ValueError('Chunk identity, content hash or version mismatch')
            ids.add(chunk.chunk_id)
        batch, size = [], 0
        for chunk in chunks:
            document = dict(paper_id=chunk.paper_id, version=chunk.version,
                chunk_id=chunk.chunk_id, source_kind=chunk.source_kind.value,
                title=paper.title or paper.paper_id, section=' / '.join(chunk.section_path),
                content=chunk.text, chunk=chunk.to_dict())
            document.update(self._embedding_fields(chunk.chunk_id, embeddings))
            if chunk.page_spans:
                document.update(page_start=chunk.page_spans[0].start, page_end=chunk.page_spans[-1].end)
            raw = (json.dumps({'index': {'_id': chunk.chunk_id}}) + '\n' +
                   json.dumps(document, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
            if len(raw) > self.max_batch_bytes:
                raise ValueError('A chunk exceeds max_batch_bytes; split it before indexing')
            if batch and (len(batch) >= self.batch_size or size + len(raw) > self.max_batch_bytes):
                self._bulk(batch)
                batch, size = [], 0
            batch.append((chunk.chunk_id, raw))
            size += len(raw)
        if batch:
            self._bulk(batch)

    def _validate_embeddings(self, version, chunks, embeddings):
        if embeddings:
            raise ValueError('BM25-only index does not accept embeddings')

    def _embedding_fields(self, chunk_id, embeddings):
        return {}

    @staticmethod
    def _version_filter(paper_id: str, version: str) -> dict:
        return {'bool': {'filter': [{'term': {'paper_id': paper_id}}, {'term': {'version': version}}]}}

    def lexical_search(self, query: SearchQuery, *, limit: int) -> tuple[RetrievalHit, ...]:
        if not 1 <= limit <= 100:
            raise ValueError('limit must be between 1 and 100')
        active = self._lexical_active(query.paper_ids)
        if not active:
            return ()
        if len(active) > 512:
            raise ValueError('Select at most 512 papers for this local baseline')
        response = self.client.post('/' + self.index_name + '/_search', json={
            'size': limit, 'track_total_hits': False,
            'sort': [{'_score': 'desc'}, {'chunk_id': 'asc'}],
            'query': {'bool': {
                'must': [{'multi_match': {'query': query.text, 'fields': ['title^2', 'section^2', 'content']}}],
                'filter': [
                    {'bool': {'should': [self._version_filter(p,v) for p,v in sorted(active)], 'minimum_should_match': 1}},
                    {'terms': {'source_kind': [kind.value for kind in query.source_kinds]}},
                ] + self._lexical_filters()}}})
        response.raise_for_status()
        data = response.json()
        if data.get('timed_out') or data.get('_shards', {}).get('failed', 0):
            raise RuntimeError('Search incomplete; no partial evidence returned')
        # A deletion or activation during the ES request must invalidate old hits.
        current = {(v.paper_id, v.version) for v in self.catalog.list_active_versions()}
        hits = []
        for hit in data['hits']['hits']:
            chunk = chunk_from_dict(hit['_source']['chunk'])
            if ((chunk.paper_id, chunk.version) in active & current
                    and chunk.source_kind in query.source_kinds):
                hits.append(RetrievalHit(chunk, RetrievalChannel.BM25, len(hits)+1, float(hit['_score'])))
        return tuple(hits)

    def _lexical_active(self, paper_ids):
        return {(v.paper_id,v.version) for v in self.catalog.list_active_versions()
                if not paper_ids or v.paper_id in paper_ids}

    def _lexical_filters(self):
        return []

    def count_version(self, paper_id: str, version: str) -> int:
        response = self.client.post('/' + self.index_name + '/_count',
                                    json={'query': self._version_filter(paper_id, version)})
        response.raise_for_status()
        data = response.json()
        if data.get('_shards', {}).get('failed', 0):
            raise RuntimeError('Incomplete count')
        return data['count']

    def delete_version(self, paper_id: str, version: str) -> None:
        active = self.catalog.get_active_version(paper_id)
        if active is not None and active.version == version:
            raise ValueError('Deactivate/soft-delete in catalog before deleting active index data')
        response = self.client.post('/' + self.index_name + '/_delete_by_query',
            params={'refresh': 'true'}, json={'query': self._version_filter(paper_id, version)})
        response.raise_for_status()
        data = response.json()
        if data.get('timed_out') or data.get('failures') or data.get('version_conflicts'):
            raise RuntimeError('Version cleanup incomplete; retry required')

    def vector_search(self, query: SearchQuery, query_vector: Vector, *, limit: int) -> tuple[RetrievalHit, ...]:
        raise NotImplementedError('Vector search arrives in step 8; this is the BM25 baseline')
