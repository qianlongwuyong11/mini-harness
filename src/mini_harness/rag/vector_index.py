"""ES dense vectors with immutable embedding identity and pre-filtered visibility."""
from dataclasses import asdict, dataclass
import hashlib
import json
from math import isfinite, sqrt
import re

from .elasticsearch import ElasticsearchPaperIndex, chunk_from_dict
from .models import RetrievalChannel, RetrievalHit


@dataclass(frozen=True)
class EmbeddingProfile:
    model_id: str
    revision: str
    dimensions: int
    tokenizer_sha256: str
    max_tokens: int = 512
    pooling: str = "attention-mask mean"
    normalize: str = "L2"
    dtype: str = "float32"
    document_prefix: str = "passage: "
    query_prefix: str = "query: "

    def __post_init__(self):
        if not self.model_id.strip() or not re.fullmatch(r"[a-f0-9]{40}",self.revision):
            raise ValueError("Model identity must contain an immutable revision")
        if not re.fullmatch(r"[a-f0-9]{64}",self.tokenizer_sha256) or not 1 <= self.dimensions <= 4096:
            raise ValueError("Invalid tokenizer fingerprint or dimensions")
        if self.normalize != "L2" or self.dtype != "float32" or self.max_tokens < 1:
            raise ValueError("Expected normalized float32 embeddings and positive token limit")

    @classmethod
    def from_metadata(cls, data):
        return cls(**{name:data[name] for name in cls.__dataclass_fields__})

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()[:16]

    def check_vector(self, vector):
        if len(vector) != self.dimensions:
            raise ValueError("Embedding dimension mismatch")
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not isfinite(x) for x in vector):
            raise ValueError("Vector values must be finite numbers")
        if abs(sqrt(sum(x*x for x in vector))-1) > 1e-4:
            raise ValueError("Vector must be L2-normalized")


class ElasticsearchVectorIndex(ElasticsearchPaperIndex):
    def __init__(self, client, catalog, profile: EmbeddingProfile, index_name=None, **kwargs):
        self.profile = profile
        super().__init__(client,catalog,index_name or "mini-harness-papers-dense-"+profile.fingerprint,**kwargs)

    def mapping(self):
        mapping = super().mapping()
        mapping['_meta'].update(mini_harness_schema='dense-v1',embedding=asdict(self.profile))
        mapping['properties']['embedding_profile'] = {'type':'keyword'}
        mapping['properties']['embedding'] = dict(type='dense_vector',dims=self.profile.dimensions,
            element_type='float',index=True,similarity='dot_product',
            index_options={'type':'hnsw','m':16,'ef_construction':100})
        return mapping

    def _validate_embeddings(self, version, chunks, embeddings):
        if (version.embedding_model,version.embedding_revision) != (self.profile.model_id,self.profile.revision):
            raise ValueError("Document embedding model/revision differs from index")
        if set(embeddings) != {c.chunk_id for c in chunks}:
            raise ValueError("Exactly one vector per chunk is required")
        for vector in embeddings.values():
            self.profile.check_vector(vector)

    def _mapping_matches(self, actual):
        # ES omits the default float element_type when serializing mappings.
        if not isinstance(actual,dict):
            return False
        actual=json.loads(json.dumps(actual))
        field=actual.get('properties',{}).get('embedding',{})
        if field.get('type')=='dense_vector':
            field.setdefault('element_type','float')
        return actual==self.mapping()

    def _embedding_fields(self, chunk_id, embeddings):
        return dict(embedding=list(embeddings[chunk_id]),embedding_profile=self.profile.fingerprint)

    def _active(self, paper_ids=()):
        return {(v.paper_id,v.version) for v in self.catalog.list_active_versions()
                if (not paper_ids or v.paper_id in paper_ids)
                and (v.embedding_model,v.embedding_revision)==(self.profile.model_id,self.profile.revision)}

    def vector_search(self, query, query_vector, *, limit):
        self.profile.check_vector(query_vector)
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        active = self._active(query.paper_ids)
        if not active:
            return ()
        if len(active)>512:
            raise ValueError("Select at most 512 papers")
        filters = [
            {'bool':{'should':[self._version_filter(p,v) for p,v in sorted(active)],'minimum_should_match':1}},
            {'terms':{'source_kind':[k.value for k in query.source_kinds]}},
            {'term':{'embedding_profile':self.profile.fingerprint}},
        ]
        response = self.client.post('/'+self.index_name+'/_search',json={
            'size':limit,'_source':['chunk'], 'track_total_hits':False,
            'sort':[{'_score':'desc'},{'chunk_id':'asc'}],
            'knn':{'field':'embedding','query_vector':list(query_vector),'k':limit,
                   'num_candidates':min(10000,max(100,limit*10)),'filter':filters}})
        response.raise_for_status()
        data=response.json()
        if data.get('timed_out') or data.get('_shards',{}).get('failed',0):
            raise RuntimeError("Incomplete vector search")
        visible = active & self._active(query.paper_ids)
        hits=[]
        for item in data['hits']['hits']:
            chunk=chunk_from_dict(item['_source']['chunk'])
            if (chunk.paper_id,chunk.version) in visible and chunk.source_kind in query.source_kinds:
                hits.append(RetrievalHit(chunk,RetrievalChannel.VECTOR,len(hits)+1,float(item['_score'])))
        return tuple(hits)

    def _lexical_active(self, paper_ids):
        return self._active(paper_ids)

    def _lexical_filters(self):
        return [{'term':{'embedding_profile':self.profile.fingerprint}}]

    def verify_version(self, version, chunks):
        """Check refreshed search visibility, exact IDs and payloads before publication."""
        response=self.client.post('/'+self.index_name+'/_refresh')
        response.raise_for_status()
        if response.json().get('_shards',{}).get('failed',0):
            raise RuntimeError("Refresh failed")
        if not chunks or self.count_version(version.paper_id,version.version)!=len(chunks):
            raise RuntimeError("Indexed chunk count differs from expected count")
        for start in range(0,len(chunks),100):
            batch=chunks[start:start+100]
            response=self.client.post('/'+self.index_name+'/_mget',json={'ids':[c.chunk_id for c in batch]})
            response.raise_for_status()
            docs=response.json()['docs']
            if len(docs)!=len(batch):
                raise RuntimeError("Incomplete document verification")
            for expected,doc in zip(batch,docs):
                source=doc.get('_source',{})
                if (not doc.get('found') or doc['_id']!=expected.chunk_id
                        or source.get('embedding_profile')!=self.profile.fingerprint
                        or source.get('chunk')!=expected.to_dict()):
                    raise RuntimeError("Indexed chunk identity/content mismatch")
        # Count vectors through the search path, rather than relying on realtime GET alone.
        response=self.client.post('/'+self.index_name+'/_count',json={'query':{'bool':{'filter':[
            self._version_filter(version.paper_id,version.version),{'exists':{'field':'embedding'}},
            {'term':{'embedding_profile':self.profile.fingerprint}}]}}})
        response.raise_for_status()
        data=response.json()
        if data.get('_shards',{}).get('failed',0) or data['count']!=len(chunks):
            raise RuntimeError("Not all vectors are searchable")
        return len(chunks)


class VectorSearchService:
    """Public query entry point: bind query encoding to the index's embedding profile."""
    def __init__(self,index,embedding):
        if EmbeddingProfile.from_metadata(embedding.metadata()) != index.profile:
            raise ValueError("Query model/tokenizer/pooling differs from index")
        self.index,self.embedding=index,embedding

    def search(self,query):
        return self.index.vector_search(query,self.embedding.embed_query(query.text),limit=query.top_k)
