"""Inspectable terminology expansion and rank-based fusion on one version snapshot."""
from dataclasses import dataclass, replace
from math import isfinite
from time import perf_counter

from .models import Evidence, SearchResult, RetrievalMode
from .vector_index import ElasticsearchVectorIndex, VectorSearchService

# Explicit, domain-specific glossary. No model calls or hidden translation.
TERMS = {
    '查询向量': 'query feature', '几何': 'geometry', '运动': 'motion',
    '自注意力': 'self attention', '距离': 'distance', '系数': 'coefficient',
    '感受野': 'receptive field', '类别': 'category', '物体': 'object',
    '速度': 'velocity', '对齐': 'alignment', '历史帧': 'previous frames',
    '采样': 'sampling', '通道': 'channel', '空间点': 'point', '混合': 'mixing',
    '双分支': 'dual branch', '慢速流': 'slow stream', '快速流': 'fast stream',
    '视频帧': 'video frames', '帧数': 'number frames', '采样点': 'sampling points',
    '自车位姿': 'ego pose', '局限': 'limitation', '消融': 'ablation',
    '柱状': 'pillar', '初始化': 'initialization',
}


@dataclass(frozen=True)
class HybridConfig:
    candidate_limit: int = 20
    rrf_k: int = 60
    bm25_weight: float = 1.0
    vector_weight: float = 1.0
    expand_terms: bool = True

    def __post_init__(self):
        if not 1 <= self.candidate_limit <= 100 or self.rrf_k < 1:
            raise ValueError('Invalid candidate limit or RRF constant')
        if any(not isfinite(v) or v <= 0 for v in (self.bm25_weight,self.vector_weight)):
            raise ValueError('Channel weights must be positive finite numbers')


def expand_query(text, enabled=True):
    matched = {key:value for key,value in TERMS.items() if enabled and key in text}
    words = list(dict.fromkeys(matched.values()))
    return text + (' ' + ' '.join(words) if words else ''), matched


def reciprocal_rank_fusion(lexical, vector, config):
    rows = {}
    for hits, name, weight in ((lexical,'bm25',config.bm25_weight),(vector,'vector',config.vector_weight)):
        seen = set()
        for hit in hits:
            key = hit.chunk.chunk_id
            if key in seen:
                continue
            seen.add(key)
            row = rows.setdefault(key, dict(chunk=hit.chunk, score=0., bm25_rank=None,vector_rank=None,
                                            bm25_contribution=0.,vector_contribution=0.))
            if row['chunk'] != hit.chunk:
                raise ValueError('Same chunk ID has conflicting payloads')
            contribution = weight/(config.rrf_k+hit.rank)
            row['score'] += contribution
            row[name+'_rank'] = hit.rank
            row[name+'_contribution'] = contribution
    return sorted(rows.values(),key=lambda row:(-row['score'],row['chunk'].chunk_id))


class _SnapshotCatalog:
    def __init__(self,catalog,versions):
        self.catalog,self.versions=catalog,versions
    def list_active_versions(self):
        return self.versions
    def __getattr__(self,name):
        return getattr(self.catalog,name)


class HybridSearchService:
    def __init__(self,index,embedding,config=HybridConfig()):
        VectorSearchService(index,embedding)  # Validate encoder identity before either route.
        self.index,self.embedding,self.config=index,embedding,config

    def search(self,query):
        return self.search_with_trace(query)[0]

    def search_with_trace(self,query):
        started=perf_counter()
        catalog=self.index.catalog
        versions=tuple(v for v in catalog.list_active_versions()
                       if (not query.paper_ids or v.paper_id in query.paper_ids)
                       and (v.embedding_model,v.embedding_revision)==(self.index.profile.model_id,self.index.profile.revision))
        snapshot=_SnapshotCatalog(catalog,versions)
        index=ElasticsearchVectorIndex(self.index.client,snapshot,self.index.profile,self.index.index_name,
                                       analyzer=self.index.analyzer)
        rewritten,terms=expand_query(query.text,self.config.expand_terms)
        trace=dict(original_query=query.text,bm25_query=rewritten,vector_query=query.text,matched_terms=terms,
                   active_versions=[(v.paper_id,v.version) for v in versions],source_kinds=[k.value for k in query.source_kinds],
                   embedding_profile=index.profile.fingerprint,config=self.config.__dict__.copy())
        lexical=vector=()
        if versions:
            limit=max(query.top_k,self.config.candidate_limit)
            t=perf_counter()
            lexical=index.lexical_search(replace(query,text=rewritten),limit=limit)
            trace['bm25_ms']=(perf_counter()-t)*1000
            t=perf_counter()
            encoded=self.embedding.embed_query(query.text)
            trace['encoding_ms']=(perf_counter()-t)*1000
            t=perf_counter()
            vector=index.vector_search(query,encoded,limit=limit)
            trace['knn_ms']=(perf_counter()-t)*1000
        # Both ES routes use the same frozen prefilter. Recheck live state after both finish.
        visible={(v.paper_id,v.version) for v in catalog.list_active_versions()} & {(v.paper_id,v.version) for v in versions}
        def allowed(hit):
            return (hit.chunk.paper_id,hit.chunk.version) in visible and hit.chunk.source_kind in query.source_kinds
        lexical=tuple(h for h in lexical if allowed(h))
        vector=tuple(h for h in vector if allowed(h))
        fused=reciprocal_rank_fusion(lexical,vector,self.config)
        evidence=[]
        for row in fused[:query.top_k]:
            c=row['chunk']
            paper=catalog.get_paper(c.paper_id)
            if paper is None:
                continue
            evidence.append(Evidence('evidence-'+c.chunk_id,c.paper_id,c.version,c.chunk_id,paper.title or c.paper_id,
                c.text,c.source_kind,c.section_path,c.page_spans,c.bboxes,row['score'],row['bm25_rank'],row['vector_rank']))
        trace['bm25_candidates']=[h.to_dict() for h in lexical]
        trace['vector_candidates']=[h.to_dict() for h in vector]
        trace['fusion']=[dict(chunk_id=r['chunk'].chunk_id,**{k:v for k,v in r.items() if k!='chunk'}) for r in fused]
        trace['total_ms']=(perf_counter()-started)*1000
        return SearchResult(query,tuple(evidence),RetrievalMode.HYBRID,len(fused)),trace
