"""Application-scoped paper services; models and ES are initialized on first use."""
from dataclasses import dataclass
from pathlib import Path
import threading
from time import perf_counter
from ..telemetry import record_metric

from .catalog import SQLitePaperCatalog
from .models import Evidence, SourceKind


@dataclass(frozen=True)
class RagConfig:
    catalog_path: str = '.mini-harness/rag/vector-catalog.sqlite3'
    root: str = '.mini-harness/rag/ingestion'
    es_config: str = '.mini-harness/elasticsearch/connection.json'
    model_root: str = '.mini-harness/rag/models'
    index_name: str | None = None
    evidence_tokens: int = 6000


class PaperRuntime:
    def __init__(self, config=RagConfig()):
        self.config = config
        self.catalog = SQLitePaperCatalog(config.catalog_path)
        self._lock = threading.RLock()
        self._es = self._embedding = self._search = self._worker = None
        self._thread = None
        self._stop = threading.Event()
        self._closed = False

    def _load(self):
        with self._lock:
            if self._closed:
                raise RuntimeError('Paper services are closed')
            if self._search is not None:
                return
            from .embedding import LocalE5Embedding, load_default_config
            from .es_service import ESConfig, ESService
            from .vector_index import EmbeddingProfile, ElasticsearchVectorIndex
            from .hybrid import HybridSearchService
            from .adapters.llamaindex import LlamaIndexSearchAdapter
            from .chunking import StructuredChunker, EmbeddingTokenizer
            from .parsers.docling import DoclingPaperParser, DoclingParserConfig
            from .ingestion import IngestionWorker
            cfg = load_default_config()
            model = Path(self.config.model_root) / cfg.model_id.split('/')[-1]
            embedding = LocalE5Embedding(cfg, model)
            es = ESService(ESConfig.load(Path(self.config.es_config)))
            try:
                index = ElasticsearchVectorIndex(es.client, self.catalog,
                    EmbeddingProfile.from_metadata(embedding.metadata()), self.config.index_name, analyzer='english')
                hybrid = HybridSearchService(index, embedding)
                search = LlamaIndexSearchAdapter(hybrid, trace_search=hybrid.search_with_trace)
                parser = DoclingPaperParser(DoclingParserConfig(Path(self.config.root)/'parsed'))
                chunker = StructuredChunker(EmbeddingTokenizer(model/'tokenizer.json', cfg.model_id))
                worker = IngestionWorker(self.catalog, parser, chunker, embedding, index, self.config.root)
            except BaseException:
                es.close()
                raise
            self._es, self._embedding, self._search, self._worker = es, embedding, search, worker

    def list_papers(self):
        with self.catalog._connection() as db:
            rows = db.execute('SELECT paper_id FROM papers WHERE deleted=0 ORDER BY paper_id').fetchall()
        result = []
        for (identity,) in rows:
            paper = self.catalog.get_paper(identity)
            if paper is None:
                continue
            version = self.catalog.get_active_version(identity)
            sections = []
            if version:
                seen = set()
                for block in self.catalog.get_blocks(identity, version.version):
                    key = (block.section_path, block.source_kind)
                    if key not in seen:
                        seen.add(key)
                        sections.append(dict(section_path=list(block.section_path),source_kind=block.source_kind.value))
            result.append(dict(paper_id=identity,title=paper.title or identity,
                ready=version is not None,version=version.version if version else None,sections=sections))
        return result

    def search(self, query):
        # Avoid loading the encoder for an empty catalog.
        if not self.catalog.list_active_versions():
            from .models import SearchResult, RetrievalMode
            return SearchResult(query, (), RetrievalMode.HYBRID, 0)
        started=perf_counter()
        with self._lock:
            acquired=perf_counter()
            was_loaded=self._search is not None
            try:
                self._load()
            finally:
                record_metric('rag/runtime',dict(lock_wait_ms=(acquired-started)*1000,
                    initialization_ms=(perf_counter()-acquired)*1000,cold=not was_loaded))
            return self._search.search(query)

    def read_section(self, paper_id, version, section_path):
        self._active(paper_id, version)
        blocks = self.catalog.get_blocks(paper_id, version)
        return tuple(b for b in blocks if b.section_path[:len(section_path)] == section_path)

    def _active(self, paper_id, version):
        active = self.catalog.get_active_version(paper_id)
        if active is None or active.version != version:
            raise ValueError('Paper deleted, not ready, or version no longer active; search again')
        return active

    def validate(self, evidence):
        self._active(evidence.paper_id, evidence.version)
        blocks = self.catalog.get_blocks(evidence.paper_id, evidence.version)
        located = [b for b in blocks if b.source_kind == evidence.source_kind
                   and b.section_path == evidence.section_path]
        pages = {p for b in located for s in b.page_spans for p in range(s.start,s.end+1)}
        required = {p for s in evidence.page_spans for p in range(s.start,s.end+1)}
        if not required or not required <= pages:
            raise ValueError('Evidence page locator is missing')
        if evidence.bboxes and any(box not in tuple(x for b in located for x in b.bboxes) for box in evidence.bboxes):
            raise ValueError('Evidence bounding box is missing')

    def section_evidence(self, paper_id, version, section_path, source_kind=SourceKind.PAPER):
        blocks = self.read_section(paper_id, version, section_path)
        paper = self.catalog.get_paper(paper_id)
        return tuple(Evidence('block-'+b.block_id,paper_id,version,b.block_id,paper.title or paper_id,
            b.text,b.source_kind,b.section_path,b.page_spans,b.bboxes)
            for b in blocks if b.source_kind is source_kind)

    def enqueue(self, source, *, title=None):
        # Accept work without loading neural models or waiting for an active import.
        from .embedding import load_default_config
        from .chunking import StructuredChunker, EmbeddingTokenizer
        from .parsers.docling import DoclingPaperParser, DoclingParserConfig, _atomic_copy, _sha256_file
        if self._closed or self._stop.is_set():
            raise RuntimeError('Paper services are closing')
        source = Path(source).resolve()
        if source.suffix.lower() != '.pdf' or not source.is_file():
            raise ValueError('Existing PDF required')
        cfg = load_default_config()
        model = Path(self.config.model_root) / cfg.model_id.split('/')[-1]
        chunker = StructuredChunker(EmbeddingTokenizer(model/'tokenizer.json',cfg.model_id))
        parser = DoclingPaperParser(DoclingParserConfig(Path(self.config.root)/'parsed'))
        digest = _sha256_file(source)
        target = Path(self.config.root).resolve()/'sources'/(digest+'.pdf')
        target.parent.mkdir(parents=True,exist_ok=True)
        _atomic_copy(source,target,digest)
        return self.catalog.enqueue_import(str(target),digest,parser_version=parser.parser_version,
            chunker_version=chunker.chunker_version,embedding_model=cfg.model_id,
            embedding_revision=cfg.revision,title=title or source.stem)

    def start_worker(self):
        with self._lock:
            self._load()
            if self._thread and self._thread.is_alive():
                return
            def work():
                import logging
                while not self._stop.is_set():
                    try:
                        with self._lock:
                            if self._stop.is_set():
                                break
                            self._worker.run_once()
                    except Exception:
                        logging.getLogger(__name__).exception('Paper import failed')
                    self._stop.wait(1)
            self._thread = threading.Thread(target=work, name='paper-import', daemon=False)
            self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        with self._lock:
            self._closed = True
            if self._es:
                self._es.close()
            self._es = self._embedding = self._search = self._worker = None
