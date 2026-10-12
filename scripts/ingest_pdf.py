"""Durable local PDF import; run worker separately from the chat process."""
import argparse
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mini_harness.rag.catalog import SQLitePaperCatalog
from mini_harness.rag.chunking import EmbeddingTokenizer, StructuredChunker
from mini_harness.rag.embedding import LocalE5Embedding, load_default_config
from mini_harness.rag.es_service import ESConfig, ESService
from mini_harness.rag.ingestion import IngestionWorker
from mini_harness.rag.parsers.docling import DoclingPaperParser, DoclingParserConfig
from mini_harness.rag.vector_index import EmbeddingProfile, ElasticsearchVectorIndex


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('action',choices=['import','enqueue','work','status','retry','cancel'])
    cli.add_argument('target',nargs='?')
    cli.add_argument('--catalog',type=Path,default=ROOT/'.mini-harness/rag/vector-catalog.sqlite3')
    cli.add_argument('--root',type=Path,default=ROOT/'.mini-harness/rag/ingestion')
    cli.add_argument('--once',action='store_true')
    cli.add_argument('--index',default=None)
    args = cli.parse_args()
    catalog = SQLitePaperCatalog(args.catalog)
    if args.action in ('status','retry','cancel'):
        if not args.target:
            cli.error('Job ID required')
        job = getattr(catalog,{'status':'get_job','retry':'retry_job','cancel':'cancel_job'}[args.action])(args.target)
        print(job.to_json() if job else 'null')
        return
    cfg = load_default_config()
    model = ROOT/'.mini-harness/rag/models'/cfg.model_id.split('/')[-1]
    embedding = LocalE5Embedding(cfg,model)
    parser = DoclingPaperParser(DoclingParserConfig(args.root/'parsed'))
    chunker = StructuredChunker(EmbeddingTokenizer(model/'tokenizer.json',cfg.model_id))
    es = ESService(ESConfig.load(ROOT/'.mini-harness/elasticsearch/connection.json'))
    try:
        index = ElasticsearchVectorIndex(es.client,catalog,EmbeddingProfile.from_metadata(embedding.metadata()),args.index,analyzer='english')
        worker = IngestionWorker(catalog,parser,chunker,embedding,index,args.root)
        if args.action in ('import','enqueue'):
            if not args.target:
                cli.error('PDF path required')
            job = worker.enqueue(args.target)
            print(job.to_json(),flush=True)
            if args.action == 'import':
                result = worker.run_once(job_id=job.job_id)
                print((result or catalog.get_job(job.job_id)).to_json(),flush=True)
        elif args.once:
            job = worker.run_once()
            print(job.to_json() if job else 'null',flush=True)
        else:
            worker.serve(threading.Event())
    finally:
        es.close()


if __name__ == '__main__':
    main()
