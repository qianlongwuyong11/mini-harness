"""Prepare a new immutable staging artifact version for the selected tokenizer/model."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chunking_eval import atomic_write
from embedding_eval import CHUNKS
from mini_harness.rag.catalog import _decode
from mini_harness.rag.chunking import EmbeddingTokenizer, StructuredChunker
from mini_harness.rag.embedding import load_default_config
from mini_harness.rag.models import PaperVersion, SourceBlock, SourceKind


def main():
    config = load_default_config()
    model_path = ROOT / ".mini-harness/rag/models" / config.model_id.split("/")[-1]
    chunker = StructuredChunker(EmbeddingTokenizer(model_path / "tokenizer.json", config.model_id))
    source = ROOT / ".mini-harness/rag/parse-eval/1f75ba27b113731af6a73f52"
    originals = tuple(_decode(SourceBlock,json.dumps(b)) for b in json.loads((source / "source-blocks.json").read_text(encoding="utf-8")))
    parse = json.loads((source / "parse-report.json").read_text(encoding="utf-8"))
    identity = json.dumps([parse["source_sha256"],parse["parser_version"],chunker.chunker_version,config.model_id,config.revision])
    version = PaperVersion(originals[0].paper_id,hashlib.sha256(identity.encode()).hexdigest()[:24],
        parse["source_sha256"],parse["parser_version"],chunker.chunker_version,
        embedding_model=config.model_id,embedding_revision=config.revision)
    blocks = tuple(replace(b,version=version.version) for b in originals)
    chunks = chunker.split(version,blocks)
    papers = [c for c in chunks if c.source_kind is SourceKind.PAPER]
    previous = [c for c in json.loads(CHUNKS.read_text(encoding="utf-8")) if c["source_kind"]=="paper"]
    # Reuse benchmark vectors only if every actual model input remains byte-for-byte identical.
    assert [c.text for c in papers] == [c["text"] for c in previous]
    benchmark = ROOT / ".mini-harness/rag/embedding-eval/base"
    manifest = json.loads((benchmark / "manifest.json").read_text())
    assert manifest["embedding"]["revision"] == config.revision
    assert manifest["corpus_sha256"] == hashlib.sha256(CHUNKS.read_bytes()).hexdigest()
    assert manifest["chunk_ids"] == [c["chunk_id"] for c in previous]
    vectors = np.load(benchmark / "vectors.npy")
    assert vectors.shape == (len(papers),config.dimensions)
    target = ROOT / ".mini-harness/rag/selected" / version.version
    version = replace(version,chunk_count=len(papers))
    for name, value in (("source-blocks.json",[b.to_dict() for b in blocks]),("chunks.json",[c.to_dict() for c in chunks]),
                        ("version.json",version.to_dict())):
        atomic_write(target / name,json.dumps(value,ensure_ascii=False,indent=2))
    # JSON vector output shares the same durable writer and explicit chunk ID association.
    atomic_write(target / "embeddings.json",json.dumps({c.chunk_id:v.tolist() for c,v in zip(papers,vectors)}))
    selected = dict(version=version.to_dict(),embedding=manifest["embedding"],artifact=str(target),
                    indexed_source_kinds=["paper"],chunk_ids=[c.chunk_id for c in papers],status="staging_not_indexed")
    atomic_write(target / "manifest.json",json.dumps(selected,indent=2))
    atomic_write(ROOT / "docs/embedding-selected.json",json.dumps(selected,indent=2))
    print("Prepared",len(papers),"paper vectors and",len(chunks),"total chunks in",target)


if __name__ == "__main__":
    main()
