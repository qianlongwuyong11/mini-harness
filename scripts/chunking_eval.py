"""Evaluate chunking from saved parser artifacts, without running PDF models."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mini_harness.rag.catalog import _decode
from mini_harness.rag.chunking import EmbeddingTokenizer, StructuredChunker
from mini_harness.rag.models import PaperVersion, SourceBlock


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("blocks", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / ".mini-harness/rag/tokenizers/multilingual-e5-small/tokenizer.json")
    parser.add_argument("--output", type=Path, default=ROOT / ".mini-harness/rag/chunk-eval")
    args = parser.parse_args()
    raw = args.blocks.read_bytes()
    originals = tuple(_decode(SourceBlock, json.dumps(b)) for b in json.loads(raw))
    report = json.loads(args.blocks.with_name("parse-report.json").read_text(encoding="utf-8"))
    tokenizer = EmbeddingTokenizer(args.tokenizer, "intfloat/multilingual-e5-small")
    chunker = StructuredChunker(tokenizer)
    version_id = hashlib.sha256(raw + chunker.chunker_version.encode()).hexdigest()[:24]
    version = PaperVersion(originals[0].paper_id, version_id, report["source_sha256"],
                           report["parser_version"], chunker.chunker_version)
    blocks = tuple(replace(b, version=version_id) for b in originals)
    chunks = chunker.split(version, blocks)
    assert chunks == chunker.split(version, blocks)
    assert {s for c in chunks for s in c.source_block_ids} == {b.block_id for b in blocks}
    assert all(c.token_count <= chunker.config.max_tokens and c.page_spans for c in chunks)
    lookup = {b.block_id: b for b in blocks}
    metadata = [{"chunk_id": c.chunk_id,
                 "block_kinds": list(dict.fromkeys(lookup[b].kind.value for b in c.source_block_ids)),
                 "extraction_warnings": list(dict.fromkeys(w for b in c.source_block_ids for w in lookup[b].extraction_warnings))}
                for c in chunks]
    directory = args.output / version_id
    atomic_write(directory / "source-blocks.json", json.dumps([b.to_dict() for b in blocks], ensure_ascii=False, indent=2))
    atomic_write(directory / "chunks.json", json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2))
    atomic_write(directory / "chunk-metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
    lines = ["# SparseBEV chunk review", "", "Candidate tokenizer only; embedding model selection is pending.", ""]
    for c, meta in zip(chunks, metadata):
        lines += [f"## {c.ordinal}: {c.chunk_id}",
                  f"Section: {' / '.join(c.section_path)}", "",
                  f"Pages: {[(s.start, s.end) for s in c.page_spans]}; tokens: {c.token_count}; source: {c.source_kind.value}; kinds: {meta['block_kinds']}",
                  f"Source blocks: {', '.join(c.source_block_ids)}", "",
                  f"Warnings: {meta['extraction_warnings']}", "", c.text, ""]
    atomic_write(directory / "chunks.md", "\n".join(lines))
    summary = dict(source_blocks=len(blocks), chunks=len(chunks), max_tokens=max(c.token_count for c in chunks),
                   chunker_version=chunker.chunker_version, tokenizer=tokenizer.identity,
                   version=version.to_dict(), artifacts=str(directory), repeatable=True, all_blocks_covered=True)
    atomic_write(ROOT / "docs/sparsebev-chunking-report.json", json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
