"""Structure-first chunking. Tokenizer artifacts are supplied by the embedding adapter."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, Sequence

from .models import BlockKind, Chunk, PageSpan, PaperVersion, SourceBlock, content_hash, stable_chunk_id


class TokenCounter(Protocol):
    identity: str
    def count(self, text: str) -> int: ...
    def offsets(self, text: str) -> list[tuple[int, int]]: ...


class EmbeddingTokenizer:
    """Count the actual model input, including prefix and special tokens; never truncate."""
    def __init__(self, path: str | Path, model_id: str, prefix: str = "passage: "):
        from tokenizers import Tokenizer
        raw = Path(path).read_bytes()
        self.tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.prefix = prefix
        self.identity = model_id + ":" + hashlib.sha256(raw).hexdigest() + ":" + prefix

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(self.prefix + text).ids)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return self.tokenizer.encode(text, add_special_tokens=False).offsets


@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 384

    def __post_init__(self):
        if self.max_tokens < 16:
            raise ValueError("max_tokens must be at least 16")


class StructuredChunker:
    def __init__(self, tokenizer: TokenCounter, config: ChunkingConfig = ChunkingConfig()):
        self.tokenizer, self.config = tokenizer, config
        identity = json.dumps(["structure-v1", config.max_tokens, tokenizer.identity])
        self.chunker_version = "structure-v1-" + hashlib.sha256(identity.encode()).hexdigest()[:16]

    def _pieces(self, block: SourceBlock) -> list[str]:
        text = block.text.strip()
        if self.tokenizer.count(text) <= self.config.max_tokens:
            return [text]
        if block.kind is BlockKind.TABLE:
            lines = text.splitlines()
            start = next((i for i, line in enumerate(lines) if line.lstrip().startswith("|")), len(lines))
            if len(lines) < start + 3 or not lines[start+1].lstrip().startswith("|"):
                raise ValueError("Oversized table requires Markdown header and complete rows")
            header = "\n".join(lines[:start+2])
            pieces, rows = [], []
            for row in lines[start+2:]:
                if self.tokenizer.count(header + "\n" + row) > self.config.max_tokens:
                    raise ValueError("A table row exceeds the token budget; manual handling required")
                if rows and self.tokenizer.count("\n".join([header, *rows, row])) > self.config.max_tokens:
                    pieces.append("\n".join([header, *rows]))
                    rows = []
                rows.append(row)
            if rows:
                pieces.append("\n".join([header, *rows]))
            return pieces
        if block.kind in (BlockKind.FORMULA, BlockKind.HEADING):
            raise ValueError("Atomic formula or heading exceeds token budget")
        pieces = []
        while text:
            if self.tokenizer.count(text) <= self.config.max_tokens:
                pieces.append(text)
                break
            offsets = self.tokenizer.offsets(text)
            end = offsets[min(len(offsets), self.config.max_tokens) - 1][1]
            while end > 0 and self.tokenizer.count(text[:end]) > self.config.max_tokens:
                end //= 2
            if end == 0:
                raise ValueError("Tokenizer cannot fit text within token budget")
            # Slice the original string, not decoded token IDs: preserve Unicode and symbols.
            pieces.append(text[:end])
            text = text[end:]
        return pieces

    def split(self, version: PaperVersion, blocks: Sequence[SourceBlock]) -> tuple[Chunk, ...]:
        if version.chunker_version != self.chunker_version:
            raise ValueError("PaperVersion.chunker_version does not match chunker configuration")
        if len({b.block_id for b in blocks}) != len(blocks):
            raise ValueError("Duplicate source block IDs")
        chunks, pending, texts = [], [], []

        def flush():
            if not pending:
                return
            text = "\n\n".join(texts)
            pages = sorted({p for b in pending for s in b.page_spans for p in range(s.start, s.end + 1)})
            spans = []
            for p in pages:
                if spans and spans[-1].end + 1 == p:
                    spans[-1] = PageSpan(spans[-1].start, p)
                else:
                    spans.append(PageSpan(p, p))
            ordinal = len(chunks)
            chunks.append(Chunk(
                chunk_id=stable_chunk_id(version.paper_id, version.version, self.chunker_version, ordinal),
                paper_id=version.paper_id, version=version.version, ordinal=ordinal,
                text=text, content_hash=content_hash(text), token_count=self.tokenizer.count(text),
                source_kind=pending[0].source_kind, section_path=pending[0].section_path,
                page_spans=tuple(spans), source_block_ids=tuple(dict.fromkeys(b.block_id for b in pending)),
                bboxes=tuple(dict.fromkeys(box for b in pending for box in b.bboxes)),
            ))
            pending.clear()
            texts.clear()

        for block in blocks:
            if (block.paper_id, block.version) != (version.paper_id, version.version):
                raise ValueError("Source block belongs to a different paper/version")
            isolated = block.kind in (BlockKind.TABLE, BlockKind.FIGURE_CAPTION, BlockKind.ANNOTATION, BlockKind.HEADING)
            if pending and (isolated or block.section_path != pending[0].section_path
                            or block.source_kind != pending[0].source_kind):
                flush()
            pieces = self._pieces(block)
            if len(pieces) > 1:
                flush()
            for piece in pieces:
                if pending and self.tokenizer.count("\n\n".join([*texts, piece])) > self.config.max_tokens:
                    flush()
                pending.append(block)
                texts.append(piece)
                if isolated or len(pieces) > 1:
                    flush()
        flush()
        # Adjacency cannot silently cross from author evidence to user annotations.
        return tuple(replace(c,
            previous_chunk_id=chunks[i-1].chunk_id if i and chunks[i-1].source_kind == c.source_kind else None,
            next_chunk_id=chunks[i+1].chunk_id if i+1 < len(chunks) and chunks[i+1].source_kind == c.source_kind else None,
        ) for i, c in enumerate(chunks))
