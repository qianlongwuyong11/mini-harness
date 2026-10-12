from dataclasses import replace
import unittest

from mini_harness.rag.chunking import ChunkingConfig, StructuredChunker
from mini_harness.rag.models import BlockKind, PageSpan, PaperVersion, SourceBlock, SourceKind


class CharacterTokenizer:
    # Deliberately simple injected test double; production uses the model tokenizer.
    identity = "test-only"
    def count(self, text): return len(text) + 2
    def offsets(self, text): return [(i, i+1) for i in range(len(text))]


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.chunker = StructuredChunker(CharacterTokenizer(), ChunkingConfig(64))
        self.version = PaperVersion("p", "v", "hash", "parser", self.chunker.chunker_version)

    def block(self, name, text, kind=BlockKind.TEXT, page=1, section=("Method",), source=SourceKind.PAPER):
        return SourceBlock(name, "p", "v", text, kind, source, section, (PageSpan(page, page),))

    def test_cross_page_formula_context_and_repeatability(self):
        blocks = [self.block("a", "Definition"), self.block("b", "x+y", BlockKind.FORMULA, 2),
                  self.block("c", "Explanation", page=2)]
        result = self.chunker.split(self.version, blocks)
        self.assertEqual(result, self.chunker.split(self.version, blocks))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].source_block_ids, ("a", "b", "c"))
        self.assertEqual(result[0].page_spans, (PageSpan(1, 2),))

    def test_unicode_split_preserves_all_content(self):
        text = "中α🙂 mathematical text " * 20
        result = self.chunker.split(self.version, [self.block("a", text)])
        self.assertEqual("".join(c.text for c in result), text.strip())
        self.assertTrue(all(c.token_count <= 64 for c in result))

    def test_table_rows_and_headers(self):
        header = "| A | B |\n|---|---|"
        rows = [f"| {i} | value |" for i in range(12)]
        result = self.chunker.split(self.version, [self.block("table", header + "\n" + "\n".join(rows), BlockKind.TABLE)])
        self.assertEqual([row for c in result for row in c.text.splitlines()[2:]], rows)
        self.assertTrue(all(c.text.startswith(header) and c.token_count <= 64 for c in result))

    def test_source_and_section_boundaries(self):
        result = self.chunker.split(self.version, [self.block("a", "text"),
            self.block("b", "other", section=("Results",)),
            self.block("c", "note", source=SourceKind.USER_ANNOTATION)])
        self.assertEqual(len(result), 3)
        self.assertIsNone(result[1].next_chunk_id)
        self.assertIsNone(result[2].previous_chunk_id)
        self.assertEqual(result[0].next_chunk_id, result[1].chunk_id)

    def test_identity_mismatch_and_oversize_atomic_blocks(self):
        with self.assertRaises(ValueError):
            self.chunker.split(replace(self.version, chunker_version="other"), [])
        with self.assertRaises(ValueError):
            self.chunker.split(self.version, [self.block("a", "x" * 100, BlockKind.FORMULA)])
        other = StructuredChunker(CharacterTokenizer(), ChunkingConfig(128))
        self.assertNotEqual(other.chunker_version, self.chunker.chunker_version)
