from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from pypdf import PdfWriter
from pypdf.annotations import FreeText

from mini_harness.rag.models import BlockKind, SourceKind
from mini_harness.rag.parsers.docling import (
    DoclingParserConfig,
    PlainTextPdfFallbackParser,
    _annotation_blocks,
    _locations,
    _page_spans,
)


class DoclingParserUnitTests(unittest.TestCase):
    def test_config_is_cpu_and_ocr_off_by_default(self) -> None:
        config = DoclingParserConfig(Path("artifacts"))
        self.assertEqual(config.device, "cpu")
        self.assertFalse(config.do_ocr)
        self.assertTrue(config.do_table_structure)
        self.assertFalse(config.do_formula_enrichment)

    def test_formula_enrichment_is_deferred(self) -> None:
        with self.assertRaisesRegex(ValueError, "Formula enrichment is deferred"):
            DoclingParserConfig(Path("artifacts"), do_formula_enrichment=True)

    def test_page_spans_and_bbox_normalization(self) -> None:
        self.assertEqual(
            [(span.start, span.end) for span in _page_spans([4, 3, 4, 7])],
            [(3, 4), (7, 7)],
        )
        prov = [SimpleNamespace(
            page_no=3,
            bbox=SimpleNamespace(l=100, t=300, r=50, b=200),
        )]
        spans, boxes = _locations(prov)
        self.assertEqual((spans[0].start, spans[0].end), (3, 3))
        self.assertEqual((boxes[0].x0, boxes[0].y0, boxes[0].x1, boxes[0].y1),
                         (50, 200, 100, 300))

    def test_pdf_annotations_are_isolated_from_paper_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotated.pdf"
            writer = PdfWriter()
            writer.add_blank_page(300, 400)
            writer.add_annotation(0, FreeText(text="用户批注", rect=(10, 20, 100, 60)))
            with path.open("wb") as stream:
                writer.write(stream)
            blocks = _annotation_blocks(path, "paper", "version", 0)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].source_kind, SourceKind.USER_ANNOTATION)
        self.assertEqual(blocks[0].kind, BlockKind.ANNOTATION)
        self.assertEqual(blocks[0].text, "用户批注")
        self.assertEqual(blocks[0].page_spans[0].start, 1)

    def test_fallback_is_explicitly_marked_unverified(self) -> None:
        self.assertIn("Layout unverified", PlainTextPdfFallbackParser.LIMITATION)
        self.assertEqual(
            PlainTextPdfFallbackParser.PARSER_VERSION,
            "pypdf-plain-text-fallback-v1",
        )


if __name__ == "__main__":
    unittest.main()
