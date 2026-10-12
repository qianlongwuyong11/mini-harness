"""Docling PDF adapter with durable and inspectable parsing artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter

from ..models import (
    BlockKind, BoundingBox, CoordinateSpace, PageSpan, Paper, PaperVersion,
    SourceBlock, SourceKind,
)


@dataclass(frozen=True)
class DoclingParserConfig:
    artifacts_root: Path
    device: str = "cpu"
    num_threads: int = 4
    do_ocr: bool = False
    do_table_structure: bool = True
    do_formula_enrichment: bool = False

    def __post_init__(self) -> None:
        if self.do_formula_enrichment:
            raise ValueError("Formula enrichment is deferred; use ordinary parsing")
        if self.device not in ("cpu", "auto", "cuda"):
            raise ValueError("device must be cpu, auto or cuda")
        if self.num_threads < 1:
            raise ValueError("num_threads must be positive")


@dataclass(frozen=True)
class ParseArtifacts:
    directory: Path
    source_pdf: Path
    docling_json: Path
    markdown: Path
    source_blocks_json: Path
    report_json: Path


class DoclingParseError(RuntimeError):
    """Conversion failed or produced no source-located paper blocks."""


class DoclingPaperParser:
    PARSER_VERSION = "docling-2.137.0-adapter-v3"

    def __init__(self, config: DoclingParserConfig) -> None:
        self.config = config
        self.last_artifacts: ParseArtifacts | None = None

    @property
    def parser_version(self) -> str:
        options = {key: value for key, value in vars(self.config).items()
                   if key != "artifacts_root"}
        digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()[:12]
        return f"{self.PARSER_VERSION}-{digest}"

    def parse(self, paper: Paper, version: PaperVersion) -> tuple[SourceBlock, ...]:
        self.last_artifacts = None
        if version.parser_version != self.parser_version:
            raise ValueError("Register a version matching this parser configuration")
        if version.paper_id != paper.paper_id or version.source_hash != paper.source_hash:
            raise ValueError("Paper and version identity mismatch")
        source = Path(paper.source_uri).resolve()
        if source.suffix.lower() != ".pdf" or not source.is_file():
            raise ValueError("DoclingPaperParser requires an existing local PDF")
        actual_hash = _sha256_file(source)
        if actual_hash.lower() != paper.source_hash.lower():
            raise ValueError("Source PDF SHA-256 differs from catalog identity")

        from docling.datamodel.accelerator_options import AcceleratorDevice
        from docling.datamodel.base_models import ConversionStatus, InputFormat
        from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline = PdfPipelineOptions(
            accelerator_options=AcceleratorOptions(
                num_threads=self.config.num_threads,
                device=AcceleratorDevice(self.config.device),
            ),
            do_ocr=self.config.do_ocr,
            do_table_structure=self.config.do_table_structure,
            do_formula_enrichment=self.config.do_formula_enrichment,
            do_picture_description=False,
            do_chart_extraction=False,
            generate_page_images=False,
            generate_picture_images=False,
        )
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)},
        )
        with tempfile.TemporaryDirectory(prefix="mini-harness-docling-") as temporary:
            clean_source = Path(temporary) / source.name
            _without_annotations(source, clean_source)
            result = converter.convert(clean_source, raises_on_error=False)
        if result.status is not ConversionStatus.SUCCESS:
            errors = [str(item) for item in result.errors]
            raise DoclingParseError(f"Docling status={result.status.value}: {errors}")

        blocks, conversion_report = _to_source_blocks(
            result.document, paper.paper_id, version.version
        )
        annotations = _annotation_blocks(
            source, paper.paper_id, version.version, len(blocks)
        )
        blocks = blocks + annotations
        if not any(block.source_kind is SourceKind.PAPER for block in blocks):
            raise DoclingParseError("Docling produced no source-located paper blocks")

        identity = hashlib.sha256(
            f"{paper.paper_id}\x1f{version.version}".encode("utf-8")
        ).hexdigest()[:24]
        directory = self.config.artifacts_root.resolve() / identity
        directory.mkdir(parents=True, exist_ok=True)
        artifacts = ParseArtifacts(
            directory=directory,
            source_pdf=directory / "source.pdf",
            docling_json=directory / "docling.json",
            markdown=directory / "document.md",
            source_blocks_json=directory / "source-blocks.json",
            report_json=directory / "parse-report.json",
        )
        _atomic_copy(source, artifacts.source_pdf, actual_hash)
        _atomic_text(
            artifacts.docling_json,
            json.dumps(result.document.export_to_dict(), ensure_ascii=False, indent=2),
        )
        _atomic_text(artifacts.markdown, result.document.export_to_markdown())
        _atomic_text(
            artifacts.source_blocks_json,
            json.dumps(
                [block.to_dict() for block in blocks], ensure_ascii=False, indent=2
            ),
        )
        report = {
            "parser_version": self.parser_version,
            "bbox_convention": "PDF_POINTS: bottom-left origin; y increases upward",
            "annotation_isolation": "All annotation objects removed from temporary parser input; original preserved",
            "source_sha256": actual_hash,
            "docling_status": result.status.value,
            "docling_errors": [str(item) for item in result.errors],
            "config": {
                "device": self.config.device,
                "num_threads": self.config.num_threads,
                "do_ocr": self.config.do_ocr,
                "do_table_structure": self.config.do_table_structure,
                "do_formula_enrichment": self.config.do_formula_enrichment,
            },
            "paper_block_count": sum(
                b.source_kind is SourceKind.PAPER for b in blocks
            ),
            "annotation_block_count": len(annotations),
            **conversion_report,
        }
        _atomic_text(
            artifacts.report_json, json.dumps(report, ensure_ascii=False, indent=2)
        )
        self.last_artifacts = artifacts
        return blocks


class PlainTextPdfFallbackParser:
    """Last-resort page text; structure, tables, formulas and columns are unverified."""

    PARSER_VERSION = "pypdf-plain-text-fallback-v1"
    LIMITATION = (
        "Layout unverified: headings, columns, tables, formulas and figures are not "
        "structurally parsed. Do not treat these blocks as complex-layout evidence."
    )

    def parse(self, paper: Paper, version: PaperVersion) -> tuple[SourceBlock, ...]:
        source = Path(paper.source_uri).resolve()
        if version.paper_id != paper.paper_id or version.source_hash != paper.source_hash:
            raise ValueError("Paper and version identity mismatch")
        if _sha256_file(source).lower() != paper.source_hash.lower():
            raise ValueError("Source PDF SHA-256 differs from catalog identity")
        blocks = []
        for page_number, page in enumerate(PdfReader(source).pages, 1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            blocks.append(
                SourceBlock(
                    block_id=_stable_block_id(
                        paper.paper_id,
                        version.version,
                        "pypdf-fallback",
                        str(page_number),
                        len(blocks),
                    ),
                    paper_id=paper.paper_id,
                    version=version.version,
                    text=text,
                    kind=BlockKind.TEXT,
                    source_kind=SourceKind.PAPER,
                    section_path=("UNVERIFIED_FALLBACK_TEXT",),
                    page_spans=(PageSpan(page_number, page_number),),
                    extraction_warnings=(self.LIMITATION,),
                )
            )
        if not blocks:
            raise DoclingParseError("Fallback parser found no selectable PDF text")
        return tuple(blocks)


def _to_source_blocks(
    document: Any, paper_id: str, version: str
) -> tuple[tuple[SourceBlock, ...], dict[str, Any]]:
    from docling_core.types.doc import (
        FormulaItem, PictureItem, SectionHeaderItem, TableItem, TextItem, TitleItem,
    )

    blocks: list[SourceBlock] = []
    section_path: list[str] = []
    skipped_without_location = 0
    kind_counts: dict[str, int] = {}
    page_heights = {number: page.size.height for number, page in document.pages.items()}
    for item, _level in document.iterate_items(with_groups=False):
        kind: BlockKind | None = None
        text = ""
        warnings: tuple[str, ...] = ()
        if isinstance(item, TableItem):
            kind = BlockKind.TABLE
            text = item.export_to_markdown(document).strip()
        elif isinstance(item, PictureItem):
            # Captions are separate TextItems with their own provenance. Mapping
            # the picture as a second caption duplicates text and assigns wrong bbox.
            continue
        elif isinstance(item, FormulaItem):
            kind = BlockKind.FORMULA
            text = (item.text or item.orig or "[Formula: inspect original PDF]").strip()
            warnings = ("Formula transcription is unverified; inspect original PDF for exact mathematics",)
        elif isinstance(item, SectionHeaderItem):
            kind = BlockKind.HEADING
            text = item.text.strip()
            level = max(1, int(item.level))
            number = re.match(r"^(\d+(?:\.\d+)*)\.?\s", text)
            if number:
                level = len(number.group(1).split("."))
            section_path[:] = section_path[: level - 1]
            section_path.append(text)
        elif isinstance(item, TitleItem):
            kind = BlockKind.HEADING
            text = item.text.strip()
        elif isinstance(item, TextItem):
            text = item.text.strip()
            label = str(getattr(item.label, "value", item.label)).lower()
            kind = (
                BlockKind.FIGURE_CAPTION if "caption" in label else BlockKind.TEXT
            )
        if kind is None or not text:
            continue
        spans, boxes = _locations(getattr(item, "prov", ()), page_heights)
        if not spans:
            skipped_without_location += 1
            continue
        ordinal = len(blocks)
        source_ref = str(getattr(item, "self_ref", ordinal))
        block_id = _stable_block_id(
            paper_id, version, "docling", source_ref, ordinal
        )
        blocks.append(
            SourceBlock(
                block_id=block_id,
                paper_id=paper_id,
                version=version,
                text=text,
                kind=kind,
                source_kind=SourceKind.PAPER,
                section_path=tuple(section_path),
                page_spans=spans,
                bboxes=boxes,
                extraction_warnings=warnings,
            )
        )
        kind_counts[kind.value] = kind_counts.get(kind.value, 0) + 1
    return tuple(blocks), {
        "kind_counts": kind_counts,
        "skipped_textual_items_without_location": skipped_without_location,
    }


def _locations(
    provenance: Any,
    page_heights: dict[int, float] | None = None,
) -> tuple[tuple[PageSpan, ...], tuple[BoundingBox, ...]]:
    pages: list[int] = []
    boxes: list[BoundingBox] = []
    for prov in provenance:
        page = int(prov.page_no)
        pages.append(page)
        bbox = prov.bbox
        y0, y1 = sorted((float(bbox.t), float(bbox.b)))
        origin = str(getattr(getattr(bbox, "coord_origin", "BOTTOMLEFT"), "value",
                             getattr(bbox, "coord_origin", "BOTTOMLEFT")))
        if origin == "TOPLEFT":
            if page_heights is None or page not in page_heights:
                raise ValueError("Page height required for top-left coordinate conversion")
            y0, y1 = page_heights[page] - y1, page_heights[page] - y0
        boxes.append(
            BoundingBox(
                page=page,
                x0=min(float(bbox.l), float(bbox.r)),
                y0=y0,
                x1=max(float(bbox.l), float(bbox.r)),
                y1=y1,
                coordinate_space=CoordinateSpace.PDF_POINTS,
            )
        )
    return _page_spans(pages), tuple(boxes)


def _page_spans(pages: list[int]) -> tuple[PageSpan, ...]:
    unique = sorted(set(pages))
    if not unique:
        return ()
    spans: list[PageSpan] = []
    start = end = unique[0]
    for page in unique[1:]:
        if page == end + 1:
            end = page
        else:
            spans.append(PageSpan(start, end))
            start = end = page
    spans.append(PageSpan(start, end))
    return tuple(spans)


def _annotation_blocks(
    source: Path, paper_id: str, version: str, offset: int
) -> tuple[SourceBlock, ...]:
    user_annotation_subtypes = {
        "Text", "FreeText", "Ink", "Highlight", "Underline", "StrikeOut",
        "Squiggly", "Stamp", "Caret",
    }
    blocks: list[SourceBlock] = []
    reader = PdfReader(source)
    for page_number, page in enumerate(reader.pages, 1):
        for annotation_index, reference in enumerate(page.get("/Annots", ())):
            annotation = reference.get_object()
            subtype = str(annotation.get("/Subtype", "/Unknown")).lstrip("/")
            if subtype not in user_annotation_subtypes:
                # Link annotations are part of ordinary PDF navigation, not user notes.
                continue
            contents = str(annotation.get("/Contents", "")).strip()
            text = contents or f"PDF {subtype} annotation (no semantic text)"
            boxes: tuple[BoundingBox, ...] = ()
            rect = annotation.get("/Rect")
            if rect and len(rect) == 4:
                x0, y0, x1, y1 = map(float, rect)
                if x0 != x1 and y0 != y1:
                    boxes = (
                        BoundingBox(
                            page=page_number,
                            x0=min(x0, x1),
                            y0=min(y0, y1),
                            x1=max(x0, x1),
                            y1=max(y0, y1),
                            coordinate_space=CoordinateSpace.PDF_POINTS,
                        ),
                    )
            ordinal = offset + len(blocks)
            blocks.append(
                SourceBlock(
                    block_id=_stable_block_id(
                        paper_id,
                        version,
                        "annotation",
                        f"{page_number}:{annotation_index}",
                        ordinal,
                    ),
                    paper_id=paper_id,
                    version=version,
                    text=text,
                    kind=BlockKind.ANNOTATION,
                    source_kind=SourceKind.USER_ANNOTATION,
                    section_path=("PDF annotation", subtype),
                    page_spans=(PageSpan(page_number, page_number),),
                    bboxes=boxes,
                    extraction_warnings=("Non-text annotation; meaning not inferred",) if not contents else (),
                )
            )
    return tuple(blocks)


def _stable_block_id(
    paper_id: str, version: str, source: str, item_ref: str, ordinal: int
) -> str:
    raw = "\x1f".join((paper_id, version, source, item_ref, str(ordinal)))
    return "block-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _without_annotations(source: Path, target: Path) -> None:
    """Render/infer on a temporary copy without notes; never edit the source."""
    writer = PdfWriter()
    writer.clone_document_from_reader(PdfReader(source))
    writer.remove_annotations(subtypes=None)
    with target.open("wb") as stream:
        writer.write(stream)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, target: Path, expected_hash: str) -> None:
    if target.exists() and _sha256_file(target) == expected_hash:
        return
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
        temp_path = Path(temporary.name)
        with source.open("rb") as stream:
            shutil.copyfileobj(stream, temporary)
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        if _sha256_file(temp_path) != expected_hash:
            raise OSError("Copied PDF checksum mismatch")
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_text(target: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        delete=False,
        newline="\n",
    ) as temporary:
        temp_path = Path(temporary.name)
        temporary.write(text)
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
