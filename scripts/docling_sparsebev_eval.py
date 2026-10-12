"""Reproducible Step 5 experiment for the user's SparseBEV PDF."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import psutil
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_harness.rag.models import Paper, PaperVersion, SourceKind
from mini_harness.rag.parsers.docling import DoclingPaperParser, DoclingParserConfig
from mini_harness.rag.parsers.docling import PlainTextPdfFallbackParser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def memory_sampler(stop: threading.Event, samples: list[int]) -> None:
    process = psutil.Process()
    while not stop.wait(0.1):
        try:
            total = process.memory_info().rss
            total += sum(
                child.memory_info().rss for child in process.children(recursive=True)
                if child.is_running()
            )
            samples.append(total)
        except (psutil.Error, OSError):
            pass


def render_pages(pdf_path: Path, output: Path, pages: tuple[int, ...]) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(pdf_path)
    rendered = []
    for page_number in pages:
        target = output / f"page-{page_number}.png"
        image = document[page_number - 1].render(scale=2).to_pil()
        image.save(target)
        rendered.append(str(target.resolve()))
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pdf", nargs="?", default=r"C:\Users\yjq\Desktop\Perception\SparseBEV.pdf"
    )
    parser.add_argument(
        "--artifacts", default=str(ROOT / ".mini-harness/rag/parse-eval")
    )
    parser.add_argument(
        "--report", default=str(ROOT / "docs/sparsebev-docling-report.json")
    )
    args = parser.parse_args()
    source = Path(args.pdf).resolve()
    digest = sha256_file(source)
    paper = Paper(
        paper_id="sparsebev-" + digest[:16],
        source_uri=str(source),
        source_hash=digest,
        title="SparseBEV",
    )
    docling_parser = DoclingPaperParser(
        DoclingParserConfig(artifacts_root=Path(args.artifacts))
    )
    version = PaperVersion(
        paper_id=paper.paper_id,
        version=docling_parser.parser_version + "-" + digest[:12],
        source_hash=digest,
        parser_version=docling_parser.parser_version,
        chunker_version="not-run-step-5",
    )
    process = psutil.Process()
    baseline = process.memory_info().rss
    samples = [baseline]
    stop = threading.Event()
    sampler = threading.Thread(target=memory_sampler, args=(stop, samples), daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        blocks = docling_parser.parse(paper, version)
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        sampler.join(timeout=2)
    artifacts = docling_parser.last_artifacts
    assert artifacts is not None
    paper_blocks = [b for b in blocks if b.source_kind is SourceKind.PAPER]
    annotation_blocks = [
        b for b in blocks if b.source_kind is SourceKind.USER_ANNOTATION
    ]
    baseline_started = time.perf_counter()
    baseline_blocks = PlainTextPdfFallbackParser().parse(paper, version)
    pypdf_seconds = time.perf_counter() - baseline_started
    # Persist a clearly labelled comparison, not a replacement of structured output.
    (artifacts.directory / "pypdf-baseline.json").write_text(
        json.dumps([b.to_dict() for b in baseline_blocks], ensure_ascii=False, indent=2),
        encoding="utf-8")
    assert all(b.page_spans for b in paper_blocks)
    assert any(b.kind.value == "table" for b in paper_blocks)
    assert any(b.kind.value == "formula" for b in paper_blocks)
    paper_text = "\n".join(b.text for b in paper_blocks)
    for annotation in annotation_blocks:
        if len(annotation.text) > 15 and not annotation.text.startswith("PDF "):
            assert annotation.text not in paper_text, "Annotation leaked into paper blocks"
    section_index = next(
        (index for index, block in enumerate(paper_blocks)
         if "Scale-adaptive Self Attention" in block.text),
        None,
    )
    neighborhood = []
    if section_index is not None:
        for block in paper_blocks[section_index : section_index + 14]:
            neighborhood.append(
                {
                    "kind": block.kind.value,
                    "pages": [[p.start, p.end] for p in block.page_spans],
                    "section_path": list(block.section_path),
                    "text": block.text[:500],
                    "warnings": block.extraction_warnings,
                }
            )
    artifact_sizes = {
        path.name: path.stat().st_size
        for path in (
            artifacts.source_pdf,
            artifacts.docling_json,
            artifacts.markdown,
            artifacts.source_blocks_json,
            artifacts.report_json,
        )
    }
    report = {
        "sample": str(source),
        "source_sha256": digest,
        "docling_version": "2.137.0",
        "wall_seconds": round(elapsed, 3),
        "baseline_rss_mib": round(baseline / 1024 / 1024, 2),
        "peak_process_tree_rss_mib": round(max(samples) / 1024 / 1024, 2),
        "peak_increment_mib": round((max(samples) - baseline) / 1024 / 1024, 2),
        "paper_blocks": len(paper_blocks),
        "annotation_blocks": len(annotation_blocks),
        "pypdf_baseline": {
            "seconds": round(pypdf_seconds, 3), "page_blocks": len(baseline_blocks),
            "characters": sum(len(b.text) for b in baseline_blocks),
            "limitation": PlainTextPdfFallbackParser.LIMITATION,
        },
        "annotation_summary": [
            {
                "page": block.page_spans[0].start,
                "subtype": block.section_path[-1],
                "text": block.text,
                "bbox_count": len(block.bboxes),
            }
            for block in annotation_blocks
        ],
        "section_3_2_found": section_index is not None,
        "section_3_2_neighborhood": neighborhood,
        "artifact_directory": str(artifacts.directory.resolve()),
        "artifact_sizes_bytes": artifact_sizes,
        "rendered_pages": render_pages(
            source, Path(args.artifacts) / "rendered", (3, 4, 6, 7)
        ),
    }
    parse_report = json.loads(artifacts.report_json.read_text(encoding="utf-8"))
    report["parse_report"] = parse_report
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rendered_report = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    sys.stdout.buffer.write(rendered_report.encode("utf-8"))


if __name__ == "__main__":
    main()
