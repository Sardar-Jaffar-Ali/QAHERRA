#!/usr/bin/env python
"""Convert a local R8--R19 DOCX collection into QAHERRA evidence units."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterator

from docx import Document
from docx.document import Document as DocumentObject
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", required=True, help="Directory containing Rel-8 ... Rel-19")
    parser.add_argument("--output", required=True, help="Output evidence_units_full.csv")
    parser.add_argument("--chunk-size", type=int, default=1600)
    parser.add_argument("--overlap", type=int, default=250)
    return parser.parse_args()


def iter_blocks(document: DocumentObject) -> Iterator[Paragraph | Table]:
    """Yield paragraphs and tables in their original document order."""
    paragraph_lookup = {paragraph._element: paragraph for paragraph in document.paragraphs}
    table_lookup = {table._element: table for table in document.tables}
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P) and child in paragraph_lookup:
            yield paragraph_lookup[child]
        elif isinstance(child, CT_Tbl) and child in table_lookup:
            yield table_lookup[child]


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Produce character-bounded overlapping chunks without dropping text."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must exceed overlap")
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Prefer a nearby whitespace boundary while preserving the size cap.
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def flatten_table(table: Table) -> str:
    """Represent every non-empty table row independently from narrative text."""
    lines: list[str] = []
    for row in table.rows:
        cells = [re.sub(r"\s+", " ", cell.text).strip() for cell in row.cells]
        cells = [cell for cell in cells if cell]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def heading_level(paragraph: Paragraph) -> int | None:
    """Return the Word heading level or None for ordinary paragraphs."""
    style = paragraph.style.name if paragraph.style else ""
    match = re.match(r"Heading\s+(\d+)", style, flags=re.I)
    return int(match.group(1)) if match else None


def extract_document(path: Path, chunk_size: int, overlap: int) -> list[dict[str, str]]:
    """Create provenance-preserving text and table units for one specification."""
    document = Document(path)
    doc_id = path.stem
    headings: list[str] = []
    section_text: list[str] = []
    section_path = "Front Matter"
    units: list[dict[str, str]] = []

    def flush_text() -> None:
        joined = " ".join(section_text).strip()
        for chunk in split_text(joined, chunk_size, overlap):
            units.append(
                {
                    "doc_id": doc_id,
                    "evidence_type": "text",
                    "section_path": section_path,
                    "text": chunk,
                }
            )
        section_text.clear()

    for block in iter_blocks(document):
        if isinstance(block, Paragraph):
            text = re.sub(r"\s+", " ", block.text).strip()
            if not text:
                continue
            level = heading_level(block)
            if level is not None:
                flush_text()
                headings[:] = headings[: level - 1]
                headings.append(text)
                section_path = " > ".join(headings)
            else:
                section_text.append(text)
        else:
            flush_text()
            table_text = flatten_table(block)
            if table_text:
                units.append(
                    {
                        "doc_id": doc_id,
                        "evidence_type": "table",
                        "section_path": section_path,
                        "text": table_text,
                    }
                )
    flush_text()
    return units


def main() -> None:
    args = parse_args()
    root = Path(args.corpus_root)
    documents = sorted(
        path
        for release in range(8, 20)
        for path in (root / f"Rel-{release}").rglob("*.docx")
    )
    if not documents:
        raise FileNotFoundError(f"No DOCX specifications found under {root}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    failures = output.with_suffix(".failures.csv")

    with output.open("w", newline="", encoding="utf-8-sig") as handle, failures.open(
        "w", newline="", encoding="utf-8"
    ) as failure_handle:
        writer = csv.DictWriter(
            handle, fieldnames=["doc_id", "evidence_type", "section_path", "text"]
        )
        writer.writeheader()
        failure_writer = csv.DictWriter(failure_handle, fieldnames=["path", "error"])
        failure_writer.writeheader()

        total = 0
        for path in tqdm(documents, desc="Specifications"):
            try:
                rows = extract_document(path, args.chunk_size, args.overlap)
                writer.writerows(rows)
                handle.flush()
                total += len(rows)
            except Exception as error:
                failure_writer.writerow({"path": str(path), "error": repr(error)})
                failure_handle.flush()

    print(f"Documents processed: {len(documents):,}")
    print(f"Evidence units written: {total:,}")
    print("Evidence CSV:", output)
    print("Failures:", failures)


if __name__ == "__main__":
    main()

