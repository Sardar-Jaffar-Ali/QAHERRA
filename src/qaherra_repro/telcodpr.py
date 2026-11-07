"""Exact Telco-DPR document-and-top-section relevance matching."""

from __future__ import annotations

import re
from pathlib import PurePath


def normalise_document(value: object) -> str:
    """Match source documents despite paths, case, and optional DOCX suffixes."""
    text = str(value).replace("\\", "/").split("/")[-1].strip().lower()
    return re.sub(r"\.docx?$", "", text)


def gold_top_section(section: object) -> str:
    """Extract the Telco-DPR gold top-level section from its `|` hierarchy."""
    first = str(section).split("|", maxsplit=1)[0].strip()
    token = first.split(maxsplit=1)[0] if first else ""
    return token.rstrip(":").lower()


def retrieved_top_section(section_path: object) -> str:
    """Extract the evidence top-level section from its `>` hierarchy."""
    first = str(section_path).split(">", maxsplit=1)[0].strip()
    token = first.split(maxsplit=1)[0] if first else ""
    return token.rstrip(":").lower()


def is_relevant(
    retrieved_document: object,
    retrieved_section: object,
    gold_document: object,
    gold_section: object,
) -> bool:
    """Require both the normalized source document and top section to match."""
    return (
        normalise_document(retrieved_document) == normalise_document(gold_document)
        and retrieved_top_section(retrieved_section) == gold_top_section(gold_section)
    )

