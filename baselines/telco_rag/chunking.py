"""Character chunking used by the controlled Telco-RAG adaptation."""

from __future__ import annotations

import re


SEPARATORS = re.compile(r"[\s,.\-!?\[\](){}\":;<>]+")


def split_text(text: str, chunk_size: int = 1600, overlap: int = 250) -> list[str]:
    """Split text at a nearby separator while preserving fixed overlap."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            match = SEPARATORS.search(text, end)
            if match:
                end = match.end()
        if end <= start:
            end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks

