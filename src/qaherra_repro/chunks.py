"""Fixed-size chunk collection for the four-document scope.

The chunk text is read directly from the four-document index metadata. An
earlier version of this code resolved each chunk through the corpus-wide
``faiss_rows`` table in ``chunk_metadata.db`` using the metadata's ``row_id``.
That identifier is a *local* position within the four-document index (0-433),
not a corpus-wide row, so the join silently returned text from unrelated
specifications: the fetched content matched the expected hash for 0.0% of rows
and fell inside the four source documents for 0.5% of them.

Reading the text from the metadata file removes the lookup entirely.
``verify_chunk_metadata`` re-checks the property the bug violated, so a
regression cannot pass unnoticed: SHA-1 of every chunk's text must equal the
``text_hash`` recorded beside it.

See ``docs/CHUNK_LOOKUP_FIX.md`` for the diagnosis and the affected numbers.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass

import numpy as np


@dataclass
class ChunkCollection:
    """A fixed-size chunk collection with its source labels."""

    texts: list[str]
    doc_labels: list[str]
    sources: list[str]
    hashes: list[str]

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def distinct_sources(self) -> set[str]:
        return set(self.sources)


def _doc_label(source: object) -> str:
    """Return the header label for a chunk, matching the evidence-unit runs.

    The evidence-unit runs label each source with its specification identifier,
    so the chunk runs label each chunk with its file stem. Both appear in the
    prompt as ``[Source r | label]``, which keeps the prompt format identical
    across the two representations.
    """
    return os.path.basename(str(source)).replace(".docx", "")


def load_four_doc_chunks(metadata_pickle: str) -> ChunkCollection:
    """Load the four-document chunk collection from its index metadata."""
    frame = pickle.load(open(metadata_pickle, "rb")).reset_index(drop=True)
    return ChunkCollection(
        texts=frame.text.astype(str).tolist(),
        doc_labels=[_doc_label(source) for source in frame.source],
        sources=[str(source) for source in frame.source],
        hashes=[str(value) for value in frame.text_hash],
    )


def verify_chunk_metadata(
    collection: ChunkCollection,
    *,
    expected_size: int | None = 434,
    minimum_hash_match: float = 1.0,
) -> dict[str, float]:
    """Check that the loaded text is the text the index was built from.

    Raises
    ------
    AssertionError
        If the collection has an unexpected size, or if SHA-1 of the loaded
        text does not reproduce the recorded ``text_hash`` for every chunk.
        A hash match below 1.0 means the text no longer belongs to the index
        and no result computed from it is valid.
    """
    matches = [
        hashlib.sha1(text.encode("utf-8")).hexdigest() == digest
        for text, digest in zip(collection.texts, collection.hashes)
    ]
    rate = float(np.mean(matches)) if matches else 0.0
    if expected_size is not None:
        assert len(collection) == expected_size, (
            f"expected {expected_size} chunks, found {len(collection)}"
        )
    assert rate >= minimum_hash_match, (
        f"SHA-1 of chunk text matches the recorded hash for only {rate:.3f} of "
        "chunks; the text does not belong to this index"
    )
    return {
        "chunks": float(len(collection)),
        "hash_match": rate,
        "distinct_sources": float(len(collection.distinct_sources)),
    }


def cross_check_bm25_metadata(
    collection: ChunkCollection, bm25_metadata_pickle: str
) -> bool:
    """Confirm the dense and lexical indexes cover the same chunk set."""
    other = pickle.load(open(bm25_metadata_pickle, "rb")).reset_index(drop=True)
    return set(collection.hashes) == {str(value) for value in other.text_hash}


def build_context(
    collection: ChunkCollection, positions, max_characters: int = 14_000
) -> str:
    """Assemble the generator context, identical to the evidence-unit builder."""
    parts: list[str] = []
    total = 0
    for rank, position in enumerate(positions, start=1):
        block = (
            f"[Source {rank} | {collection.doc_labels[position]}]\n"
            f"{collection.texts[position]}"
        )
        if total + len(block) > max_characters:
            break
        parts.append(block)
        total += len(block) + 2
    return "\n\n---\n\n".join(parts)
