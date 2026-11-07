"""Exact deduplication rule applied to evidence units and fixed-size chunks.

The rule is deliberately minimal and is the same one the chunk collection was
built with, so that a comparison between the two representations measures the
representation and not the deduplication policy:

* two pieces are duplicates when the SHA-1 hashes of their *raw* searchable
  content are identical **and** they belong to the same specification series;
* the series is the first two digits of the specification number;
* no normalization of case, whitespace, or punctuation is applied;
* text and table units are treated alike;
* the first occurrence in corpus order is retained.

Non-standard document identifiers (those that do not match ``NNNNN-xxx``) are
deduplicated only against themselves, because their series cannot be read off
the identifier.

Removing a duplicate removes one occurrence's provenance from the retrievable
collection, so :func:`build_duplicate_map` also returns the mapping that links
every original position to the unit retained in its place. Publishing that
mapping is what keeps the specification, release, and section of every removed
occurrence recoverable.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np

STANDARD_DOC_ID = re.compile(r"^(\d{2})\d{3}-")


def series_of(doc_id: object) -> str:
    """Return the deduplication group of a document identifier.

    A standard 3GPP identifier such as ``38901-g10`` belongs to series ``38``.
    Anything else forms a group of its own, prefixed to keep it distinct from a
    numeric series.
    """
    text = str(doc_id)
    match = STANDARD_DOC_ID.match(text)
    return match.group(1) if match else f"DOC:{text}"


def content_key(doc_id: object, text: object) -> tuple[str, str]:
    """Return the ``(series, SHA-1)`` key that identifies duplicate content."""
    digest = hashlib.sha1(str(text).encode("utf-8")).hexdigest()
    return series_of(doc_id), digest


@dataclass
class DuplicateMap:
    """Result of deduplicating a collection in corpus order.

    Attributes
    ----------
    kept_positions:
        Original position of every retained unit, in retained order. Length is
        the size of the deduplicated collection.
    representative_of:
        For every original position, the index of the retained unit that
        represents it. Length is the size of the original collection. This is
        the mapping that keeps removed occurrences traceable.
    removed_by_type:
        Count of removed units per ``evidence_type``.
    original_by_type:
        Count of original units per ``evidence_type``.
    nonstandard_units:
        Number of units whose identifier does not match ``NNNNN-xxx``.
    """

    kept_positions: np.ndarray
    representative_of: np.ndarray
    removed_by_type: dict[str, int] = field(default_factory=dict)
    original_by_type: dict[str, int] = field(default_factory=dict)
    nonstandard_units: int = 0

    @property
    def original_units(self) -> int:
        return int(len(self.representative_of))

    @property
    def deduplicated_units(self) -> int:
        return int(len(self.kept_positions))

    @property
    def removed_units(self) -> int:
        return self.original_units - self.deduplicated_units

    @property
    def duplicate_fraction(self) -> float:
        """Share of the original collection that is an exact duplicate."""
        return self.removed_units / self.original_units if self.original_units else 0.0

    def sizes(self) -> dict[str, float]:
        """Return the figures reported in the paper as a flat mapping."""
        return {
            "original_units": self.original_units,
            "deduplicated_units": self.deduplicated_units,
            "removed_units": self.removed_units,
            "duplicate_percent": round(100 * self.duplicate_fraction, 2),
            "removed_text": self.removed_by_type.get("text", 0),
            "removed_table": self.removed_by_type.get("table", 0),
            "original_text": self.original_by_type.get("text", 0),
            "original_table": self.original_by_type.get("table", 0),
            "nonstandard_units": self.nonstandard_units,
        }


def build_duplicate_map(
    records: Iterable[tuple[object, object, object]],
    *,
    expected_units: int | None = None,
    on_kept=None,
) -> DuplicateMap:
    """Deduplicate a collection streamed in corpus order.

    Parameters
    ----------
    records:
        Iterable of ``(doc_id, evidence_type, text)`` in corpus order. Corpus
        order matters, because the first occurrence is the one retained.
    expected_units:
        Optional expected length, used to preallocate the mapping and to fail
        early when the input file is not the one the paper used.
    on_kept:
        Optional callback invoked as ``on_kept(retained_index, doc_id, text)``
        for every retained unit, so that a caller can shard the retained text
        to disk without holding the whole collection in memory.
    """
    first_seen: dict[tuple[str, str], int] = {}
    kept: list[int] = []
    representative = (
        np.empty(expected_units, dtype=np.int32) if expected_units else None
    )
    growing: list[int] = [] if representative is None else []
    removed: dict[str, int] = {}
    original: dict[str, int] = {}
    nonstandard = 0

    position = 0
    for doc_id, evidence_type, text in records:
        kind = str(evidence_type)
        original[kind] = original.get(kind, 0) + 1
        key = content_key(doc_id, text)
        if key[0].startswith("DOC:"):
            nonstandard += 1
        index = first_seen.get(key)
        if index is None:
            index = len(kept)
            first_seen[key] = index
            kept.append(position)
            if on_kept is not None:
                on_kept(index, doc_id, text)
        else:
            removed[kind] = removed.get(kind, 0) + 1
        if representative is not None:
            if position >= len(representative):
                raise ValueError(
                    f"input is longer than expected_units={expected_units}"
                )
            representative[position] = index
        else:
            growing.append(index)
        position += 1

    if expected_units is not None and position != expected_units:
        raise ValueError(
            f"expected {expected_units} units in corpus order, read {position}"
        )

    rep = (
        representative[:position]
        if representative is not None
        else np.asarray(growing, dtype=np.int32)
    )
    return DuplicateMap(
        kept_positions=np.asarray(kept, dtype=np.int32),
        representative_of=rep,
        removed_by_type=removed,
        original_by_type=original,
        nonstandard_units=nonstandard,
    )


def stream_evidence_csv(
    path: str,
    *,
    chunksize: int = 200_000,
    columns: tuple[str, ...] = ("doc_id", "evidence_type", "text"),
) -> Iterator[tuple[object, object, object]]:
    """Stream ``(doc_id, evidence_type, text)`` from the evidence CSV in order."""
    import pandas as pd

    for frame in pd.read_csv(path, chunksize=chunksize, usecols=list(columns)):
        for row in zip(*(frame[column].astype(str) for column in columns)):
            yield row


def duplicate_slots(positions: Iterable[int], representative_of: np.ndarray) -> int:
    """Count how many of a top-k list's slots are filled by repeated content.

    A list of ten units holding only six distinct contents has four duplicate
    slots. This is the quantity reported as 4.12 of ten slots on TeleQnA.
    """
    positions = list(positions)
    distinct = {int(representative_of[position]) for position in positions}
    return len(positions) - len(distinct)
