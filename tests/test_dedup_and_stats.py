"""Unit tests for the deduplication rule and the paired statistics.

These need no 3GPP data, no index, and no API key.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from qaherra_repro.chunks import ChunkCollection, build_context, verify_chunk_metadata
from qaherra_repro.dedup import (
    build_duplicate_map,
    content_key,
    duplicate_slots,
    series_of,
)
from qaherra_repro.stats import (
    confirmation_criterion_met,
    holm,
    paired_compare,
)


def test_series_is_the_first_two_digits_of_the_specification_number():
    assert series_of("38901-g10") == "38"
    assert series_of("36777-f00_1") == "36"
    assert series_of("not-a-spec") == "DOC:not-a-spec"


def test_content_key_uses_raw_text_without_normalisation():
    assert content_key("38901-g10", "A  b") != content_key("38901-g10", "A b")
    assert content_key("38901-g10", "A b") != content_key("38901-g10", "a b")


def test_duplicates_are_removed_only_within_the_same_series():
    records = [
        ("38901-g10", "text", "same"),   # kept, retained index 0
        ("38901-h10", "text", "same"),   # duplicate within series 38
        ("36777-f00", "text", "same"),   # kept, different series
        ("38901-g10", "table", "other"), # kept
    ]
    mapping = build_duplicate_map(records, expected_units=4)
    assert mapping.deduplicated_units == 3
    assert mapping.removed_units == 1
    assert mapping.removed_by_type == {"text": 1}
    assert mapping.kept_positions.tolist() == [0, 2, 3]
    assert mapping.representative_of.tolist() == [0, 0, 1, 2]


def test_the_first_occurrence_in_corpus_order_is_the_one_kept():
    records = [("38901-g10", "text", "x"), ("38901-g10", "text", "x")]
    mapping = build_duplicate_map(records)
    assert mapping.kept_positions.tolist() == [0]


def test_a_shorter_input_than_declared_is_rejected():
    with pytest.raises(ValueError):
        build_duplicate_map([("38901-g10", "text", "x")], expected_units=2)


def test_duplicate_slots_counts_repeated_content_in_a_top_k_list():
    representative_of = np.array([0, 0, 1, 2, 2], dtype=np.int32)
    assert duplicate_slots([0, 1, 2], representative_of) == 1
    assert duplicate_slots([0, 2, 3], representative_of) == 0
    assert duplicate_slots([0, 1, 3, 4], representative_of) == 2


def _collection(texts: list[str]) -> ChunkCollection:
    return ChunkCollection(
        texts=texts,
        doc_labels=[f"doc{i}" for i in range(len(texts))],
        sources=[f"rel/doc{i}.docx" for i in range(len(texts))],
        hashes=[hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts],
    )


def test_chunk_verification_accepts_text_that_matches_its_hash():
    checks = verify_chunk_metadata(_collection(["a", "b"]), expected_size=2)
    assert checks["hash_match"] == 1.0


def test_chunk_verification_rejects_text_that_does_not_match_its_hash():
    collection = _collection(["a", "b"])
    collection.texts[0] = "text from another specification"
    with pytest.raises(AssertionError):
        verify_chunk_metadata(collection, expected_size=2)


def test_context_stops_before_exceeding_the_budget():
    collection = _collection(["x" * 100, "y" * 100])
    context = build_context(collection, [0, 1], max_characters=150)
    assert "y" * 100 not in context
    assert context.count("[Source") == 1


def test_mcnemar_uses_only_the_discordant_pairs():
    result = paired_compare("t", [1, 1, 0, 1], [1, 0, 0, 0])
    assert result.a_only == 2
    assert result.b_only == 0
    assert result.p == pytest.approx(0.5, abs=1e-9)


def test_identical_configurations_give_no_evidence_of_a_difference():
    result = paired_compare("t", [1, 0, 1], [1, 0, 1])
    assert (result.a_only, result.b_only) == (0, 0)
    assert result.p == 1.0
    assert result.difference == 0.0


def test_holm_is_monotone_and_bounded():
    adjusted = holm([0.001, 0.02, 0.4])
    assert adjusted == sorted(adjusted)
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    assert adjusted[0] == pytest.approx(0.003)


def test_holm_corrects_against_the_declared_family_size():
    assert holm([0.001], family_size=5)[0] == pytest.approx(0.005)
    with pytest.raises(ValueError):
        holm([0.01, 0.02, 0.03], family_size=2)


def test_the_confirmation_criterion_compares_the_interval_with_the_original_gap():
    assert confirmation_criterion_met(1.85, 2.0)
    assert not confirmation_criterion_met(2.4, 2.0)
