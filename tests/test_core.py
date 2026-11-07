"""Small tests for logic that does not require proprietary standards data."""

import numpy as np

from qaherra_repro.generation import parse_answer
from qaherra_repro.retrieval import reciprocal_rank_fusion
from qaherra_repro.telcodpr import (
    gold_top_section,
    normalise_document,
    retrieved_top_section,
)


def test_answer_parser() -> None:
    assert parse_answer("B", list("ABCD")) == "B"
    assert parse_answer("Option C", list("ABCD")) == "C"
    assert parse_answer("unknown", list("ABCD")) is None


def test_weighted_rrf() -> None:
    indices, scores = reciprocal_rank_fusion(
        [[1, 2], [2, 3]], [1.0, 1.0], constant=5.0, limit=3
    )
    assert indices[0] == 2
    assert np.all(scores[:-1] >= scores[1:])


def test_telcodpr_normalisation() -> None:
    assert normalise_document("Rel-15/36777-f00_1.docx") == "36777-f00_1"
    assert gold_top_section("5 Performance | 5.1 Requirements") == "5"
    assert retrieved_top_section("5\tPerformance > 5.1\tRequirements") == "5"

