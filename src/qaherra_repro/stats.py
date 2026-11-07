"""Paired significance testing exactly as reported in the paper.

Every accuracy comparison is paired at the question level, so the tests act on
per-question outcomes rather than on aggregate accuracy:

* a two-sided exact McNemar test on the discordant pairs;
* a 95% confidence interval from 10,000 paired bootstrap resamples, taken as
  percentiles of the resampled mean difference;
* Holm correction applied *within* a declared family of related comparisons.

The families are fixed in advance and listed in :data:`HOLM_FAMILIES`. Holm's
correction depends on the family size, so a result is reproducible only when
the family is declared rather than chosen after the p-values are seen. The
helper :func:`holm` therefore takes the family size explicitly and refuses to
correct more comparisons than the family declares.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.stats import binomtest

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0

#: Declared Holm families, as stated in Section V-A of the paper.
HOLM_FAMILIES: dict[str, int] = {
    "retrieval_cutoffs": 6,
    "vs_qaherra_within_setting": None,  # size is the number of rows in that column
    "units_vs_chunks": 5,
    "reranking": 5,
    "option_conditioning": 3,
    "table_gate": 2,
}

#: The five members of the ``units_vs_chunks`` family, in reporting order.
UNITS_VS_CHUNKS_COMPARISONS = (
    "BM25, Q-small, four documents",
    "Dense, Q-small, four documents",
    "BM25, Q-small, full corpus",
    "BM25, TeleQnA, full corpus",
    "BM25, TeleQnA, full corpus, context-capped control",
)


@dataclass
class PairedResult:
    """Outcome of one paired comparison between two configurations."""

    name: str
    n: int
    accuracy_a: float
    accuracy_b: float
    a_only: int
    b_only: int
    difference: float
    ci_low: float
    ci_high: float
    p: float
    p_holm: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _as_bool_array(values) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype == object or array.dtype.kind in "US":
        array = np.char.lower(array.astype(str)) == "true"
    return array.astype(bool)


def paired_compare(name: str, outcomes_a, outcomes_b) -> PairedResult:
    """Compare two configurations on the same questions, in the same order.

    ``outcomes_a`` and ``outcomes_b`` are per-question correctness for the two
    configurations, aligned question by question. The difference is reported as
    ``a - b`` in percentage points.
    """
    a = _as_bool_array(outcomes_a)
    b = _as_bool_array(outcomes_b)
    if a.shape != b.shape:
        raise ValueError(
            f"{name}: {a.size} and {b.size} outcomes cannot be paired"
        )
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    discordant = a_only + b_only
    p = binomtest(a_only, discordant, 0.5).pvalue if discordant else 1.0

    difference = a.astype(int) - b.astype(int)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, difference.size, (BOOTSTRAP_RESAMPLES, difference.size))
    resampled = difference[draws].mean(axis=1) * 100
    low, high = np.percentile(resampled, [2.5, 97.5])

    return PairedResult(
        name=name,
        n=int(a.size),
        accuracy_a=float(100 * a.mean()),
        accuracy_b=float(100 * b.mean()),
        a_only=a_only,
        b_only=b_only,
        difference=float(100 * difference.mean()),
        ci_low=float(low),
        ci_high=float(high),
        p=float(p),
    )


def holm(p_values, family_size: int | None = None) -> list[float]:
    """Holm-corrected p-values within a family of a declared size.

    ``family_size`` defaults to the number of p-values supplied. Passing a
    larger declared size corrects as if the remaining members were tested,
    which is what keeps a partially reported family honest. Passing a smaller
    one is an error.
    """
    values = [float(value) for value in p_values]
    size = family_size if family_size is not None else len(values)
    if size < len(values):
        raise ValueError(
            f"family declares {size} comparisons but {len(values)} were supplied"
        )
    order = np.argsort(values)
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (size - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def apply_holm(results: list[PairedResult], family_size: int | None = None) -> list[PairedResult]:
    """Fill ``p_holm`` on a list of results treated as one family."""
    for result, adjusted in zip(results, holm([r.p for r in results], family_size)):
        result.p_holm = adjusted
    return results


def confirmation_criterion_met(
    chunk_minus_unit_ci_high: float, original_chunk_advantage: float = 2.0
) -> bool:
    """Evaluate the criterion fixed before the deduplication comparison.

    The criterion is met when the upper bound of the 95% confidence interval
    for the chunk-over-unit difference falls below the chunk advantage measured
    before deduplication. It is stated here so that the verdict is computed,
    not asserted.
    """
    return chunk_minus_unit_ci_high < original_chunk_advantage
