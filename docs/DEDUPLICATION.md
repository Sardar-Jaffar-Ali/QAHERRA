# Deduplication and provenance

3GPP specifications are re-issued release after release, so the same
requirement exists in several versions at once. Any comparison between two ways
of segmenting such a corpus is therefore also a comparison between two
deduplication policies, unless the policy is stated and applied identically to
both. This note states the policy.

## The rule

Two pieces of evidence are duplicates when

1. the SHA-1 hashes of their **raw** searchable content are identical, and
2. they belong to the same **specification series**, taken as the first two
   digits of the specification number.

No normalization of case, whitespace, or punctuation is applied. Text units and
table units are treated alike. The first occurrence in corpus order is
retained. A document identifier that does not match `NNNNN-xxx` forms a group
of its own, because its series cannot be read off the identifier.

This is the rule the fixed-size chunk collection was built with. Applying it
unchanged to the evidence units is what makes the full-corpus comparison a
comparison of representations.

The implementation is `src/qaherra_repro/dedup.py`; `scripts/dedup_stage_a_build.py`
applies it to the full corpus.

## Effect on the corpus

| Quantity | Value |
|---|---:|
| Original evidence units | 2,396,136 |
| Removed as exact duplicates | 1,656,134 |
| Retained | 740,002 |
| Share removed | 69.1% |
| Mean duplicate slots per original top-10 | 4.12 of 10 |

The last row is the quantity that matters for answer quality: in an
unduplicated collection, ten retrieved slots carry ten distinct passages; here
they carry fewer than six.

## What deduplication does to provenance

Removing a duplicate removes one occurrence from the retrievable collection, so
a retained unit carries the specification, release, and section of the single
occurrence that was kept. The other occurrences are not lost, but they are no
longer visible at answer time.

`representative_of.npy` is the published mapping that keeps them recoverable.
It holds one entry per original unit, giving the index of the retained unit that
represents it, so every removed copy can be traced back to its own
specification, release, and section:

```python
import numpy as np, pandas as pd
representative_of = np.load("dedup_outputs/representative_of.npy")   # 2,396,136
kept = np.load("dedup_outputs/kept_positions.npy")                   #   740,002

# every original occurrence represented by retained unit 12345
originals = np.flatnonzero(representative_of == 12345)
```

Retrieval and generation never consult this mapping. An answer grounded in the
deduplicated collection is therefore traceable to one release of a repeated
passage, and to all of them only through the mapping. `scripts/dedup_stage_b_retrieval.py`
reports a `dedup_aware` variant that resolves relevance through the mapping,
which separates this provenance effect from a ranking effect.

## The confirmation criterion

Before the deduplication experiment was run, the criterion for confirming that
unequal deduplication explains the full-corpus result was fixed as follows: the
upper bound of the 95% confidence interval for the chunk-over-unit difference
must fall below the 2-point chunk advantage measured without deduplication.

The criterion is evaluated in code, not asserted, by
`qaherra_repro.stats.confirmation_criterion_met`. On the released outputs the
deduplicated units reach 75.3% against 73.6% without deduplication (+1.7
points, 95% CI 0.55 to 2.85, Holm-corrected p = 0.017), the chunk advantage
falls to 0.3 points with an interval reaching 1.85, and the criterion is met.

The criterion is met narrowly, and `docs/STATISTICS.md` states the limits of
that conclusion.
