# Statistical reporting

Every accuracy comparison in the paper is paired at the question level. The
tests therefore act on per-question outcomes, which the released files contain,
and not on aggregate accuracy. The implementation is
`src/qaherra_repro/stats.py`.

## Tests

| Quantity | Method |
|---|---|
| Difference between two configurations | two-sided exact McNemar test on the discordant pairs |
| Interval for that difference | 10,000 paired bootstrap resamples, percentile interval, seed 0 |
| Multiple comparisons | Holm correction within a declared family |

A two-sided exact McNemar test is a binomial test on the discordant pairs
against p = 0.5, so a comparison with few disagreements has little power
however large the benchmark. On Q-small's 100 questions, a non-significant
difference reflects limited power rather than equivalence.

## Declared families

Holm's correction depends on the size of the family, so a corrected p-value is
reproducible only when the family is declared in advance rather than chosen
after the raw p-values are seen. The families are:

| Family | Size |
|---|---:|
| Retrieval cutoffs | 6 |
| All comparisons against QAHERRA within one QA setting | number of rows in that column |
| Evidence units versus fixed-size chunks | 5 |
| Reranking | 5 |
| Option conditioning | 3 |
| Table gate | 2 |

`qaherra_repro.stats.holm` takes the family size explicitly and refuses to
correct more comparisons than the family declares, so a partially reported
family cannot be corrected as if it were smaller.

## The family of five, in full

All five members are reported, so that the correction can be checked:

| Comparison | Units | Chunks | Units only | Chunks only | Raw p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| BM25, Q-small, four documents | 84.0 | 66.0 | 22 | 4 | 0.000534 | **0.003** |
| Dense, Q-small, four documents | 74.0 | 60.0 | 24 | 10 | 0.024307 | 0.073 |
| BM25, Q-small, full corpus | 74.0 | 66.0 | | | 0.200 | 0.200 |
| BM25, TeleQnA, full corpus | 73.6 | 75.6 | 111 | 151 | 0.0158 | 0.063 |
| BM25, TeleQnA, capped-context control | 73.6 | 75.15 | | | 0.0662 | 0.132 |

Only the first is significant after correction. The four-document dense gain,
the full-corpus Q-small gain, and the TeleQnA reversal are all in the direction
the aggregate accuracies suggest but do not survive correction, and the paper
describes them that way.

## Failed API calls

A failed generator call is not an incorrect answer. Every run script writes a
failed call to a separate error file, never to the result file, and retries only
the questions that have no successful record when it is run again. The
significance tests are computed only once every question in the benchmark has a
successful record; a partial run exits instead of producing a number.

`scripts/verify_results.py` re-checks this property on the released files and
reports the count of failed calls found in each.

## Reproducing the numbers

```bash
python scripts/verify_results.py --paths configs/paths.yaml
```

The script recomputes every accuracy, every paired test, and every
context-length figure from the per-question files and compares each with the
value printed in the paper. It exits non-zero on any disagreement.
