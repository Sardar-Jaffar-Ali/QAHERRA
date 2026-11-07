# Reproducing each result in the paper

Every claim is listed with the script that produces it, the file the number is
computed from, and the value to expect. Run
`python scripts/verify_results.py --paths configs/paths.yaml` to check all of
them at once without an API key.

## Order of operations

```
scripts/prepare_evidence.py        evidence units from the specifications
scripts/build_indexes.py           BM25 and FAISS indexes
scripts/run_retrieval_eval.py      Figure 2, Table IV
scripts/run_qa.py                  Table III, evidence-unit rows
scripts/run_chunk_baseline_4doc.py Table III, four-document chunk rows
scripts/dedup_stage_a_build.py     corpus sizes, the deduplication mapping
scripts/dedup_stage_b_retrieval.py retrieval after deduplication
scripts/dedup_stage_c_prompts.py   context lengths, duplicate slots
scripts/dedup_stage_d_qa.py        Section V-D deduplication result
scripts/train_table_gate.py        Section V-E3 gate comparison
scripts/verify_results.py          recompute and check everything
scripts/make_figure3.py            Figure 3
```

## Table II, implementation settings

Every value is in `configs/qaherra.yaml`. The encoder adaptation
hyperparameters are in the `dense` block; the fusion constant, weights, and
candidate count in `retrieval`; the cross-encoder and final evidence count in
`reranking`; the generator, temperature, and context budget in `generation`.

## Figure 2 and Table IV, retrieval on 523 held-out Telco-DPR queries

| Claim | Expected | Source |
|---|---|---|
| BM25 R@20 | 50.10 | `results/retrieval_ablation.csv`, `run_retrieval_eval.py` |
| QAHERRA R@20 | 52.01 | same |
| Queries gained, lost | 10, 0 of 261 BM25 misses | per-query file |
| Exact McNemar p, Holm p | 0.0020, 0.012 | `run_retrieval_eval.py` |
| Adapted minus unadapted dense, R@1 and R@20 | +2.68, +1.34 | `results/retrieval_ablation.csv` |
| Fusion share of the R@20 gain with the unadapted encoder | about 60% | `(51.24 - 50.10) / (52.01 - 50.10)` |
| Table boost effect on R@10 and R@20 | none | `results/retrieval_ablation.csv` |

A unit counts as relevant only when it comes from the gold specification,
including its release, and the gold top-level section. The matching rule is
`src/qaherra_repro/telcodpr.py`, and a textually equivalent passage from
another release counts as a miss.

## Table III, end-to-end accuracy

Every reported cell has a per-question file; `results/qa_accuracy.csv` lists
them all with their source. The configurations that need the corrected chunk
runs are produced by `scripts/run_chunk_baseline_4doc.py`.

### Configurations without a per-question file

Two cells of an earlier draft are **not reported** in the paper, because no
per-question output survives for them: dense retrieval over fixed-size chunks
in the full-corpus Q-small and TeleQnA settings. They appear as `--` in
Table III. The same holds for the Telco-RAG adaptation, whose earlier aggregate
figures cannot be recomputed. `baselines/telco_rag/` retains the adaptation
code; the numbers are not claimed.

Nothing in the paper is computed from an aggregate figure alone.

## Section V-D, evidence representation and search scope

| Claim | Expected | Source |
|---|---|---|
| Units answer 27% more than chunks with BM25, four documents | 84 / 66 | `results/units_vs_chunks_tests.csv` |
| Units answer 23% more with dense retrieval | 74 / 60 | same |
| BM25 discordant counts, four documents | 22 units, 4 chunks | same |
| Dense discordant counts, four documents | 24 units, 10 chunks | same |
| Holm-corrected p, five-comparison family | 0.003, 0.073, 0.200, 0.063, 0.132 | `docs/STATISTICS.md` |
| TeleQnA reversal | 151 chunks, 111 units | `results/units_vs_chunks_tests.csv` |
| Duplicate share of the corpus | 69.1% | `results/corpus_sizes.csv` |
| Duplicate slots per top-10 | 4.12 of 10 | `dedup_stage_c_prompts.py` |
| Deduplicated accuracy | 75.3% | `dedup_stage_d_qa.py` |
| Deduplication gain | +1.7, CI 0.55 to 2.85, Holm p = 0.017 | `results/dedup_qa_tests.csv` |
| Against chunks after deduplication | indistinguishable, CI reaches 1.85 | same |
| Capped-chunk control | 75.15, Holm p = 0.132 | same |
| Context characters, units then deduplicated units | 10,465 then 11,095 | `results/context_lengths.csv` |
| Prompt tokens, deduplicated units against uncapped chunks | 2,906 against 3,485, about 17% fewer | same |

The 17% figure compares deduplicated units against the **uncapped** chunk run.
Comparing against the capped control instead reverses the direction, so the
comparison basis has to be stated with the number.

## Section V-E3, query-aware table boosting

| Claim | Expected | Source |
|---|---|---|
| Annotated questions, agreement | 300, Cohen's kappa 0.73 | `evaluation/TABLE_GATE_PROTOCOL.md` |
| Keyword gate recall, F1 | 13.33, 0.19 | `results/table_gate_validation.csv` |
| Learned gate recall, F1 | 80.00, 0.74 | same |
| Answers changed by removing the boost on TeleQnA | 3 of 2,000 | per-question files |

The learned gate is trained and measured on the same labelled set, so its score
is an optimistic upper bound and is reported as one.

## Section V-E4 and V-F, option conditioning and cost

`results/option_conditioning.csv` holds Table V. Retrieval latency is one value
per benchmark, shared by both configurations, because candidate discovery uses
the question only; the component latencies sum to the end-to-end figure on that
basis.

## Figure 3

```bash
python scripts/verify_results.py --paths configs/paths.yaml --write-results
python scripts/make_figure3.py --out fig/fig3_representation_scope.pdf
```

The figure reads `results/units_vs_chunks_tests.csv`, so its bars and its
significance brackets are the same numbers as Table III and `docs/STATISTICS.md`.
