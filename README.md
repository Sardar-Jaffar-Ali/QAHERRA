# QAHERRA

Reproducibility package for *Scaling Evidence Retrieval for 3GPP Question
Answering: Provenance-Preserving Units, Hybrid Fusion, and Reranking*.

QAHERRA converts 3GPP Releases 8-19 into provenance-preserving evidence units,
retrieves candidates by fusing lexical and telecom-adapted dense rankings, and
reorders them with a cross-encoder. The paper evaluates each of those three
design choices separately, both in a search restricted to four source
documents and across the full corpus, and finds that the choice which helps
most in a focused search does not transfer to the full corpus.

## Start here

Every number in the paper is computed from per-question output files. The
fastest way to check the paper is to recompute them, which needs no API key, no
GPU, and no index:

```bash
pip install -r requirements.txt && pip install -e .
cp configs/paths.example.yaml configs/paths.yaml   # then edit the paths
python scripts/verify_results.py --paths configs/paths.yaml
```

The script recomputes every accuracy, every paired significance test, and every
context-length figure, compares each with the value printed in the paper, and
exits non-zero on any disagreement. `docs/REPRODUCING_THE_PAPER.md` maps each
claim to the script and file behind it.

## Headline results

| Comparison | Scope | Units | Chunks | Holm-corrected p |
|---|---|---:|---:|---:|
| BM25 over evidence units against fixed-size chunks | four documents | 84.0 | 66.0 | **0.003** |
| Dense retrieval, same comparison | four documents | 74.0 | 60.0 | 0.073 |
| BM25, same comparison | full-corpus Q-small | 74.0 | 66.0 | 0.200 |
| BM25, same comparison | TeleQnA | 73.6 | 75.6 | 0.063 |
| BM25, capped-context control | TeleQnA | 73.6 | 75.15 | 0.132 |

69.1% of the evidence units are exact duplicates within their specification
series, and duplicates fill 4.12 of the ten evidence slots given to the
generator. Applying the chunk collection's own deduplication rule to the units
raises their TeleQnA accuracy to 75.3%, a gain of 1.7 points (95% CI 0.55 to
2.85, Holm-corrected p = 0.017), which leaves them statistically
indistinguishable from fixed-size chunks while supplying about 17% fewer prompt
tokens.

Lexical-dense fusion raises Recall@20 on 523 held-out Telco-DPR queries from
50.10% to 52.01%, recovering relevant evidence for 10 of the 261 queries BM25
misses and losing none that BM25 already retrieves (Holm-corrected p = 0.012).
Query-aware table boosting and option-conditioned reranking show no measurable
benefit.

## Repository layout

```text
configs/                       locked experiment settings, path template
src/qaherra_repro/
  chunks.py                    fixed-size chunk collection and its integrity checks
  data.py                      Q-small, TeleQnA, and evidence loading
  dedup.py                     the exact deduplication rule and its mapping
  generation.py                prompt construction and the locked generator
  pipeline.py                  fusion, table boost, reranking, answer selection
  retrieval.py                 lexical and dense retrieval
  reranking.py                 cross-encoder reranking
  stats.py                     exact McNemar, paired bootstrap, declared Holm families
  telcodpr.py                  the retrieval relevance criterion
scripts/                       one entry point per reported result
baselines/telco_rag/           controlled Telco-RAG adaptation
results/                       aggregate results, each naming its source file
docs/                          reproduction map, deduplication, statistics, data
prompts/                       the exact answer-selection prompt
notebooks/                     the exploratory notebook, credentials removed
evaluation/                    the table-gate annotation protocol
tests/                         unit tests that need no 3GPP data
```

## Reproducing from scratch

```bash
python scripts/prepare_evidence.py        --paths configs/paths.yaml
python scripts/build_indexes.py           --paths configs/paths.yaml
python scripts/run_retrieval_eval.py      --paths configs/paths.yaml   # Fig. 2, Table IV
python scripts/run_qa.py --method qaherra_qonly --dataset teleqna --scope full
python scripts/run_chunk_baseline_4doc.py --retriever bm25  --paths configs/paths.yaml
python scripts/run_chunk_baseline_4doc.py --retriever dense --paths configs/paths.yaml
python scripts/dedup_stage_a_build.py     --paths configs/paths.yaml   # about 6 min, 12 GB RAM
python scripts/dedup_stage_b_retrieval.py --paths configs/paths.yaml
python scripts/dedup_stage_c_prompts.py   --paths configs/paths.yaml
python scripts/dedup_stage_d_qa.py        --paths configs/paths.yaml
python scripts/verify_results.py          --paths configs/paths.yaml --write-results
python scripts/make_figure3.py
```

`make help` lists the same steps as targets.

For `run_qa.py`, replace `qaherra_qonly` with `qaherra_option`, `no_rag`,
`bm25`, `tspec_llm`, or `chat3gpp`, and `--scope full` with `--scope four_doc`.

Set the API key only through the environment:

```bash
export OPENAI_API_KEY="..."        # PowerShell: $env:OPENAI_API_KEY="..."
```

## Two things worth knowing before comparing against an earlier draft

**The four-document chunk baselines were defective and were rerun.** The chunk
text was resolved through the corpus-wide chunk store with an index-local
position, which returned text from unrelated specifications for every chunk.
The chunk text is now read directly from the index metadata, and a startup
assertion checks that SHA-1 of every chunk's text reproduces its recorded
hash. `docs/CHUNK_LOOKUP_FIX.md` gives the diagnosis and the affected numbers;
the corrected gap is smaller than the defective one.

**Two cells of an earlier draft are not reported.** Dense retrieval over
fixed-size chunks in the two full-corpus settings has no surviving
per-question output, so those cells are blank rather than quoted from an
aggregate. The same applies to the Telco-RAG adaptation.

## Reliability conventions

Every run appends one record per question as soon as it completes, so an
interrupted run resumes without repeating work. A failed API call is written to
a separate error file and is never counted as an incorrect answer; rerunning a
script retries only the questions with no successful record, and the
significance tests run only once every question has one.

Holm correction is applied within families that are declared in advance and
listed in `docs/STATISTICS.md`. `qaherra_repro.stats.holm` takes the family
size explicitly and refuses to correct more comparisons than the family
declares.

## External implementations

* Telco-RAG: <https://github.com/netop-team/Telco-RAG> at
  `df92a34d3ad824fe4ef45b81fc956498a353a943`
* Chat3GPP: <https://github.com/huangl22/Chat3GPP> at
  `ce65e04da7474ff972516dd807b0278c1bfbf3b3`

`bash scripts/setup_external_baselines.sh` obtains these exact revisions. The
code here implements the controlled adaptations used for the reported
comparison; `docs/BASELINE_ADAPTATIONS.md` states what is preserved and what is
standardized.

## Data

3GPP specifications, extracted standards content, credentials, model weights,
and indexes are not redistributed here. `docs/DATA_AVAILABILITY.md` lists what
is archived alongside the paper and what to obtain from its original source.

## Licence

MIT for the code; see `LICENSE`. The archived evidence collection derives from
3GPP specifications and is redistributed only to the extent those
specifications allow.
