# Data availability

## What this repository contains

* All code needed to build the evidence collection, the indexes, the
  deduplication mapping, and every reported run.
* The locked configuration of every experiment, in `configs/`.
* The aggregate results of every reported configuration, in `results/`, each
  row naming the per-question file it is computed from.
* The verification script that recomputes every reported number.

## What it does not contain

* **3GPP specifications.** The specifications are published by 3GPP and are not
  redistributed here. They are available from the 3GPP specification archive.
* **Extracted standards content.** `evidence_units_full.csv` derives from those
  specifications and is not redistributed in this repository.
* **API credentials.** Set `OPENAI_API_KEY` in the environment. No key is
  committed, and `scripts/security_scan.py` fails the build if one appears.
* **Model weights, embeddings, and indexes.** The adapted encoder, the FAISS
  indexes, and the BM25 indexes are rebuilt by `scripts/build_indexes.py`.

## Archived artifacts

The following are deposited in the archive that accompanies the paper, under a
single DOI, and are what a reader needs in order to check a number without
rerunning anything:

| Artifact | Size | Purpose |
|---|---|---|
| `evidence_units_full.csv` | large | the 2,396,136 evidence units with provenance |
| `kept_positions.npy` | 2.9 MB | original position of each of the 740,002 retained units |
| `representative_of.npy` | 9.6 MB | the deduplication mapping, one entry per original unit |
| per-question outputs | moderate | predictions, retrieved evidence, latency, and token usage for every configuration in Table III |
| `telcodpr_per_query_523.jsonl` | small | per-query retrieval outcomes on the held-out split |
| `finetuned_model/` | 90 MB | the telecom-adapted encoder and its model card |

Set the paths to these in `configs/paths.yaml`, then run

```bash
python scripts/verify_results.py --paths configs/paths.yaml
```

which needs no API key, no index, and no GPU.

## Third-party data and code

| Source | Use |
|---|---|
| TSpec-LLM | the 100-question Q-small benchmark |
| TeleQnA | the 2,000 `Standards specifications` questions |
| Telco-DPR | 1,218 training queries and the disjoint 523-query test split |
| `sentence-transformers/all-MiniLM-L6-v2` | base dense encoder |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | cross-encoder reranker |
| Telco-RAG, Chat3GPP | controlled adaptations, at the revisions pinned in the README |

Each is used under its own licence and is cited in the paper.

## Licence

The code in this repository is released under the MIT licence; see `LICENSE`.
The archived evidence collection derives from 3GPP specifications and is
redistributed only to the extent those specifications allow.
