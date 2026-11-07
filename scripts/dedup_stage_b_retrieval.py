#!/usr/bin/env python
"""Stage B: retrieval over the deduplicated units on the 523 Telco-DPR queries.

Builds a BM25 index over the retained units from Stage A and evaluates
Recall@{1,5,10,20,50,100} and MRR@100 against the non-deduplicated run, under
the paper's relevance criterion: a unit counts as relevant only when it comes
from the gold specification, including its release, and the gold top-level
section.

Two variants are reported, because deduplication changes which occurrence of a
repeated passage remains retrievable:

``dedup``
    The retained unit's own provenance decides relevance. This is the strict
    reading, and the one the paper's criterion implies.
``dedup_aware``
    A retained unit counts as relevant when *any* occurrence it represents
    matches, resolved through the published mapping. This isolates how much of
    any change is a provenance effect rather than a ranking effect.

The script first validates the non-deduplicated baseline against its reported
Recall@{5,10,20} of 41.30, 46.46, and 50.10, and refuses to produce results if
the baseline file does not reproduce them. It also checks that the compact
integer tokenization used for the large index reproduces the ordinary ``bm25s``
index on a 50,000-unit sample, so that memory savings cannot change a score.

No API calls.

Usage
-----
    python scripts/dedup_stage_b_retrieval.py --paths configs/paths.yaml
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import pickle
import sys
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
from bm25s.tokenization import Tokenized
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qaherra_repro.config import load_yaml, require_path  # noqa: E402
from qaherra_repro.stats import holm  # noqa: E402
from qaherra_repro.telcodpr import is_relevant  # noqa: E402

CUTOFFS = (1, 5, 10, 20, 50, 100)
BASELINE_RECALL = {5: 41.30, 10: 46.46, 20: 50.10}
SHARD_SIZE = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.yaml")
    return parser.parse_args()


def validate_baseline(path) -> list[dict]:
    """Load the non-deduplicated per-query file and check its reported recall."""
    queries = [json.loads(line) for line in open(path)]
    if len(queries) != 523:
        raise SystemExit(f"expected 523 queries, found {len(queries)}")
    for cutoff, expected in BASELINE_RECALL.items():
        got = round(
            100 * np.mean([int(q[f"bm25_hit_at_{cutoff}"]) for q in queries]), 2
        )
        if got != expected:
            raise SystemExit(
                f"baseline Recall@{cutoff} is {got}, expected {expected}; "
                "this is not the run the paper reports"
            )
    inconsistent = {
        q["query_index"]
        for q in queries
        for cutoff in CUTOFFS
        if (0 < int(q["bm25_first_relevant_rank_within_100"]) <= cutoff)
        != bool(int(q[f"bm25_hit_at_{cutoff}"]))
    }
    if inconsistent:
        raise SystemExit(
            f"first-relevant rank disagrees with the hit flags for "
            f"{len(inconsistent)} queries"
        )
    return queries


def to_arrays(texts: list[str], vocabulary: dict[str, int]) -> list[np.ndarray]:
    """Tokenize with the paper's settings into compact int32 arrays."""
    tokens = bm25s.tokenize(
        texts, stopwords="en", return_ids=False, show_progress=False
    )
    return [
        np.fromiter(
            (vocabulary.setdefault(word, len(vocabulary)) for word in document),
            dtype=np.int32,
            count=len(document),
        )
        for document in tokens
    ]


def check_tokenizer_equivalence(sample: list[str], questions: list[str]) -> None:
    """Confirm the compact index scores identically to the ordinary one."""
    reference = bm25s.BM25()
    reference.index(
        bm25s.tokenize(sample, stopwords="en", show_progress=False), show_progress=False
    )
    vocabulary: dict[str, int] = {}
    compact = bm25s.BM25()
    compact.index(
        Tokenized(ids=to_arrays(sample, vocabulary), vocab=vocabulary),
        show_progress=False,
    )
    tokenized = bm25s.tokenize(questions, stopwords="en", show_progress=False)
    ids_a, scores_a = reference.retrieve(tokenized, k=100, show_progress=False)
    ids_b, scores_b = compact.retrieve(tokenized, k=100, show_progress=False)
    identical = np.allclose(scores_a, scores_b, rtol=0, atol=1e-6)
    differing = ids_a != ids_b
    ties_only = bool(
        np.all(np.isclose(scores_a[differing], scores_b[differing], atol=1e-6))
    )
    print(
        f"tokenizer equivalence on {len(sample):,} units x {len(questions)} queries: "
        f"scores identical={identical} | differing positions={int(differing.sum())} "
        f"(all exact ties={ties_only})"
    )
    if not (identical and ties_only):
        raise SystemExit("the compact index does not reproduce the ordinary index")


def main() -> None:
    args = parse_args()
    paths = load_yaml(args.paths)
    out_dir = Path(require_path(paths, "dedup_output_dir"))
    work_dir = Path(require_path(paths, "dedup_work_dir"))

    queries = validate_baseline(require_path(paths, "telcodpr_per_query"))
    question_texts = [q["question"] for q in queries]

    meta = pickle.load(open(work_dir / "meta.pkl", "rb"))
    doc_ids, section_paths, shards = (
        meta["doc_ids"],
        meta["section_paths"],
        meta["shards"],
    )
    representative_of = np.load(out_dir / "representative_of.npy")

    check_tokenizer_equivalence(
        pickle.load(open(work_dir / "texts_000.pkl", "rb")), question_texts
    )
    gc.collect()

    vocabulary: dict[str, int] = {}
    ids: list[np.ndarray] = []
    for shard in range(shards):
        ids += to_arrays(
            pickle.load(open(work_dir / f"texts_{shard:03d}.pkl", "rb")), vocabulary
        )
        gc.collect()
    if len(ids) != len(doc_ids):
        raise SystemExit(
            f"token arrays ({len(ids)}) do not match retained units ({len(doc_ids)})"
        )
    index = bm25s.BM25()
    index.index(Tokenized(ids=ids, vocab=vocabulary), show_progress=False)
    del ids
    gc.collect()
    print(f"indexed {len(doc_ids):,} deduplicated units | vocabulary {len(vocabulary):,}")

    tokenized = bm25s.tokenize(question_texts, stopwords="en", show_progress=False)
    top, _ = index.retrieve(tokenized, k=100, show_progress=False)

    needed = {int(value) for row in top for value in row}
    members: dict[int, list[tuple[str, str]]] = collections.defaultdict(list)
    position = 0
    for frame in pd.read_csv(
        require_path(paths, "evidence_csv"),
        chunksize=200_000,
        usecols=["doc_id", "section_path"],
    ):
        for doc_id, section in zip(
            frame.doc_id.astype(str), frame.section_path.astype(str)
        ):
            represented = int(representative_of[position])
            if represented in needed:
                members[represented].append((doc_id, section))
            position += 1

    rows = []
    for query, row in zip(queries, top):
        gold_document = json.loads(query["gold_documents"])[0]
        gold_section = json.loads(query["gold_sections"])[0]
        strict = [
            is_relevant(
                doc_ids[int(i)], section_paths[int(i)], gold_document, gold_section
            )
            for i in row
        ]
        aware = [
            any(
                is_relevant(d, s, gold_document, gold_section)
                for d, s in members[int(i)]
            )
            for i in row
        ]
        record = {"query_index": query["query_index"]}
        for name, relevance in (("dedup", strict), ("dedup_aware", aware)):
            first = next((k + 1 for k, hit in enumerate(relevance) if hit), -1)
            record[f"{name}_first_rank"] = first
            for cutoff in CUTOFFS:
                record[f"{name}_hit_at_{cutoff}"] = int(any(relevance[:cutoff]))
        record["orig_first_rank"] = int(query["bm25_first_relevant_rank_within_100"])
        for cutoff in CUTOFFS:
            record[f"orig_hit_at_{cutoff}"] = int(query[f"bm25_hit_at_{cutoff}"])
        rows.append(record)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "telcodpr_per_query_dedup.csv", index=False)

    reciprocal = lambda series: np.where(series > 0, 1.0 / series.clip(lower=1), 0.0)
    rng = np.random.default_rng(0)
    summary = []
    for name in ("dedup", "dedup_aware"):
        tests = []
        for cutoff in (5, 10, 20):
            new = frame[f"{name}_hit_at_{cutoff}"].values
            old = frame[f"orig_hit_at_{cutoff}"].values
            gained = int(np.sum((new == 1) & (old == 0)))
            lost = int(np.sum((new == 0) & (old == 1)))
            tests.append(
                {
                    "variant": name,
                    "metric": f"Recall@{cutoff}",
                    "original": 100 * old.mean(),
                    "deduplicated": 100 * new.mean(),
                    "gained": gained,
                    "lost": lost,
                    "p": binomtest(gained, gained + lost, 0.5).pvalue
                    if gained + lost
                    else 1.0,
                }
            )
        new_mrr = reciprocal(frame[f"{name}_first_rank"])
        old_mrr = reciprocal(frame["orig_first_rank"])
        difference = new_mrr - old_mrr
        draws = rng.integers(0, difference.size, (10_000, difference.size))
        low, high = np.percentile(difference[draws].mean(axis=1), [2.5, 97.5])
        tests.append(
            {
                "variant": name,
                "metric": "MRR@100",
                "original": old_mrr.mean(),
                "deduplicated": new_mrr.mean(),
                "gained": int(np.sum(difference > 0)),
                "lost": int(np.sum(difference < 0)),
                "p": np.nan,
                "ci_lo": low,
                "ci_hi": high,
            }
        )
        for test, adjusted in zip(tests[:3], holm([t["p"] for t in tests[:3]])):
            test["p_holm"] = adjusted
        summary += tests

    table = pd.DataFrame(summary)
    table["abs_change"] = table.deduplicated - table.original
    table["rel_change_pct"] = 100 * table.abs_change / table.original
    table.to_csv(out_dir / "telcodpr_dedup_summary.csv", index=False)
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
