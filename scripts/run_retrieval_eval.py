#!/usr/bin/env python
"""Evaluate BM25, dense retrieval, and QAHERRA on all 523 Telco-DPR queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

from qaherra_repro.config import load_yaml, require_path, set_seed
from qaherra_repro.data import CsvEvidenceStore, load_table_mask
from qaherra_repro.evaluation import exact_mcnemar, holm_correct
from qaherra_repro.retrieval import (
    BM25Retriever,
    DenseRetriever,
    KeywordTableGate,
    boost_table_candidates,
    reciprocal_rank_fusion,
)
from qaherra_repro.telcodpr import is_relevant


CUTOFFS = [1, 5, 10, 20, 50, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qaherra.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    return parser.parse_args()


def bootstrap_difference(
    baseline: np.ndarray,
    method: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Paired bootstrap CI for the absolute recall difference in percentage points."""
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float32)
    count = len(baseline)
    for sample in range(samples):
        indices = rng.integers(0, count, count)
        differences[sample] = 100.0 * (
            method[indices].mean() - baseline[indices].mean()
        )
    low, high = np.quantile(differences, [0.025, 0.975])
    return float(low), float(high)


def outcomes(
    rankings: np.ndarray,
    metadata: dict[int, dict[str, str]],
    gold_documents: list[str],
    gold_sections: list[str],
    cutoff: int,
) -> np.ndarray:
    """Mark a query correct when any top-k row matches document and top section."""
    values = np.zeros(len(rankings), dtype=bool)
    for query_index, ranking in enumerate(rankings):
        for raw_index in ranking[:cutoff]:
            index = int(raw_index)
            if index < 0:
                continue
            row = metadata[index]
            if is_relevant(
                row["doc_id"],
                row["section_path"],
                gold_documents[query_index],
                gold_sections[query_index],
            ):
                values[query_index] = True
                break
    return values


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    paths = load_yaml(args.paths)
    seed = int(config["seed"])
    set_seed(seed)

    # Public Telco-DPR has 523 held-out relevance rows and one gold target per
    # query. All 523 are evaluated; no hand-selected 418-query subset is used.
    queries = pd.DataFrame(
        load_dataset("ismailduru/Telco-DPR", "queries", split="queries")
    )
    relevance = pd.DataFrame(
        load_dataset("ismailduru/Telco-DPR", "relevant_docs", split="test")
    )
    corpus = pd.DataFrame(
        load_dataset("ismailduru/Telco-DPR", "corpus", split="small")
    )
    evaluation = (
        relevance
        .merge(queries[["id", "question"]], left_on="query_id", right_on="id")
        .merge(
            corpus[["id", "Document_file_name", "Section_name"]],
            left_on="corpus_id",
            right_on="id",
            suffixes=("_query", "_corpus"),
        )
        .reset_index(drop=True)
    )
    if len(evaluation) != 523:
        raise ValueError(f"Expected 523 Telco-DPR test queries, found {len(evaluation)}")

    retrieval = config["retrieval"]
    bm25 = BM25Retriever(
        require_path(paths, "bm25_index"),
        depth=int(retrieval["pool_size"]),
    )
    dense = DenseRetriever(
        require_path(paths, "domain_model"),
        require_path(paths, "dense_faiss"),
        depth=int(retrieval["pool_size"]),
    )
    table_mask = load_table_mask(require_path(paths, "evidence_csv"))
    gate = KeywordTableGate(config["table_gate"]["patterns"])

    count = len(evaluation)
    depth = int(retrieval["pool_size"])
    bm25_rankings = np.full((count, depth), -1, dtype=np.int64)
    dense_rankings = np.full((count, depth), -1, dtype=np.int64)
    hybrid_rankings = np.full((count, depth), -1, dtype=np.int64)
    gate_values = np.zeros(count, dtype=bool)

    for index, row in evaluation.iterrows():
        question = str(row["question"])
        bm_indices, _, _ = bm25.search(question)
        dense_indices, dense_scores, _ = dense.search(question)
        adjusted_indices, _, activated = boost_table_candidates(
            question,
            dense_indices,
            dense_scores,
            table_mask,
            gate,
            float(retrieval["table_boost"]),
        )
        fused_indices, _ = reciprocal_rank_fusion(
            [bm_indices, adjusted_indices],
            [float(retrieval["bm25_weight"]), float(retrieval["dense_weight"])],
            float(retrieval["rrf_constant"]),
            depth,
        )
        bm25_rankings[index] = bm_indices
        dense_rankings[index] = dense_indices
        hybrid_rankings[index] = fused_indices
        gate_values[index] = activated
        if index < 5 or (index + 1) % 50 == 0:
            print(f"Retrieved {index + 1}/{count}")

    needed = set(int(value) for value in np.concatenate(
        [bm25_rankings.ravel(), dense_rankings.ravel(), hybrid_rankings.ravel()]
    ) if int(value) >= 0)
    metadata = CsvEvidenceStore(require_path(paths, "evidence_csv")).fetch(needed)
    gold_documents = evaluation["Document_file_name"].astype(str).tolist()
    gold_sections = evaluation["Section_name"].astype(str).tolist()

    method_rankings = {
        "BM25": bm25_rankings,
        "Domain-adapted dense": dense_rankings,
        "QAHERRA hybrid": hybrid_rankings,
    }
    method_outcomes: dict[str, dict[int, np.ndarray]] = {}
    result_rows: list[dict[str, object]] = []
    for method, rankings in method_rankings.items():
        method_outcomes[method] = {}
        row: dict[str, object] = {"method": method}
        for cutoff in CUTOFFS:
            values = outcomes(
                rankings, metadata, gold_documents, gold_sections, cutoff
            )
            method_outcomes[method][cutoff] = values
            row[f"R@{cutoff}"] = 100.0 * float(values.mean())
        result_rows.append(row)

    p_values = []
    significance_rows = []
    for cutoff in CUTOFFS:
        baseline = method_outcomes["BM25"][cutoff]
        method = method_outcomes["QAHERRA hybrid"][cutoff]
        bm25_only, qaherra_only, p_value = exact_mcnemar(baseline, method)
        low, high = bootstrap_difference(baseline, method, args.bootstrap, seed)
        p_values.append(p_value)
        significance_rows.append(
            {
                "k": cutoff,
                "bm25_hits": int(baseline.sum()),
                "qaherra_hits": int(method.sum()),
                "bm25_recall_percent": 100.0 * float(baseline.mean()),
                "qaherra_recall_percent": 100.0 * float(method.mean()),
                "bm25_only": bm25_only,
                "qaherra_only": qaherra_only,
                "mcnemar_exact_p": p_value,
                "ci95_low_pp": low,
                "ci95_high_pp": high,
            }
        )

    corrected = holm_correct(p_values)
    for row, corrected_p in zip(significance_rows, corrected):
        row["holm_corrected_p"] = float(corrected_p)
        row["significant_after_holm"] = bool(corrected_p < 0.05)

    output_dir = Path(paths["output_dir"]) / "telcodpr_523"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(result_rows)
    significance = pd.DataFrame(significance_rows)
    results.to_csv(output_dir / "retrieval_results.csv", index=False)
    significance.to_csv(output_dir / "retrieval_significance.csv", index=False)
    np.savez_compressed(
        output_dir / "retrieval_rankings.npz",
        bm25=bm25_rankings,
        dense=dense_rankings,
        qaherra=hybrid_rankings,
        table_gate=gate_values,
    )
    print("\nRETRIEVAL RESULTS")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\nEXACT PAIRED SIGNIFICANCE")
    print(significance.to_string(index=False))


if __name__ == "__main__":
    main()

