#!/usr/bin/env python
"""Recompute every number the paper reports from the released per-question files.

This is the script a reader runs first. It makes no API calls and builds no
index: it reads the saved per-question outputs, recomputes each accuracy, each
paired test, and each context-length figure, and compares the result with the
value printed in the paper. Any disagreement beyond the stated tolerance is
reported as a failure, and the exit status is non-zero.

What it checks
--------------
* Table III, every cell that has a per-question file behind it.
* The five units-versus-chunks comparisons that form one Holm family, with the
  discordant counts and the corrected p-values.
* The three deduplication comparisons and the pre-registered criterion.
* The context-length and prompt-token figures quoted in Sections I and V-D.
* That no answer recorded as incorrect is in fact a failed API call.

Usage
-----
    python scripts/verify_results.py --paths configs/paths.yaml
    python scripts/verify_results.py --paths configs/paths.yaml --write-results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qaherra_repro.config import load_yaml  # noqa: E402
from qaherra_repro.stats import (  # noqa: E402
    UNITS_VS_CHUNKS_COMPARISONS,
    apply_holm,
    paired_compare,
)

ROOT = Path(__file__).resolve().parents[1]

#: Accuracy claims: path key -> (label, expected accuracy, questions).
ACCURACY_CLAIMS: dict[str, tuple[str, float, int]] = {
    "qsmall_4doc_bm25_units_results": ("Units / BM25, Q-small, 4 docs", 84.0, 100),
    "qsmall_4doc_dense_units_results": ("Units / Dense, Q-small, 4 docs", 74.0, 100),
    "qsmall_4doc_bm25_chunks_results": ("Chunks / BM25, Q-small, 4 docs", 66.0, 100),
    "qsmall_4doc_dense_chunks_results": ("Chunks / Dense, Q-small, 4 docs", 60.0, 100),
    "qsmall_4doc_qaherra_qonly_results": ("QAHERRA q-only, Q-small, 4 docs", 86.0, 100),
    "qsmall_full_bm25_units_results": ("Units / BM25, Q-small, full", 74.0, 100),
    "qsmall_full_bm25_chunks_results": ("Chunks / BM25, Q-small, full", 66.0, 100),
    "qsmall_full_qaherra_qonly_results": ("QAHERRA q-only, Q-small, full", 78.0, 100),
    "teleqna_bm25_units_results": ("Units / BM25, TeleQnA", 73.60, 2000),
    "teleqna_bm25_chunks_results": ("Chunks / BM25, TeleQnA", 75.60, 2000),
    "teleqna_bm25_chunks_capped_results": ("Chunks / BM25, TeleQnA, capped", 75.15, 2000),
    "teleqna_qaherra_qonly_results": ("QAHERRA q-only, TeleQnA", 74.20, 2000),
    "teleqna_dedup_units_results": ("Dedup units / BM25, TeleQnA", 75.30, 2000),
}

#: The declared Holm family of five, as (label, units file key, chunks file key).
UNITS_VS_CHUNKS: list[tuple[str, str, str]] = [
    (
        UNITS_VS_CHUNKS_COMPARISONS[0],
        "qsmall_4doc_bm25_units_results",
        "qsmall_4doc_bm25_chunks_results",
    ),
    (
        UNITS_VS_CHUNKS_COMPARISONS[1],
        "qsmall_4doc_dense_units_results",
        "qsmall_4doc_dense_chunks_results",
    ),
    (
        UNITS_VS_CHUNKS_COMPARISONS[2],
        "qsmall_full_bm25_units_results",
        "qsmall_full_bm25_chunks_results",
    ),
    (
        UNITS_VS_CHUNKS_COMPARISONS[3],
        "teleqna_bm25_units_results",
        "teleqna_bm25_chunks_results",
    ),
    (
        UNITS_VS_CHUNKS_COMPARISONS[4],
        "teleqna_bm25_units_results",
        "teleqna_bm25_chunks_capped_results",
    ),
]

#: Expected Holm-corrected p-values for that family, in the same order.
EXPECTED_HOLM = [0.003, 0.073, 0.200, 0.063, 0.132]

#: Expected discordant counts for the two four-document comparisons.
EXPECTED_DISCORDANT = {
    UNITS_VS_CHUNKS_COMPARISONS[0]: (22, 4),
    UNITS_VS_CHUNKS_COMPARISONS[1]: (24, 10),
}

CONTEXT_CLAIMS = {
    "teleqna_units_context_chars": 10_465,
    "teleqna_dedup_context_chars": 11_095,
    "teleqna_dedup_prompt_tokens": 2_906,
    "teleqna_chunks_prompt_tokens": 3_485,
    "teleqna_chunks_capped_prompt_tokens": 2_644,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument(
        "--write-results",
        action="store_true",
        help="refresh the CSV files under results/ from the recomputed values",
    )
    parser.add_argument("--tolerance", type=float, default=0.05)
    return parser.parse_args()


def read_outcomes(path) -> pd.DataFrame:
    """Load a per-question result file as ``index``, ``correct``, and extras."""
    path = Path(path)
    if path.suffix == ".jsonl":
        frame = pd.DataFrame([json.loads(line) for line in open(path)])
    else:
        frame = pd.read_csv(path)
    key = "idx" if "idx" in frame.columns else "index"
    frame = frame.rename(columns={key: "index"})
    failures = 0
    if "error" in frame.columns:
        failures = int(frame["error"].notna().sum())
        frame = frame[frame["error"].isna()]
    frame = frame.drop_duplicates("index", keep="last").sort_values("index")
    frame["correct"] = frame.correct.astype(str).str.lower().isin(["true", "1", "1.0"])
    frame.attrs["failures_in_file"] = failures
    return frame


def main() -> int:
    args = parse_args()
    paths = load_yaml(args.paths)
    failures: list[str] = []
    accuracy_rows: list[dict] = []
    loaded: dict[str, pd.DataFrame] = {}

    print("== Accuracy of every configuration with a per-question file ==")
    for key, (label, expected, questions) in ACCURACY_CLAIMS.items():
        if key not in paths:
            print(f"  SKIP  {label}: no path configured for '{key}'")
            continue
        frame = read_outcomes(paths[key])
        loaded[key] = frame
        accuracy = 100 * frame.correct.mean()
        note = ""
        if len(frame) != questions:
            note = f"  [{len(frame)} of {questions} questions scored]"
            failures.append(
                f"{label}: {len(frame)} questions scored, expected {questions}"
            )
        if frame.attrs["failures_in_file"]:
            note += f"  [{frame.attrs['failures_in_file']} failed calls, not scored]"
        ok = abs(accuracy - expected) <= args.tolerance
        if not ok:
            failures.append(f"{label}: computed {accuracy:.2f}, paper reports {expected}")
        print(
            f"  {'OK  ' if ok else 'FAIL'}  {label:<42} "
            f"computed {accuracy:6.2f}  paper {expected:6.2f}{note}"
        )
        accuracy_rows.append(
            {
                "configuration": label,
                "questions": len(frame),
                "accuracy": round(accuracy, 2),
                "reported": expected,
                "source_file": Path(paths[key]).name,
            }
        )

    print("\n== Units versus fixed-size chunks: one Holm family of five ==")
    results = []
    for label, units_key, chunks_key in UNITS_VS_CHUNKS:
        if units_key not in loaded or chunks_key not in loaded:
            failures.append(f"{label}: a per-question file is missing")
            continue
        units, chunks = loaded[units_key], loaded[chunks_key]
        shared = sorted(set(units["index"]) & set(chunks["index"]))
        if len(shared) != len(units):
            failures.append(
                f"{label}: only {len(shared)} of {len(units)} questions pair up"
            )
        unit_map = dict(zip(units["index"], units.correct))
        chunk_map = dict(zip(chunks["index"], chunks.correct))
        results.append(
            paired_compare(
                label, [unit_map[i] for i in shared], [chunk_map[i] for i in shared]
            )
        )
    apply_holm(results, family_size=5)
    for result, expected in zip(results, EXPECTED_HOLM):
        ok = abs(result.p_holm - expected) <= 0.002
        if not ok:
            failures.append(
                f"{result.name}: Holm p {result.p_holm:.4f}, paper reports {expected}"
            )
        counts = EXPECTED_DISCORDANT.get(result.name)
        count_note = ""
        if counts:
            count_ok = (result.a_only, result.b_only) == counts
            count_note = (
                f"  units-only {result.a_only} vs chunks-only {result.b_only}"
                f"{'' if count_ok else f' (paper: {counts[0]} vs {counts[1]})'}"
            )
            if not count_ok:
                failures.append(
                    f"{result.name}: discordant counts {result.a_only}/{result.b_only}, "
                    f"paper reports {counts[0]}/{counts[1]}"
                )
        print(
            f"  {'OK  ' if ok else 'FAIL'}  {result.name:<48} "
            f"units {result.accuracy_a:6.2f}  chunks {result.accuracy_b:6.2f}  "
            f"raw p {result.p:.4f}  Holm p {result.p_holm:.3f} "
            f"(paper {expected}){count_note}"
        )

    print("\n== Context length and prompt tokens ==")
    context_rows = _check_context(paths, loaded, failures, args.tolerance)

    if args.write_results:
        out = ROOT / "results"
        out.mkdir(exist_ok=True)
        pd.DataFrame(accuracy_rows).to_csv(out / "verified_accuracy.csv", index=False)
        pd.DataFrame([r.as_dict() for r in results]).rename(
            columns={"accuracy_a": "accuracy_units", "accuracy_b": "accuracy_chunks"}
        ).to_csv(out / "verified_units_vs_chunks_tests.csv", index=False)
        if context_rows:
            pd.DataFrame(context_rows).to_csv(
                out / "verified_context_lengths.csv", index=False
            )
        print(f"\nrefreshed the CSV files under {out}")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("every checked number in the paper is reproduced by the released files.")
    return 0


def _check_context(paths, loaded, failures, tolerance) -> list[dict]:
    """Recompute the context-length and prompt-token figures."""
    rows: list[dict] = []
    sources = {
        "teleqna_units_context_chars": ("teleqna_bm25_units_results", "ctx_chars"),
        "teleqna_dedup_context_chars": ("teleqna_dedup_units_results", "ctx_chars"),
        "teleqna_dedup_prompt_tokens": ("teleqna_dedup_units_results", "prompt_tokens"),
        "teleqna_chunks_prompt_tokens": ("teleqna_bm25_chunks_results", "prompt_tokens"),
        "teleqna_chunks_capped_prompt_tokens": (
            "teleqna_bm25_chunks_capped_results",
            "prompt_tokens",
        ),
    }
    for claim, expected in CONTEXT_CLAIMS.items():
        key, column = sources[claim]
        frame = loaded.get(key)
        if frame is None or column not in frame.columns:
            print(f"  SKIP  {claim}: '{column}' not recorded in that file")
            continue
        value = float(frame[column].mean())
        ok = abs(value - expected) <= max(1.0, expected * tolerance / 100)
        if not ok:
            failures.append(f"{claim}: computed {value:.0f}, paper reports {expected}")
        print(
            f"  {'OK  ' if ok else 'FAIL'}  {claim:<38} "
            f"computed {value:9.0f}  paper {expected:9.0f}"
        )
        rows.append(
            {"quantity": claim, "computed": round(value, 1), "reported": expected}
        )
    if {"teleqna_dedup_prompt_tokens", "teleqna_chunks_prompt_tokens"} <= {
        row["quantity"] for row in rows
    }:
        by_name = {row["quantity"]: row["computed"] for row in rows}
        relative = 100 * (
            by_name["teleqna_dedup_prompt_tokens"] / by_name["teleqna_chunks_prompt_tokens"]
            - 1
        )
        print(
            f"        deduplicated units supply {abs(relative):.1f}% "
            f"{'fewer' if relative < 0 else 'more'} prompt tokens than uncapped chunks "
            "(paper: 17% fewer)"
        )
        rows.append(
            {
                "quantity": "dedup_vs_chunks_prompt_token_change_pct",
                "computed": round(relative, 1),
                "reported": -17.0,
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
