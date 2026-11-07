#!/usr/bin/env python
"""Stage D: TeleQnA accuracy over the deduplicated evidence units.

Answers all 2,000 TeleQnA questions from the deduplicated top-10 lists saved by
Stage C, then runs the three paired comparisons the paper reports:

* against evidence units without deduplication (expected: +1.7 points,
  95% CI 0.55-2.85, Holm-corrected p = 0.017);
* against fixed-size chunks (expected: statistically indistinguishable);
* against fixed-size chunks with the context capped at the evidence-unit mean,
  which separates the effect of context length from the effect of duplicates.

The confirmation criterion was fixed before this comparison was run: it is met
when the upper bound of the 95% confidence interval for the chunk-over-unit
difference falls below the 2-point chunk advantage measured without
deduplication. The script computes the verdict rather than asserting it.

Retrieval is not repeated here; the top-10 lists come from Stage C, so the
generation step is reproducible on its own. Every answer is appended to disk
immediately. A failed API call is written to a separate error file and is never
counted as an incorrect answer; rerunning the script retries only the
questions that have no successful record, and the tests are computed only once
all 2,000 questions have one.

Usage
-----
    export OPENAI_API_KEY=...
    python scripts/dedup_stage_d_qa.py --paths configs/paths.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qaherra_repro.config import load_yaml, require_path  # noqa: E402
from qaherra_repro.data import load_teleqna  # noqa: E402
from qaherra_repro.generation import (  # noqa: E402
    OpenAIAnswerGenerator,
    build_answer_prompt,
    parse_answer,
)
from qaherra_repro.stats import (  # noqa: E402
    apply_holm,
    confirmation_criterion_met,
    paired_compare,
)

SHARD_SIZE = 50_000
BUDGET = 14_000
ORIGINAL_CHUNK_ADVANTAGE = 2.0
LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qaherra.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def load_outcomes(path) -> dict[int, bool]:
    """Read per-question correctness from a JSONL or CSV result file."""
    path = str(path)
    if path.endswith(".jsonl"):
        frame = pd.DataFrame([json.loads(line) for line in open(path)])
    else:
        frame = pd.read_csv(path)
    key = "idx" if "idx" in frame.columns else "index"
    if "error" in frame.columns:
        frame = frame[frame["error"].isna()]
    frame = frame.drop_duplicates(key, keep="last")
    correct = frame.correct.astype(str).str.lower().isin(["true", "1", "1.0"])
    return dict(zip(frame[key].astype(int), correct))


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    paths = load_yaml(args.paths)
    out_dir = Path(require_path(paths, "dedup_output_dir"))
    work_dir = Path(require_path(paths, "dedup_work_dir"))
    result_path = out_dir / "bm25dedup_teleqna_full.jsonl"
    error_path = out_dir / "bm25dedup_teleqna_full_errors.jsonl"

    comparison = pd.read_csv(out_dir / "teleqna_prompt_comparison.csv")
    rankings = {
        int(index): json.loads(positions)
        for index, positions in zip(
            comparison["index"], comparison.dedup_top10_original_positions
        )
    }
    questions = load_teleqna(require_path(paths, "teleqna")).reset_index(drop=True)
    if len(questions) != 2000 or set(rankings) != set(questions["index"].astype(int)):
        raise SystemExit("the saved top-10 lists do not align with the 2,000 questions")

    meta = pickle.load(open(work_dir / "meta.pkl", "rb"))
    shards = meta["shards"]
    representative_of = np.load(out_dir / "representative_of.npy")

    needed_positions = {p for L in rankings.values() for p in L}
    needed_representatives = {int(representative_of[p]) for p in needed_positions}
    texts: dict[int, str] = {}
    for shard in range(shards):
        low = shard * SHARD_SIZE
        wanted = [r for r in needed_representatives if low <= r < low + SHARD_SIZE]
        if wanted:
            payload = pickle.load(open(work_dir / f"texts_{shard:03d}.pkl", "rb"))
            for represented in wanted:
                texts[represented] = payload[represented - low]
            del payload
            gc.collect()

    doc_id_of: dict[int, str] = {}
    position = 0
    for frame in pd.read_csv(
        require_path(paths, "evidence_csv"), chunksize=200_000, usecols=["doc_id"]
    ):
        for doc_id in frame.doc_id.astype(str):
            if position in needed_positions:
                doc_id_of[position] = doc_id
            position += 1

    def context(positions: list[int]) -> str:
        parts: list[str] = []
        total = 0
        for rank, slot in enumerate(positions, start=1):
            block = (
                f"[Source {rank} | {doc_id_of[slot]}]\n"
                f"{texts[int(representative_of[slot])]}"
            )
            if total + len(block) > BUDGET:
                break
            parts.append(block)
            total += len(block) + 2
        return "\n\n---\n\n".join(parts)

    generator = OpenAIAnswerGenerator(
        config["generation"]["model"], config["generation"]["temperature"]
    )
    done = (
        {json.loads(line)["index"] for line in open(result_path)}
        if result_path.exists()
        else set()
    )
    todo = [row for row in questions.itertuples() if int(row.index) not in done]
    print(f"{len(done)} already answered, {len(todo)} to answer")

    def work(row) -> None:
        prompt_context = context(rankings[int(row.index)])
        prompt = build_answer_prompt(row.question, row.options, prompt_context)
        last_error = None
        for attempt in range(3):
            try:
                raw = generator.answer(prompt)
                prediction = parse_answer(raw["raw_answer"], list(row.options))
                record = {
                    "index": int(row.index),
                    "gold": row.gold,
                    "prediction": prediction,
                    "correct": prediction == row.gold,
                    "ctx_chars": len(prompt_context),
                    "raw_answer": raw["raw_answer"],
                    "prompt_tokens": raw["prompt_tokens"],
                    "returned_model": raw["returned_model"],
                }
                with LOCK, open(result_path, "a") as handle:
                    handle.write(json.dumps(record) + "\n")
                return
            except Exception as error:
                last_error = repr(error)
                time.sleep(2 * (attempt + 1))
        with LOCK, open(error_path, "a") as handle:
            handle.write(json.dumps({"index": int(row.index), "error": last_error}) + "\n")

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, todo))

    answers = (
        pd.DataFrame([json.loads(line) for line in open(result_path)])
        .drop_duplicates("index", keep="last")
        .sort_values("index")
    )
    missing = 2000 - len(answers)
    if missing:
        raise SystemExit(
            f"{missing} questions are still unanswered (see {error_path.name}). "
            "Rerun this script to retry only those. No tests are computed from a "
            "partial run."
        )
    print(
        f"all 2,000 answered | accuracy {100 * answers.correct.mean():.2f}% "
        f"| mean context {answers.ctx_chars.mean():.0f} chars "
        f"| mean prompt tokens {answers.prompt_tokens.mean():.0f}"
    )

    dedup = dict(zip(answers["index"].astype(int), answers.correct.astype(bool)))
    others = [
        ("units, no deduplication", require_path(paths, "teleqna_bm25_units_results")),
        ("chunks", require_path(paths, "teleqna_bm25_chunks_results")),
        ("chunks, capped context", require_path(paths, "teleqna_bm25_chunks_capped_results")),
    ]
    results = []
    for name, path in others:
        other = load_outcomes(path)
        shared = sorted(set(dedup) & set(other))
        if len(shared) != 2000:
            raise SystemExit(f"{name}: only {len(shared)} paired questions")
        results.append(
            paired_compare(
                f"dedup units vs {name}",
                [dedup[i] for i in shared],
                [other[i] for i in shared],
            )
        )
    apply_holm(results, family_size=3)

    table = pd.DataFrame([result.as_dict() for result in results])
    table = table.rename(
        columns={"accuracy_a": "dedup_units", "accuracy_b": "other"}
    )
    table.to_csv(out_dir / "dedup_qa_tests.csv", index=False)
    print(table.round(4).to_string())

    chunks = results[1]
    met = confirmation_criterion_met(-chunks.ci_low, ORIGINAL_CHUNK_ADVANTAGE)
    print(
        "pre-registered criterion (upper bound of chunk-over-unit difference below "
        f"{ORIGINAL_CHUNK_ADVANTAGE:.1f} points): "
        f"{'MET' if met else 'NOT MET'} "
        f"(chunk advantage CI {-chunks.ci_high:.2f} to {-chunks.ci_low:.2f})"
    )


if __name__ == "__main__":
    main()
