#!/usr/bin/env python
"""Run the controlled Telco-RAG baseline with immediate per-question saving."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from qaherra_repro.config import load_yaml, require_path, set_seed
from qaherra_repro.data import load_qsmall, load_teleqna
from qaherra_repro.evaluation import append_jsonl, load_jsonl, summarise_qa
from qaherra_repro.generation import (
    OpenAIAnswerGenerator,
    build_answer_prompt,
    build_context,
    parse_answer,
)

# This adapter is outside the installed src package by design, so the import
# clearly identifies it as a controlled external-baseline compatibility layer.
from baselines.telco_rag.adapter import TelcoRAGAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["qsmall", "teleqna"], required=True)
    parser.add_argument("--scope", choices=["four_doc", "full"], default="full")
    parser.add_argument("--config", default="configs/qaherra.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    paths = load_yaml(args.paths)
    set_seed(int(config["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    questions = (
        load_qsmall(require_path(paths, "qsmall"))
        if args.dataset == "qsmall"
        else load_teleqna(require_path(paths, "teleqna"))
    )
    adapter = TelcoRAGAdapter(
        require_path(paths, "telcorag_root"),
        require_path(paths, "telcorag_series_index_dir"),
        require_path(paths, "telcorag_metadata_db"),
        four_doc_index=(
            require_path(paths, "telcorag_four_doc_index")
            if args.scope == "four_doc"
            else None
        ),
        four_doc_metadata=(
            require_path(paths, "telcorag_four_doc_metadata")
            if args.scope == "four_doc"
            else None
        ),
        generator_model=config["generation"]["model"],
        device=device,
    )
    generator = OpenAIAnswerGenerator(
        config["generation"]["model"],
        config["generation"]["temperature"],
    )

    output = Path(args.output) if args.output else (
        Path(paths["output_dir"]) / f"telco_rag_{args.dataset}_{args.scope}.jsonl"
    )
    existing = load_jsonl(output)
    completed = {int(record["index"]) for record in existing}

    for _, row in questions.iterrows():
        index = int(row["index"])
        if index in completed:
            continue
        start = time.perf_counter()
        record = {
            "index": index,
            "id": row["id"],
            "dataset": args.dataset,
            "scope": args.scope,
            "method": "telco_rag",
            "question": row["question"],
            "options": row["options"],
            "gold": row["gold"],
            "difficulty": row.get("difficulty", "Not provided"),
            "error": None,
        }
        try:
            retrieval_start = time.perf_counter()
            retrieved = adapter.retrieve(row["question"], final_k=10)
            retrieval_seconds = time.perf_counter() - retrieval_start

            # Convert the retrieved tuples to the same context format used by
            # every other controlled method.
            evidence = {position: item[3] for position, item in enumerate(retrieved["results"])}
            ranking = list(evidence)
            context, used = build_context(
                ranking,
                evidence,
                int(config["generation"]["max_context_characters"]),
            )
            prompt = build_answer_prompt(row["question"], row["options"], context)
            generation = generator.answer(prompt)
            prediction = parse_answer(generation["raw_answer"], list(row["options"]))
            record.update(generation)
            record.update(
                {
                    "prediction": prediction,
                    "correct": prediction == row["gold"],
                    "retrieval_seconds": retrieval_seconds,
                    "initial_series": retrieved["initial_series"],
                    "refined_series": retrieved["refined_series"],
                    "candidate_answers": retrieved["candidate_answers"],
                    "retrieved_documents": [item[3]["doc_id"] for item in retrieved["results"]],
                    "evidence_used": used,
                }
            )
        except Exception as error:
            record.update(
                {
                    "prediction": None,
                    "correct": False,
                    "retrieval_seconds": 0.0,
                    "generation_seconds": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "error": repr(error),
                }
            )
        record["total_seconds"] = time.perf_counter() - start
        append_jsonl(output, record)
        print(f"{index + 1}/{len(questions)} | correct={record['correct']}")

    summary = summarise_qa(load_jsonl(output))
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
