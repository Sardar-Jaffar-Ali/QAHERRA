#!/usr/bin/env python
"""Run QAHERRA or a controlled baseline with per-question crash-safe saving."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from qaherra_repro.config import load_yaml, require_path, set_seed
from qaherra_repro.data import (
    CsvEvidenceStore,
    load_qsmall,
    load_table_mask,
    load_teleqna,
)
from qaherra_repro.evaluation import append_jsonl, load_jsonl, summarise_qa
from qaherra_repro.generation import (
    OpenAIAnswerGenerator,
    build_answer_prompt,
    build_context,
    parse_answer,
)
from qaherra_repro.pipeline import ControlledRAGPipeline
from qaherra_repro.reranking import CrossEncoderReranker
from qaherra_repro.retrieval import BM25Retriever, DenseRetriever, KeywordTableGate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["no_rag", "bm25", "tspec_llm", "chat3gpp", "qaherra_qonly", "qaherra_option"],
        required=True,
    )
    parser.add_argument("--dataset", choices=["qsmall", "teleqna"], required=True)
    parser.add_argument("--scope", choices=["four_doc", "full"], required=True)
    parser.add_argument("--config", default="configs/qaherra.yaml")
    parser.add_argument("--baseline-config", default="configs/baselines.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "teleqna" and args.scope != "full":
        raise ValueError("TeleQnA is evaluated only with full-corpus retrieval")

    config = load_yaml(args.config)
    baselines = load_yaml(args.baseline_config)
    paths = load_yaml(args.paths)
    set_seed(int(config["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    questions = (
        load_qsmall(require_path(paths, "qsmall"))
        if args.dataset == "qsmall"
        else load_teleqna(require_path(paths, "teleqna"))
    )

    if args.scope == "four_doc":
        evidence_csv = require_path(paths, "four_doc_evidence_csv")
        bm25_path = require_path(paths, "four_doc_bm25_index")
        base_faiss = require_path(paths, "four_doc_base_dense_faiss")
        domain_faiss = require_path(paths, "four_doc_domain_dense_faiss")
    else:
        evidence_csv = require_path(paths, "evidence_csv")
        bm25_path = require_path(paths, "bm25_index")
        base_faiss = require_path(paths, "base_dense_faiss")
        domain_faiss = require_path(paths, "dense_faiss")

    retrieval_cfg = config["retrieval"]
    rerank_cfg = config["reranking"]
    generation_cfg = config["generation"]

    method_settings = baselines.get(args.method, {})
    bm25_depth = int(method_settings.get("bm25_depth", retrieval_cfg["bm25_depth"]))
    dense_depth = int(method_settings.get("dense_depth", retrieval_cfg["dense_depth"]))

    bm25 = (
        BM25Retriever(bm25_path, depth=bm25_depth)
        if args.method != "no_rag"
        else None
    )
    dense = None
    if args.method not in {"no_rag", "bm25"}:
        if args.method.startswith("qaherra"):
            dense_model = require_path(paths, "domain_model")
            dense_index = domain_faiss
        else:
            dense_model = config["dense"]["base_model"]
            dense_index = base_faiss
        dense = DenseRetriever(
            dense_model,
            dense_index,
            depth=dense_depth,
            device=device,
        )

    evidence_store = CsvEvidenceStore(evidence_csv)
    table_mask = load_table_mask(evidence_csv) if args.method.startswith("qaherra") else None
    table_gate = (
        KeywordTableGate(config["table_gate"]["patterns"])
        if args.method.startswith("qaherra")
        else None
    )
    reranker = (
        CrossEncoderReranker(
            rerank_cfg["model"],
            max_length=int(rerank_cfg["max_length"]),
            batch_size=int(rerank_cfg["batch_size"]),
            final_count=int(rerank_cfg["final_evidence_count"]),
            device=device,
        )
        if args.method.startswith("qaherra")
        else None
    )

    pipeline = ControlledRAGPipeline(
        method=args.method,
        evidence_store=evidence_store,
        bm25=bm25,
        dense=dense,
        reranker=reranker,
        table_mask=table_mask,
        table_gate=table_gate,
        qaherra_settings={
            "table_boost": retrieval_cfg["table_boost"],
            "bm25_weight": retrieval_cfg["bm25_weight"],
            "dense_weight": retrieval_cfg["dense_weight"],
            "rrf_constant": retrieval_cfg["rrf_constant"],
        },
        chat3gpp_settings=baselines["chat3gpp"],
        candidate_count=int(rerank_cfg["candidate_count"]),
        final_count=int(rerank_cfg["final_evidence_count"]),
    )
    generator = OpenAIAnswerGenerator(
        generation_cfg["model"],
        temperature=float(generation_cfg["temperature"]),
    )

    output_dir = Path(paths["output_dir"])
    output_path = Path(args.output) if args.output else (
        output_dir / f"{args.method}_{args.dataset}_{args.scope}.jsonl"
    )
    completed = load_jsonl(output_path)
    completed_indices = {int(record["index"]) for record in completed}
    print(f"Resuming from {len(completed_indices)}/{len(questions)} completed questions")

    for _, row in questions.iterrows():
        index = int(row["index"])
        if index in completed_indices:
            continue

        start = time.perf_counter()
        record = {
            "index": index,
            "id": row["id"],
            "dataset": args.dataset,
            "scope": args.scope,
            "method": args.method,
            "question": row["question"],
            "options": row["options"],
            "gold": row["gold"],
            "difficulty": row.get("difficulty", "Not provided"),
            "error": None,
        }
        try:
            retrieval = pipeline.retrieve(row["question"], row["options"])
            evidence = evidence_store.fetch(retrieval.indices)
            context, used_count = build_context(
                retrieval.indices,
                evidence,
                int(generation_cfg["max_context_characters"]),
            )
            prompt = build_answer_prompt(row["question"], row["options"], context)
            generation = generator.answer(prompt)
            prediction = parse_answer(generation["raw_answer"], list(row["options"]))

            record.update(generation)
            record.update(
                {
                    "prediction": prediction,
                    "correct": prediction == row["gold"],
                    "retrieved_indices": [int(value) for value in retrieval.indices],
                    "retrieved_documents": [
                        evidence[int(value)]["doc_id"]
                        for value in retrieval.indices
                        if int(value) >= 0
                    ],
                    "evidence_used": used_count,
                    "retrieval_seconds": retrieval.retrieval_seconds,
                    "bm25_seconds": retrieval.bm25_seconds,
                    "dense_seconds": retrieval.dense_seconds,
                    "reranking_seconds": retrieval.reranking_seconds,
                    "table_gate_activated": retrieval.table_gate_activated,
                }
            )
        except Exception as error:  # Save failures instead of losing prior work.
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
        append_jsonl(output_path, record)
        print(
            f"{index + 1}/{len(questions)} | prediction={record['prediction']} "
            f"| gold={record['gold']} | correct={record['correct']}"
        )

    records = load_jsonl(output_path)
    summary = summarise_qa(records)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Results:", output_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
