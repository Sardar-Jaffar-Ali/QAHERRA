#!/usr/bin/env python
"""Fixed-size chunk baselines over the four source documents.

Reproduces the ``Chunks / BM25`` and ``Chunks / Dense`` rows of Table III in the
four-document column, and the left-hand pair of Figure 3.

The chunk text is read from the four-document index metadata rather than looked
up in the corpus-wide chunk store; ``docs/CHUNK_LOOKUP_FIX.md`` explains why.
The run therefore starts by asserting that SHA-1 of every chunk's text equals
the ``text_hash`` recorded in that metadata, and refuses to generate anything
if it does not.

Both retrievers use the settings of the evidence-unit runs in the same scope:
the unadapted ``all-MiniLM-L6-v2`` encoder, ``bm25s`` with English stop words
and no stemming, top-10 evidence, and the 14,000-character context budget. The
encoder is deliberately the unadapted one, so that the comparison between the
two representations is unaffected by domain adaptation.

Results are appended to a JSONL file after every question. A failed API call is
written to an error file and never counted as an incorrect answer; rerunning
the script retries only the questions that have no successful record.

Usage
-----
    export OPENAI_API_KEY=...
    python scripts/run_chunk_baseline_4doc.py --retriever bm25  --paths configs/paths.yaml
    python scripts/run_chunk_baseline_4doc.py --retriever dense --paths configs/paths.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qaherra_repro.chunks import (  # noqa: E402
    build_context,
    cross_check_bm25_metadata,
    load_four_doc_chunks,
    verify_chunk_metadata,
)
from qaherra_repro.config import load_yaml, require_path, set_seed  # noqa: E402
from qaherra_repro.data import load_qsmall  # noqa: E402
from qaherra_repro.generation import (  # noqa: E402
    OpenAIAnswerGenerator,
    build_answer_prompt,
    parse_answer,
)

LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever", choices=["bm25", "dense"], required=True)
    parser.add_argument("--config", default="configs/qaherra.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def build_retriever(name: str, texts: list[str], top_k: int, encoder_name: str):
    """Return a function mapping a question to a list of chunk positions."""
    if name == "bm25":
        import bm25s

        index = bm25s.BM25()
        index.index(bm25s.tokenize(texts, stopwords="en"))

        def retrieve(question: str) -> list[int]:
            ids, _ = index.retrieve(
                bm25s.tokenize([question], stopwords="en"), k=top_k
            )
            return [int(value) for value in ids[0]]

        return retrieve

    import faiss
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(encoder_name)
    embeddings = encoder.encode(
        texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
    ).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    def retrieve(question: str) -> list[int]:
        vector = encoder.encode([question], normalize_embeddings=True).astype(np.float32)
        _, ids = index.search(vector, top_k)
        return [int(value) for value in ids[0]]

    return retrieve


def completed_records(path: Path) -> dict[int, dict]:
    """Return the successful records already on disk, keyed by question index."""
    if not path.exists():
        return {}
    done: dict[int, dict] = {}
    for line in open(path):
        record = json.loads(line)
        if "error" not in record:
            done[int(record["index"])] = record
    return done


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    paths = load_yaml(args.paths)
    set_seed(int(config.get("seed", 42)))

    budget = int(config["generation"]["max_context_characters"])
    encoder_name = config["dense"]["base_model"]
    out_dir = Path(require_path(paths, "output_dir")) / "chunks_4doc"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.retriever}_chunks_4doc"
    result_path = out_dir / f"{name}.jsonl"
    error_path = out_dir / f"{name}_errors.jsonl"

    chunks = load_four_doc_chunks(str(require_path(paths, "four_doc_chunk_metadata")))
    checks = verify_chunk_metadata(chunks)
    same_set = cross_check_bm25_metadata(
        chunks, str(require_path(paths, "four_doc_chunk_bm25_metadata"))
    )
    assert same_set, "the lexical and dense four-document indexes cover different chunks"
    print(
        f"chunks {int(checks['chunks'])} | sources {int(checks['distinct_sources'])} "
        f"| sha1(text) == text_hash {checks['hash_match']:.3f}"
    )

    questions = load_qsmall(require_path(paths, "qsmall"))
    retrieve = build_retriever(args.retriever, chunks.texts, args.top_k, encoder_name)

    rankings = {
        int(row["index"]): retrieve(row["question"]) for _, row in questions.iterrows()
    }
    json.dump(
        {str(key): value for key, value in rankings.items()},
        open(out_dir / f"{name}_rankings.json", "w"),
    )

    generator = OpenAIAnswerGenerator(
        config["generation"]["model"], config["generation"]["temperature"]
    )

    def work(row) -> None:
        index = int(row["index"])
        positions = rankings[index]
        context = build_context(chunks, positions, budget)
        try:
            raw = generator.answer(
                build_answer_prompt(row["question"], row["options"], context)
            )
            prediction = parse_answer(raw["raw_answer"], list(row["options"]))
            record = {
                "index": index,
                "gold": row["gold"],
                "prediction": prediction,
                "correct": prediction == row["gold"],
                "positions": positions,
                "sources": [chunks.sources[p] for p in positions],
                "context_chars": len(context),
                **raw,
            }
            target = result_path
        except Exception as error:  # a failed call is never scored
            record = {"index": index, "error": repr(error)}
            target = error_path
        with LOCK, open(target, "a") as handle:
            handle.write(json.dumps(record) + "\n")

    done = completed_records(result_path)
    todo = [row for _, row in questions.iterrows() if int(row["index"]) not in done]
    print(f"{len(done)} questions already answered, {len(todo)} to answer")
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, todo))

    final = completed_records(result_path)
    missing = len(questions) - len(final)
    frame = pd.DataFrame([final[key] for key in sorted(final)])
    frame.to_csv(out_dir / f"{name}.csv", index=False)
    inside = np.mean(
        [all(source in chunks.distinct_sources for source in row) for row in frame.sources]
    )
    print(
        f"{name}: answered {len(frame)}/{len(questions)} "
        f"| accuracy {100 * frame.correct.mean():.2f} "
        f"| all sources within the four documents {inside:.3f} "
        f"| mean context {frame.context_chars.mean():.0f} chars"
    )
    if missing:
        print(
            f"{missing} questions failed and are not scored. Rerun this script to "
            "retry only those questions."
        )


if __name__ == "__main__":
    main()
