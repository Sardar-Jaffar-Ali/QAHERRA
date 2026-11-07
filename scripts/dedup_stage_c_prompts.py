#!/usr/bin/env python
"""Stage C: top-10 evidence and prompt lengths before and after deduplication.

Retrieves the top ten deduplicated units for each of the 2,000 TeleQnA
questions, rebuilds the prompt both ways, and measures what deduplication
changes before any answer is generated:

* how many of the ten slots in the original top-10 hold repeated content
  (reported as 4.12 slots on average);
* the mean context length with and without deduplication (10,465 and 11,095
  characters, an increase of about 6%);
* how many questions keep an identical prompt, so their saved answer can be
  reused instead of being paid for again.

The script asserts that a question whose top-10 is unchanged also produces a
byte-identical prompt. That is the check which catches a text lookup drifting
from the index it belongs to.

No API calls. Stage A must have run first.

Usage
-----
    python scripts/dedup_stage_c_prompts.py --paths configs/paths.yaml
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import pickle
import sys
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
from bm25s.tokenization import Tokenized

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qaherra_repro.config import load_yaml, require_path  # noqa: E402
from qaherra_repro.data import load_teleqna  # noqa: E402
from qaherra_repro.dedup import duplicate_slots  # noqa: E402

SHARD_SIZE = 50_000
BUDGET = 14_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="configs/paths.yaml")
    return parser.parse_args()


def to_arrays(texts: list[str], vocabulary: dict[str, int]) -> list[np.ndarray]:
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


def main() -> None:
    args = parse_args()
    paths = load_yaml(args.paths)
    out_dir = Path(require_path(paths, "dedup_output_dir"))
    work_dir = Path(require_path(paths, "dedup_work_dir"))

    meta = pickle.load(open(work_dir / "meta.pkl", "rb"))
    shards = meta["shards"]
    kept = np.load(out_dir / "kept_positions.npy")
    representative_of = np.load(out_dir / "representative_of.npy")
    if len(kept) != 740_002 or len(representative_of) != 2_396_136:
        raise SystemExit(
            f"Stage A outputs have unexpected sizes: {len(kept):,} retained, "
            f"{len(representative_of):,} original"
        )

    vocabulary: dict[str, int] = {}
    ids: list[np.ndarray] = []
    for shard in range(shards):
        ids += to_arrays(
            pickle.load(open(work_dir / f"texts_{shard:03d}.pkl", "rb")), vocabulary
        )
        gc.collect()
    index = bm25s.BM25()
    index.index(Tokenized(ids=ids, vocab=vocabulary), show_progress=False)
    del ids
    gc.collect()
    print(f"indexed {len(kept):,} deduplicated units | vocabulary {len(vocabulary):,}")

    questions = load_teleqna(require_path(paths, "teleqna")).reset_index(drop=True)
    original = (
        pd.read_csv(require_path(paths, "teleqna_bm25_units_results"))
        .sort_values("idx")
        .reset_index(drop=True)
    )
    if not (
        len(questions) == len(original) == 2000
        and original.idx.astype(int).tolist() == questions["index"].astype(int).tolist()
    ):
        raise SystemExit("question alignment with the non-deduplicated run failed")

    top, _ = index.retrieve(
        bm25s.tokenize(questions.question.tolist(), stopwords="en", show_progress=False),
        k=10,
        show_progress=False,
    )
    del index
    gc.collect()

    new_positions = [[int(kept[int(i)]) for i in row] for row in top]
    old_positions = [
        [int(value) for value in ast.literal_eval(cell)] for cell in original.final_top10
    ]

    needed_representatives = {
        int(representative_of[p]) for lists in (old_positions, new_positions) for L in lists for p in L
    }
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

    needed_positions = {
        p for lists in (old_positions, new_positions) for L in lists for p in L
    }
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
        """The original context builder, unchanged."""
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

    rows = []
    for position, (old, new) in enumerate(zip(old_positions, new_positions)):
        old_context, new_context = context(old), context(new)
        unchanged = old == new
        if unchanged and old_context != new_context:
            raise SystemExit(
                f"question {position}: identical top-10 but different prompt text"
            )
        if unchanged:
            category = "identical"
        elif set(old) == set(new):
            category = "same units, reordered"
        else:
            category = "different units"
        rows.append(
            {
                "index": int(questions["index"][position]),
                "category": category,
                "reusable": unchanged,
                "dup_slots_in_original_top10": duplicate_slots(old, representative_of),
                "orig_ctx_chars": len(old_context),
                "dedup_ctx_chars": len(new_context),
                "dedup_top10_original_positions": json.dumps(new),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "teleqna_prompt_comparison.csv", index=False)
    counts = frame.category.value_counts().to_dict()
    print("== TeleQnA prompt comparison (2,000 questions) ==")
    print(f"identical prompts, saved answer reusable: {int(frame.reusable.sum())}")
    print(f"changed prompts, an API call is needed:    {int((~frame.reusable).sum())}")
    print(f"breakdown: {counts}")
    print(
        "original top-10 lists containing duplicates: "
        f"{int((frame.dup_slots_in_original_top10 > 0).sum())} "
        f"| mean duplicate slots per top-10: "
        f"{frame.dup_slots_in_original_top10.mean():.4f}"
    )
    print(
        f"mean context characters: original {frame.orig_ctx_chars.mean():.0f} "
        f"| deduplicated {frame.dedup_ctx_chars.mean():.0f} "
        f"({100 * (frame.dedup_ctx_chars.mean() / frame.orig_ctx_chars.mean() - 1):.1f}%)"
    )


if __name__ == "__main__":
    main()
