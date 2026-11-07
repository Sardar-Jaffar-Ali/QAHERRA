#!/usr/bin/env python
"""Stage A: deduplicate the evidence units and record the mapping.

Streams the full evidence CSV in corpus order, applies the exact rule in
``qaherra_repro.dedup`` (SHA-1 of raw content, same specification series, first
occurrence kept), and writes:

* ``kept_positions.npy`` -- original position of every retained unit;
* ``representative_of.npy`` -- for every original position, the retained unit
  that represents it. This is the published deduplication mapping, and it is
  what keeps the specification, release, and section of every removed copy
  recoverable;
* ``corpus_sizes.csv`` -- the figures reported in the paper, including the
  share of units that are exact duplicates;
* text shards of the retained units under the work directory, so that later
  stages can index them without holding the collection in memory.

Expected outcome on the released corpus: 2,396,136 original units, 1,656,134
removed, 740,002 retained, 69.1% duplicates.

No API calls. Roughly six minutes and about 12 GB of RAM on the released CSV.

Usage
-----
    python scripts/dedup_stage_a_build.py --paths configs/paths.yaml
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qaherra_repro.config import load_yaml, require_path  # noqa: E402
from qaherra_repro.dedup import build_duplicate_map  # noqa: E402

SHARD_SIZE = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qaherra.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument(
        "--expect-units",
        type=int,
        default=None,
        help="fail if the CSV does not hold exactly this many units "
        "(defaults to corpus.expected_evidence_units in the config)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    paths = load_yaml(args.paths)
    expected = args.expect_units or int(config["corpus"]["expected_evidence_units"])

    out_dir = Path(require_path(paths, "dedup_output_dir"))
    work_dir = Path(require_path(paths, "dedup_work_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    shard: list[str] = []
    shard_count = 0
    doc_ids: list[str] = []
    section_paths: list[str] = []
    current_section = [""]

    def records():
        """Yield the deduplication tuple while tracking the current section."""
        columns = ("doc_id", "evidence_type", "text", "section_path")
        for frame in pd.read_csv(
            require_path(paths, "evidence_csv"),
            chunksize=200_000,
            usecols=list(columns),
        ):
            for doc_id, kind, text, section in zip(
                *(frame[column].astype(str) for column in columns)
            ):
                current_section[0] = section
                yield doc_id, kind, text

    def on_kept(index: int, doc_id: object, text: object) -> None:
        nonlocal shard, shard_count
        doc_ids.append(str(doc_id))
        section_paths.append(current_section[0])
        shard.append(str(text))
        if len(shard) == SHARD_SIZE:
            pickle.dump(shard, open(work_dir / f"texts_{shard_count:03d}.pkl", "wb"))
            shard_count += 1
            shard = []

    mapping = build_duplicate_map(
        records(), expected_units=expected, on_kept=on_kept
    )

    if shard:
        pickle.dump(shard, open(work_dir / f"texts_{shard_count:03d}.pkl", "wb"))
        shard_count += 1

    np.save(out_dir / "kept_positions.npy", mapping.kept_positions)
    np.save(out_dir / "representative_of.npy", mapping.representative_of)
    pickle.dump(
        {"doc_ids": doc_ids, "section_paths": section_paths, "shards": shard_count},
        open(work_dir / "meta.pkl", "wb"),
    )
    sizes = mapping.sizes() | {"text_shards": shard_count}
    pd.Series(sizes).to_csv(out_dir / "corpus_sizes.csv", header=False)
    for key, value in sizes.items():
        print(f"{key:>22}: {value:,}" if isinstance(value, int) else f"{key:>22}: {value}")


if __name__ == "__main__":
    main()
