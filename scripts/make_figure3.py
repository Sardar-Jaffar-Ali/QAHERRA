#!/usr/bin/env python
"""Figure 3: BM25 accuracy over evidence units and fixed-size chunks by scope.

Draws the three paired bars the paper reports, with the Holm-corrected
significance bracket above each pair. The values come from
``results/qa_accuracy.csv`` and ``results/units_vs_chunks_tests.csv``, which
``scripts/verify_results.py --write-results`` regenerates from the per-question
files, so the figure cannot drift away from the numbers in the tables.

Usage
-----
    python scripts/make_figure3.py --out fig/fig3_representation_scope.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

#: Scope label and the comparison in ``units_vs_chunks_tests.csv`` it draws.
SCOPES = [
    ("Q-small\nfour documents", "BM25, Q-small, four documents"),
    ("Q-small\nfull corpus", "BM25, Q-small, full corpus"),
    ("TeleQnA\nfull corpus", "BM25, TeleQnA, full corpus"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="fig/fig3_representation_scope.pdf")
    parser.add_argument("--tests", default="results/units_vs_chunks_tests.csv")
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def label_for(p_holm: float, alpha: float) -> str:
    """Report a corrected p-value, or mark the comparison non-significant."""
    if p_holm >= alpha:
        return "n.s."
    if p_holm < 0.001:
        return "p < 0.001"
    return f"p = {p_holm:.3f}"


def main() -> None:
    args = parse_args()
    tests = pd.read_csv(ROOT / args.tests).set_index("name")

    units = [float(tests.loc[name, "accuracy_units"]) for _, name in SCOPES]
    chunks = [float(tests.loc[name, "accuracy_chunks"]) for _, name in SCOPES]
    brackets = [
        label_for(float(tests.loc[name, "p_holm"]), args.alpha) for _, name in SCOPES
    ]

    width = 0.36
    positions = range(len(SCOPES))
    figure, axes = plt.subplots(figsize=(6.2, 3.4))
    axes.bar(
        [p - width / 2 for p in positions],
        units,
        width,
        label="Evidence units",
        color="#2f5d8c",
        edgecolor="black",
        linewidth=0.5,
    )
    axes.bar(
        [p + width / 2 for p in positions],
        chunks,
        width,
        label="Fixed-size chunks",
        color="#c9c9c9",
        edgecolor="black",
        linewidth=0.5,
    )

    for position, (unit, chunk) in enumerate(zip(units, chunks)):
        axes.text(position - width / 2, unit + 1.0, f"{unit:.1f}", ha="center", fontsize=8)
        axes.text(position + width / 2, chunk + 1.0, f"{chunk:.1f}", ha="center", fontsize=8)

    top = max(max(units), max(chunks))
    for position, text in enumerate(brackets):
        height = max(units[position], chunks[position]) + 6.0
        axes.plot(
            [position - width / 2, position - width / 2, position + width / 2, position + width / 2],
            [height - 1.2, height, height, height - 1.2],
            color="black",
            linewidth=0.8,
        )
        axes.text(position, height + 0.8, text, ha="center", fontsize=8)

    axes.set_xticks(list(positions))
    axes.set_xticklabels([label for label, _ in SCOPES], fontsize=9)
    axes.set_ylabel("Accuracy (%)", fontsize=9)
    axes.set_ylim(0, top + 16)
    axes.legend(frameon=False, fontsize=8, loc="lower right")
    axes.spines[["top", "right"]].set_visible(False)
    axes.tick_params(labelsize=8)
    figure.tight_layout()

    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {destination} and {destination.with_suffix('.png')}")


if __name__ == "__main__":
    main()
