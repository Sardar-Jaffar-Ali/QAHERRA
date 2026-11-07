"""Crash-safe QA logging, aggregation, and paired retrieval statistics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests


def append_jsonl(path: str | Path, record: Mapping[str, object]) -> None:
    """Persist one completed question immediately and flush it to disk."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: str | Path) -> list[dict[str, object]]:
    """Load valid completed records; tolerate a partial final crash line."""
    source = Path(path)
    if not source.exists():
        return []
    records: list[dict[str, object]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def summarise_qa(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate accuracy, latency, token usage, errors, and difficulty groups."""
    frame = pd.DataFrame(records)
    if frame.empty:
        return {"questions": 0, "accuracy": 0.0}

    correct = frame["correct"].fillna(False).astype(bool)
    errors = frame.get("error", pd.Series([None] * len(frame))).notna()
    summary: dict[str, object] = {
        "questions": int(len(frame)),
        "answered": int(frame["prediction"].notna().sum()),
        "correct": int(correct.sum()),
        "accuracy_percent": float(100.0 * correct.mean()),
        "errors_or_unanswered": int((errors | frame["prediction"].isna()).sum()),
        "average_retrieval_seconds": float(frame["retrieval_seconds"].mean()),
        "average_generation_seconds": float(frame["generation_seconds"].mean()),
        "average_total_seconds": float(frame["total_seconds"].mean()),
        "prompt_tokens": int(frame["prompt_tokens"].sum()),
        "completion_tokens": int(frame["completion_tokens"].sum()),
        "total_tokens": int(frame["total_tokens"].sum()),
    }
    if "difficulty" in frame:
        grouped = frame.assign(correct=correct).groupby("difficulty", dropna=False)
        summary["difficulty"] = {
            str(name): {
                "questions": int(len(group)),
                "correct": int(group["correct"].sum()),
                "accuracy_percent": float(100.0 * group["correct"].mean()),
            }
            for name, group in grouped
        }
    return summary


def exact_mcnemar(left: np.ndarray, right: np.ndarray) -> tuple[int, int, float]:
    """Two-sided exact McNemar test for paired binary retrieval outcomes."""
    left_only = int(np.sum(left & ~right))
    right_only = int(np.sum(~left & right))
    discordant = left_only + right_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(min(left_only, right_only), discordant, 0.5).pvalue
    )
    return left_only, right_only, p_value


def holm_correct(p_values: Iterable[float]) -> np.ndarray:
    """Apply Holm family-wise correction across all evaluated cutoffs."""
    return np.asarray(multipletests(list(p_values), method="holm")[1], dtype=float)

