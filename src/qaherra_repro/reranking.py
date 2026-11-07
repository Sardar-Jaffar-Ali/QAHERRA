"""Question-only and option-conditioned cross-encoder reranking."""

from __future__ import annotations

import time
from typing import Mapping, Sequence

import numpy as np
from sentence_transformers import CrossEncoder


def format_options(options: Mapping[str, str]) -> str:
    """Render options deterministically in label order."""
    return "\n".join(f"{label}) {text}" for label, text in options.items())


def option_conditioned_query(question: str, options: Mapping[str, str]) -> str:
    """Introduce options only at reranking, never at initial retrieval."""
    compact = " ".join(f"{label}: {text}" for label, text in options.items())
    return f"{question}\nPossible answers: {compact}"


class CrossEncoderReranker:
    """Rerank a fixed candidate pool using one cross encoder."""

    def __init__(
        self,
        model_name: str,
        max_length: int,
        batch_size: int,
        final_count: int,
        device: str | None = None,
    ) -> None:
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        self.batch_size = int(batch_size)
        self.final_count = int(final_count)

    def rerank(
        self,
        question: str,
        options: Mapping[str, str],
        candidate_indices: Sequence[int],
        evidence: Mapping[int, Mapping[str, str]],
        option_conditioned: bool,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        candidates = [int(index) for index in candidate_indices if int(index) >= 0]
        query = (
            option_conditioned_query(question, options)
            if option_conditioned
            else question
        )

        pairs = []
        for index in candidates:
            row = evidence[index]
            passage = f"Document: {row['doc_id']}\nSection: {row['section_path']}\n{row['text']}"
            pairs.append((query, passage))

        start = time.perf_counter()
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        elapsed = time.perf_counter() - start

        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        order = np.argsort(-scores, kind="stable")[: self.final_count]
        return (
            np.asarray([candidates[position] for position in order], dtype=np.int64),
            scores[order],
            elapsed,
        )

