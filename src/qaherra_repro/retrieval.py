"""Lexical, dense, table-aware, and rank-fusion retrieval components."""

from __future__ import annotations

import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import bm25s
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class BM25Retriever:
    """Memory-mapped BM25S index wrapper."""

    def __init__(self, index_path: str | Path, depth: int = 100) -> None:
        self.depth = int(depth)
        self.index = bm25s.BM25.load(str(index_path), load_corpus=False, mmap=True)

    def search(self, question: str) -> tuple[np.ndarray, np.ndarray, float]:
        start = time.perf_counter()
        tokens = bm25s.tokenize([question], stopwords="en")
        indices, scores = self.index.retrieve(tokens, k=self.depth)
        return (
            np.asarray(indices[0], dtype=np.int64),
            np.asarray(scores[0], dtype=np.float32),
            time.perf_counter() - start,
        )


class DenseRetriever:
    """Sentence-Transformer query encoder with a normalized FAISS index."""

    def __init__(
        self,
        model_path: str | Path,
        index_path: str | Path,
        depth: int = 100,
        device: str | None = None,
    ) -> None:
        self.depth = int(depth)
        self.model = SentenceTransformer(str(model_path), device=device)
        self.index = faiss.read_index(str(index_path))

    def search(self, question: str) -> tuple[np.ndarray, np.ndarray, float]:
        start = time.perf_counter()
        vector = self.model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        scores, indices = self.index.search(vector, self.depth)
        return (
            np.asarray(indices[0], dtype=np.int64),
            np.asarray(scores[0], dtype=np.float32),
            time.perf_counter() - start,
        )


class KeywordTableGate:
    """Original query-aware detector used for the reported QAHERRA QA runs."""

    def __init__(self, patterns: Sequence[str]) -> None:
        # Word boundaries reduce accidental substring matches such as "bit" in
        # an unrelated longer word.
        self.patterns = [re.compile(rf"\b{re.escape(item)}\b", re.I) for item in patterns]

    def __call__(self, question: str) -> bool:
        return any(pattern.search(question) for pattern in self.patterns)


def boost_table_candidates(
    question: str,
    indices: np.ndarray,
    scores: np.ndarray,
    table_mask: np.ndarray,
    gate: KeywordTableGate,
    boost: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Add a fixed score only to retrieved table units when the gate activates."""
    adjusted = np.asarray(scores, dtype=np.float32).copy()
    activated = gate(question)
    if activated:
        valid = indices >= 0
        selected_tables = np.zeros(len(indices), dtype=bool)
        selected_tables[valid] = table_mask[indices[valid]]
        adjusted[selected_tables] += float(boost)
    order = np.argsort(-adjusted, kind="stable")
    return indices[order], adjusted[order], activated


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    weights: Sequence[float],
    constant: float,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse rankings using weighted Reciprocal Rank Fusion."""
    if len(rankings) != len(weights):
        raise ValueError("Each ranking requires exactly one fusion weight")

    fused: defaultdict[int, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, raw_index in enumerate(ranking, start=1):
            index = int(raw_index)
            if index >= 0:
                fused[index] += float(weight) / (float(constant) + rank)

    ordered = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
    indices = np.full(limit, -1, dtype=np.int64)
    scores = np.zeros(limit, dtype=np.float32)
    for position, (index, score) in enumerate(ordered):
        indices[position] = index
        scores[position] = score
    return indices, scores

