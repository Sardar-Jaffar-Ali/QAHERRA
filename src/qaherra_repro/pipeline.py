"""Method dispatcher for controlled retrieval and reranking comparisons."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .data import CsvEvidenceStore
from .reranking import CrossEncoderReranker
from .retrieval import (
    BM25Retriever,
    DenseRetriever,
    KeywordTableGate,
    boost_table_candidates,
    reciprocal_rank_fusion,
)


@dataclass
class RetrievalResult:
    """Indices, scores, timings, and gate state produced for one question."""

    indices: np.ndarray
    scores: np.ndarray
    retrieval_seconds: float
    bm25_seconds: float = 0.0
    dense_seconds: float = 0.0
    reranking_seconds: float = 0.0
    table_gate_activated: bool = False


class ControlledRAGPipeline:
    """Run the shared corpus through one locked method configuration."""

    VALID_METHODS = {
        "no_rag",
        "bm25",
        "tspec_llm",
        "chat3gpp",
        "qaherra_qonly",
        "qaherra_option",
    }

    def __init__(
        self,
        method: str,
        evidence_store: CsvEvidenceStore,
        bm25: BM25Retriever | None = None,
        dense: DenseRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        table_mask: np.ndarray | None = None,
        table_gate: KeywordTableGate | None = None,
        qaherra_settings: Mapping[str, float] | None = None,
        chat3gpp_settings: Mapping[str, float] | None = None,
        candidate_count: int = 20,
        final_count: int = 10,
    ) -> None:
        if method not in self.VALID_METHODS:
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.evidence_store = evidence_store
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.table_mask = table_mask
        self.table_gate = table_gate
        self.qaherra = dict(qaherra_settings or {})
        self.chat3gpp = dict(chat3gpp_settings or {})
        self.candidate_count = int(candidate_count)
        self.final_count = int(final_count)

    def retrieve(self, question: str, options: Mapping[str, str]) -> RetrievalResult:
        """Retrieve evidence using only signals allowed by the selected method."""
        start = time.perf_counter()

        if self.method == "no_rag":
            return RetrievalResult(
                np.asarray([], dtype=np.int64),
                np.asarray([], dtype=np.float32),
                time.perf_counter() - start,
            )

        if self.method == "bm25":
            if self.bm25 is None:
                raise RuntimeError("BM25 retriever is required")
            indices, scores, bm25_seconds = self.bm25.search(question)
            return RetrievalResult(
                indices[: self.final_count],
                scores[: self.final_count],
                time.perf_counter() - start,
                bm25_seconds=bm25_seconds,
            )

        if self.method == "tspec_llm":
            if self.dense is None:
                raise RuntimeError("Dense retriever is required")
            indices, scores, dense_seconds = self.dense.search(question)
            return RetrievalResult(
                indices[: self.final_count],
                scores[: self.final_count],
                time.perf_counter() - start,
                dense_seconds=dense_seconds,
            )

        if self.bm25 is None or self.dense is None:
            raise RuntimeError("Both BM25 and dense retrievers are required")

        bm_indices, _, bm25_seconds = self.bm25.search(question)
        dense_indices, dense_scores, dense_seconds = self.dense.search(question)

        if self.method == "chat3gpp":
            # Chat3GPP uses one global lexical and one global dense search.
            # Options are deliberately absent from both retrieval queries.
            indices, scores = reciprocal_rank_fusion(
                [bm_indices, dense_indices],
                [
                    float(self.chat3gpp.get("bm25_weight", 1.0)),
                    float(self.chat3gpp.get("dense_weight", 1.0)),
                ],
                constant=float(self.chat3gpp.get("rrf_constant", 60.0)),
                limit=self.final_count,
            )
            return RetrievalResult(
                indices,
                scores,
                time.perf_counter() - start,
                bm25_seconds=bm25_seconds,
                dense_seconds=dense_seconds,
            )

        if self.table_mask is None or self.table_gate is None or self.reranker is None:
            raise RuntimeError("QAHERRA requires the table mask, gate, and reranker")

        adjusted_indices, _, activated = boost_table_candidates(
            question,
            dense_indices,
            dense_scores,
            self.table_mask,
            self.table_gate,
            boost=float(self.qaherra.get("table_boost", 0.15)),
        )
        fused_indices, _ = reciprocal_rank_fusion(
            [bm_indices, adjusted_indices],
            [
                float(self.qaherra.get("bm25_weight", 1.0)),
                float(self.qaherra.get("dense_weight", 0.25)),
            ],
            constant=float(self.qaherra.get("rrf_constant", 5.0)),
            limit=max(self.candidate_count, self.final_count),
        )

        candidates = fused_indices[: self.candidate_count]
        evidence = self.evidence_store.fetch(candidates)
        final_indices, final_scores, reranking_seconds = self.reranker.rerank(
            question,
            options,
            candidates,
            evidence,
            option_conditioned=self.method == "qaherra_option",
        )
        return RetrievalResult(
            final_indices,
            final_scores,
            time.perf_counter() - start,
            bm25_seconds=bm25_seconds,
            dense_seconds=dense_seconds,
            reranking_seconds=reranking_seconds,
            table_gate_activated=activated,
        )
