"""Original-router Telco-RAG adapter for the standardized R8--R19 indexes."""

from __future__ import annotations

import os
import pickle
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer


class OriginalTelcoRouter:
    """Call the neural router and glossary code from the original repository."""

    def __init__(self, telcorag_root: str | Path) -> None:
        root = Path(telcorag_root).resolve()
        os.environ["TELCORAG_ROOT"] = str(root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        # Import only after TELCORAG_ROOT is set because the original code uses
        # it to locate the neural-router checkpoint and series descriptions.
        from src.query import Query  # type: ignore

        self.query_class = Query

    def route(self, question: str, refinement: str | None = None) -> tuple[str, list[int]]:
        query = self.query_class(question, [])
        query.def_TA_question()
        if refinement:
            query.enhanced_query = f"{query.query}\n{refinement}"
        query.predict_wg()
        return str(query.query), [int(value) for value in query.wg]


class SeriesIndexStore:
    """Search prebuilt per-series FAISS indexes without building them per query."""

    def __init__(
        self,
        index_dir: str | Path,
        metadata_db: str | Path,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.metadata_db = Path(metadata_db)
        self.encoder = SentenceTransformer(encoder_name, device=device)
        self._indexes: dict[int, faiss.Index] = {}
        self._rows: dict[int, dict[int, dict[str, str]]] = {}

    def _load_series(self, series: int) -> None:
        if series in self._indexes:
            return
        index_path = self.index_dir / f"series_{series}.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing series index: {index_path}")
        self._indexes[series] = faiss.read_index(str(index_path))

        connection = sqlite3.connect(self.metadata_db)
        try:
            rows = connection.execute(
                """
                SELECT f.row_id, c.text, c.release, c.source
                FROM faiss_rows AS f
                JOIN chunks AS c
                  ON c.series = f.series AND c.text_hash = f.text_hash
                WHERE f.series = ?
                ORDER BY f.row_id
                """,
                (series,),
            ).fetchall()
        finally:
            connection.close()
        self._rows[series] = {
            int(row_id): {
                "doc_id": str(source),
                "section_path": "",
                "evidence_type": "text",
                "text": str(text),
                "release": str(release),
                "series": str(series),
            }
            for row_id, text, release, source in rows
        }

    def search(self, query: str, series_numbers: Iterable[int], per_series_k: int = 10):
        vector = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        candidates: list[tuple[float, int, int, dict[str, str]]] = []
        for series in series_numbers:
            series = int(series)
            self._load_series(series)
            scores, row_ids = self._indexes[series].search(vector, per_series_k)
            for score, row_id in zip(scores[0], row_ids[0]):
                row_id = int(row_id)
                if row_id >= 0 and row_id in self._rows[series]:
                    candidates.append((float(score), series, row_id, self._rows[series][row_id]))

        candidates.sort(key=lambda value: (-value[0], value[1], value[2]))
        return candidates


class FlatIndexStore:
    """Search the dedicated Q-small four-document FAISS index."""

    def __init__(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        self.index = faiss.read_index(str(index_path))
        self.encoder = SentenceTransformer(encoder_name, device=device)
        with Path(metadata_path).open("rb") as handle:
            raw_metadata = pickle.load(handle)
        if hasattr(raw_metadata, "to_dict"):
            raw_metadata = raw_metadata.to_dict("records")
        if isinstance(raw_metadata, dict):
            raw_metadata = [raw_metadata[key] for key in sorted(raw_metadata)]
        self.rows = [dict(row) for row in raw_metadata]
        if len(self.rows) != self.index.ntotal:
            raise ValueError(
                f"Four-document metadata/index mismatch: "
                f"{len(self.rows)} rows vs {self.index.ntotal} vectors"
            )

    def search(self, query: str, series_numbers: Iterable[int], per_series_k: int = 10):
        # `series_numbers` is intentionally ignored because every vector in this
        # index belongs to one of the four allowed source documents.
        vector = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        scores, row_ids = self.index.search(vector, per_series_k)
        candidates = []
        for score, row_id in zip(scores[0], row_ids[0]):
            row_id = int(row_id)
            if row_id < 0:
                continue
            raw = self.rows[row_id]
            row = {
                "doc_id": str(raw.get("source", raw.get("doc_id", ""))),
                "section_path": str(raw.get("section_path", "")),
                "evidence_type": "text",
                "text": str(raw.get("text", "")),
                "release": str(raw.get("release", "")),
                "series": str(raw.get("series", "")),
            }
            candidates.append((float(score), int(raw.get("series", 0) or 0), row_id, row))
        return candidates


class TelcoRAGAdapter:
    """Run glossary enhancement, two neural routings, and two dense retrievals."""

    def __init__(
        self,
        telcorag_root: str | Path,
        index_dir: str | Path,
        metadata_db: str | Path,
        four_doc_index: str | Path | None = None,
        four_doc_metadata: str | Path | None = None,
        generator_model: str = "gpt-4o-mini",
        device: str | None = None,
    ) -> None:
        self.router = OriginalTelcoRouter(telcorag_root)
        self.store = (
            FlatIndexStore(four_doc_index, four_doc_metadata, device=device)
            if four_doc_index is not None and four_doc_metadata is not None
            else SeriesIndexStore(index_dir, metadata_db, device=device)
        )
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in the environment")
        self.client = OpenAI(api_key=api_key)
        self.generator_model = generator_model

    def _candidate_answers(self, question: str, context: str) -> str:
        """Generate concise candidates used only for the second routing/search."""
        prompt = f"""Provide up to five concise possible answers to the question.
Use the retrieved context and do not include unsupported claims.

QUESTION:
{question}

CONTEXT:
{context}
"""
        response = self.client.chat.completions.create(
            model=self.generator_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()

    def retrieve(self, question: str, final_k: int = 10) -> dict[str, object]:
        enhanced_question, initial_series = self.router.route(question)
        initial = self.store.search(enhanced_question, initial_series, per_series_k=10)
        initial_context = "\n\n".join(candidate[3]["text"] for candidate in initial[:10])

        candidates = self._candidate_answers(question, initial_context)
        refined_question, refined_series = self.router.route(question, candidates)
        retrieval_query = f"{refined_question}\n{candidates}"
        final = self.store.search(retrieval_query, refined_series, per_series_k=10)[:final_k]
        return {
            "initial_series": initial_series,
            "refined_series": refined_series,
            "candidate_answers": candidates,
            "results": final,
        }
