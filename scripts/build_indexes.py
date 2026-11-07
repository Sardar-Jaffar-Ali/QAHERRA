#!/usr/bin/env python
"""Build BM25 and normalized FAISS indexes from QAHERRA evidence units."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import bm25s
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from qaherra_repro.config import load_yaml, require_path, set_seed
from qaherra_repro.telcodpr import normalise_document


FOUR_DOCS = {
    "36777-f00_1",
    "38811-f40",
    "38901-g10",
    "38821-g20",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qaherra.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--scope", choices=["full", "four_doc"], default="full")
    parser.add_argument("--encoder", choices=["base", "domain"], default="domain")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    paths = load_yaml(args.paths)
    set_seed(int(config["seed"]))

    evidence_path = require_path(paths, "evidence_csv")
    frame = pd.read_csv(evidence_path, encoding="utf-8-sig")

    if args.scope == "four_doc":
        keep = frame["doc_id"].map(normalise_document).isin(FOUR_DOCS)
        frame = frame.loc[keep].reset_index(drop=True)
        output_csv = Path(paths["four_doc_evidence_csv"])
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_csv, index=False, encoding="utf-8-sig")
        bm25_path = Path(paths["four_doc_bm25_index"])
        faiss_key = f"four_doc_{args.encoder}_dense_faiss"
    else:
        bm25_path = Path(paths["bm25_index"])
        faiss_key = "dense_faiss" if args.encoder == "domain" else "base_dense_faiss"

    texts = frame["text"].fillna("").astype(str).tolist()
    print(f"Evidence units selected: {len(texts):,}")

    # BM25S stores tokenized lexical statistics and supports memory-mapped load.
    if bm25_path.exists():
        shutil.rmtree(bm25_path)
    tokens = bm25s.tokenize(texts, stopwords="en")
    bm25 = bm25s.BM25()
    bm25.index(tokens)
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    bm25.save(str(bm25_path), corpus=None)
    print("BM25 saved:", bm25_path)

    model_path = (
        require_path(paths, "domain_model")
        if args.encoder == "domain"
        else config["dense"]["base_model"]
    )
    model = SentenceTransformer(str(model_path))
    dimension = int(config["dense"]["embedding_dimension"])
    index = faiss.IndexFlatIP(dimension)

    # Encode incrementally so that the complete embedding matrix is never held
    # in RAM. Normalization makes inner product equivalent to cosine similarity.
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start : start + args.batch_size]
        vectors = model.encode(
            batch,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        index.add(vectors)
        if index.ntotal % 100_000 < args.batch_size:
            print(f"Dense vectors indexed: {index.ntotal:,}/{len(texts):,}")

    faiss_path = Path(paths[faiss_key])
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))
    print("FAISS saved:", faiss_path)


if __name__ == "__main__":
    main()

