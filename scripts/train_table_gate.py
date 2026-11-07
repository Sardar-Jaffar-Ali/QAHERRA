#!/usr/bin/env python
"""Train and evaluate the learned table gate with stratified 5-fold OOF testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from qaherra_repro.config import load_yaml, require_path, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="CSV containing question and final_reference_label")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_features(
    train_texts: list[str],
    test_texts: list[str],
    train_dense: np.ndarray,
    test_dense: np.ndarray,
):
    """Fit lexical features on training data only and append scaled dense features."""
    word = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=12_000,
    )
    character = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=18_000,
    )
    scaler = StandardScaler()
    train = hstack(
        [
            word.fit_transform(train_texts),
            character.fit_transform(train_texts),
            csr_matrix(scaler.fit_transform(train_dense)),
        ]
    ).tocsr()
    test = hstack(
        [
            word.transform(test_texts),
            character.transform(test_texts),
            csr_matrix(scaler.transform(test_dense)),
        ]
    ).tocsr()
    return train, test, word, character, scaler


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    paths = load_yaml(args.paths)
    frame = pd.read_csv(args.labels)
    texts = (
        frame["question"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip().tolist()
    )
    labels = frame["final_reference_label"].astype(int).to_numpy()
    if len(frame) != 300:
        raise ValueError(f"Expected 300 labelled questions, found {len(frame)}")

    encoder = SentenceTransformer(str(require_path(paths, "domain_model")))
    dense = encoder.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    probabilities = np.zeros(len(frame), dtype=np.float32)
    predictions = np.zeros(len(frame), dtype=np.int64)
    fold_numbers = np.zeros(len(frame), dtype=np.int64)

    for fold, (train_indices, test_indices) in enumerate(folds.split(texts, labels), start=1):
        train_texts = [texts[index] for index in train_indices]
        test_texts = [texts[index] for index in test_indices]
        train_features, test_features, _, _, _ = build_features(
            train_texts,
            test_texts,
            dense[train_indices],
            dense[test_indices],
        )
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=args.seed,
        )
        classifier.fit(train_features, labels[train_indices])
        fold_probabilities = classifier.predict_proba(test_features)[:, 1]
        probabilities[test_indices] = fold_probabilities
        predictions[test_indices] = (fold_probabilities >= 0.5).astype(int)
        fold_numbers[test_indices] = fold

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    summary = {
        "evaluation": "stratified_5_fold_out_of_fold",
        "questions": int(len(frame)),
        "table_relevant": int(labels.sum()),
        "non_table": int((labels == 0).sum()),
        "accuracy_percent": 100.0 * accuracy_score(labels, predictions),
        "precision_percent": 100.0 * precision_score(labels, predictions, zero_division=0),
        "recall_percent": 100.0 * recall_score(labels, predictions, zero_division=0),
        "f1_percent": 100.0 * f1_score(labels, predictions, zero_division=0),
        "specificity_percent": 100.0 * tn / (tn + fp),
        "roc_auc": roc_auc_score(labels, probabilities),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }

    # Train the deployment model on all labelled questions after the strictly
    # out-of-fold evaluation has been completed.
    all_features, _, word, character, scaler = build_features(texts, texts, dense, dense)
    final_classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=args.seed,
    )
    final_classifier.fit(all_features, labels)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_frame = frame.copy()
    result_frame["fold"] = fold_numbers
    result_frame["probability"] = probabilities
    result_frame["prediction"] = predictions
    result_frame.to_csv(output / "table_gate_oof_predictions.csv", index=False)
    joblib.dump(
        {
            "word_vectorizer": word,
            "char_vectorizer": character,
            "dense_scaler": scaler,
            "classifier": final_classifier,
            "threshold": 0.5,
            "seed": args.seed,
        },
        output / "learned_table_gate.joblib",
    )
    (output / "table_gate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

