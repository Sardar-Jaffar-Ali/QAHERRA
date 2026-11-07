"""Dataset parsers and memory-conscious evidence access."""

from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from pathlib import Path
from typing import Iterable

import json5
import numpy as np
import pandas as pd


LABELS = list("ABCDEFGH")


def _normalise_question(value: object) -> str:
    """Collapse whitespace without altering technical symbols."""
    return re.sub(r"\s+", " ", str(value)).strip()


def load_qsmall(path: str | Path) -> pd.DataFrame:
    """Load the 100-question Q-small JSON, including its non-strict JSON form."""
    raw = Path(path).read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = json5.loads(raw)

    records = list(payload.values()) if isinstance(payload, dict) else payload
    if len(records) != 100:
        raise ValueError(f"Expected 100 Q-small questions, found {len(records)}")

    rows: list[dict[str, object]] = []
    for position, record in enumerate(records):
        options: dict[str, str] = {}
        for option_number, label in enumerate(LABELS, start=1):
            key = f"option_{option_number}"
            if key not in record:
                break
            options[label] = str(record[key])

        match = re.search(r"option_(\d+)", str(record.get("answer", "")).lower())
        if not match:
            raise ValueError(f"Cannot parse Q-small answer at row {position}")

        rows.append(
            {
                "index": position,
                "id": f"question_{position + 1}",
                "question": _normalise_question(record["question"]),
                "options": options,
                "gold": LABELS[int(match.group(1)) - 1],
                "difficulty": str(record.get("difficulty", "Unknown")),
                "category": str(record.get("category", "")),
            }
        )
    return pd.DataFrame(rows)


def parse_teleqna_choices(value: object) -> list[str]:
    """Parse TeleQnA choices stored as Python lists or NumPy-style arrays."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(item).strip().strip("'\"") for item in value]

    text = str(value).strip()

    # The released CSV contains NumPy-style arrays whose quoted strings are
    # adjacent rather than comma separated. Python's tokenizer extracts every
    # string literal correctly, including nested quote characters.
    choices: list[str] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.STRING:
                choices.append(str(ast.literal_eval(token.string)))
    except (tokenize.TokenError, ValueError, SyntaxError):
        choices = []
    if choices:
        return choices

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, np.ndarray)):
            return [str(item) for item in parsed]
    except (ValueError, SyntaxError):
        pass
    raise ValueError(f"Cannot parse TeleQnA choices: {text[:120]}")


def load_teleqna(path: str | Path) -> pd.DataFrame:
    """Load the fixed 2,000-question standards subset used by the experiments."""
    frame = pd.read_csv(path)
    if len(frame) != 2000:
        raise ValueError(f"Expected 2,000 TeleQnA questions, found {len(frame)}")

    rows: list[dict[str, object]] = []
    for position, row in frame.iterrows():
        choices = parse_teleqna_choices(row["choices"])
        options = {LABELS[i]: choice for i, choice in enumerate(choices)}
        gold_index = int(row["answer"])
        if not 0 <= gold_index < len(options):
            raise ValueError(
                f"Invalid TeleQnA answer at row {position}: "
                f"answer={gold_index}, choices={len(options)}"
            )
        rows.append(
            {
                "index": int(position),
                "id": f"teleqna_{position}",
                "question": _normalise_question(row["question"]),
                "options": options,
                "gold": LABELS[gold_index],
                "difficulty": "Not provided",
                "subject": str(row.get("subject", "")),
            }
        )
    return pd.DataFrame(rows)


class CsvEvidenceStore:
    """Fetch selected evidence rows without retaining the multi-gigabyte CSV."""

    def __init__(self, path: str | Path, chunksize: int = 100_000) -> None:
        self.path = Path(path)
        self.chunksize = chunksize
        self._cache: dict[int, dict[str, str]] = {}

    def fetch(self, indices: Iterable[int]) -> dict[int, dict[str, str]]:
        """Return metadata for requested zero-based CSV row indices."""
        requested = {int(index) for index in indices if int(index) >= 0}
        missing = requested.difference(self._cache)
        if not missing:
            return {index: self._cache[index] for index in requested}

        start = 0
        columns = ["doc_id", "evidence_type", "section_path", "text"]
        for chunk in pd.read_csv(
            self.path,
            encoding="utf-8-sig",
            usecols=columns,
            chunksize=self.chunksize,
        ):
            stop = start + len(chunk)
            local_indices = [index for index in missing if start <= index < stop]
            for index in local_indices:
                row = chunk.iloc[index - start]
                self._cache[index] = {
                    column: "" if pd.isna(row[column]) else str(row[column])
                    for column in columns
                }
            missing.difference_update(local_indices)
            if not missing:
                break
            start = stop

        if missing:
            raise IndexError(f"Evidence rows not found: {sorted(missing)[:10]}")
        return {index: self._cache[index] for index in requested}


def load_table_mask(evidence_csv: str | Path) -> np.ndarray:
    """Load only the evidence-type column needed by QAHERRA's table boost."""
    frame = pd.read_csv(
        evidence_csv,
        encoding="utf-8-sig",
        usecols=["evidence_type"],
    )
    return (
        frame["evidence_type"]
        .fillna("")
        .astype(str)
        .str.lower()
        .eq("table")
        .to_numpy(dtype=bool)
    )
