#!/usr/bin/env python
"""Fail when common credentials or identifying local paths are committed."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".toml", ".json", ".csv"}
PATTERNS = {
    "OpenAI key": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "Windows user path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.I),
    "Google Drive private path": re.compile(r"/content/drive/MyDrive/(?!qaherra-reproducibility)"),
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
}


def main() -> None:
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        # The scanner necessarily contains the detection expressions themselves.
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{label}: {path.relative_to(ROOT)}")
    if findings:
        raise SystemExit("Security scan failed:\n" + "\n".join(findings))
    print("Security scan passed: no credential or identity patterns found.")


if __name__ == "__main__":
    main()
