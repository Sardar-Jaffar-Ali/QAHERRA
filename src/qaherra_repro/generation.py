"""Evidence-grounded multiple-choice answer generation and token logging."""

from __future__ import annotations

import os
import re
import time
from typing import Mapping, Sequence

from openai import OpenAI

from .reranking import format_options


def build_context(
    ranking: Sequence[int],
    evidence: Mapping[int, Mapping[str, str]],
    max_characters: int,
) -> tuple[str, int]:
    """Build one bounded context using the final evidence ranking."""
    blocks: list[str] = []
    characters = 0
    for rank, raw_index in enumerate(ranking, start=1):
        index = int(raw_index)
        if index < 0:
            continue
        row = evidence[index]
        block = (
            f"[Source {rank} | {row['doc_id']} | {row['section_path']}]\n"
            f"{row['text']}"
        )
        if characters + len(block) > max_characters:
            break
        blocks.append(block)
        characters += len(block) + 7
    return "\n\n---\n\n".join(blocks), len(blocks)


def build_answer_prompt(question: str, options: Mapping[str, str], context: str) -> str:
    """Return the exact controlled answer-selection prompt."""
    valid = ", ".join(options)
    return f"""You are answering a multiple-choice telecom standards question.
Use the SOURCES to choose the single best option.

Return ONLY the option letter ({valid}). No explanation.

SOURCES:
{context}

QUESTION:
{question}

OPTIONS:
{format_options(options)}

Answer (letter only):
"""


def parse_answer(raw: str | None, valid_labels: Sequence[str]) -> str | None:
    """Extract one valid option letter while rejecting ambiguous text."""
    if raw is None:
        return None
    text = str(raw).strip().upper()
    if text in valid_labels:
        return text
    match = re.search(r"\b([A-H])\b", text)
    return match.group(1) if match and match.group(1) in valid_labels else None


class OpenAIAnswerGenerator:
    """Call the locked generator and retain latency and complete token usage."""

    def __init__(self, model: str, temperature: float = 0.0) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in the environment before running QA")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = float(temperature)

    def answer(self, prompt: str) -> dict[str, object]:
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Follow the user prompt strictly."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )
        latency = time.perf_counter() - start
        usage = response.usage
        return {
            "raw_answer": (response.choices[0].message.content or "").strip(),
            "generation_seconds": latency,
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            "response_id": getattr(response, "id", None),
            "returned_model": getattr(response, "model", None),
        }

