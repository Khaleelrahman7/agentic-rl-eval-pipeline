"""Linear-time answer and fenced-code parsing."""

from __future__ import annotations

import re

BRACE = re.compile(r"[{}]")
MARKER = "\\boxed{"
FENCE = "```python\n"
QUOTES = "'\""


def normalize_answer(value: str | None) -> str | None:
    """Strip surrounding whitespace and matching quote pairs repeatedly, in one pass."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("answer must be a string or null")
    low, high = 0, len(value)
    while True:
        while low < high and value[low].isspace():
            low += 1
        while high > low and value[high - 1].isspace():
            high -= 1
        if high - low >= 2 and value[low] == value[high - 1] and value[low] in QUOTES:
            low += 1
            high -= 1
            continue
        return value[low:high]


def _matching_braces(text: str) -> dict[int, int]:
    """Map every opening brace that is ever closed to its closing brace, in one scan."""
    matches: dict[int, int] = {}
    stack: list[int] = []
    for brace in BRACE.finditer(text):
        if brace.group() == "{":
            stack.append(brace.start())
        elif stack:
            matches[stack.pop()] = brace.start()
    return matches


def extract_last_boxed(text: str) -> str | None:
    """Return the normalized content of the balanced boxed expression that begins last."""
    if not isinstance(text, str):
        raise ValueError("response must be a string")
    if MARKER not in text:
        return None
    matches = _matching_braces(text)
    start = text.rfind(MARKER)
    while start >= 0:
        brace = start + len(MARKER) - 1
        if brace in matches:
            return normalize_answer(text[brace + 1:matches[brace]])
        start = text.rfind(MARKER, 0, start)
    return None


def extract_first_python(text: str) -> str | None:
    if not isinstance(text, str):
        raise ValueError("response must be a string")
    start = text.find(FENCE)
    if start < 0:
        return None
    content_start = start + len(FENCE)
    end = text.find("```", content_start)
    return None if end < 0 else text[content_start:end]
