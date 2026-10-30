"""Strict loader for the problems file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import read_utf8, strict_loads


PROBLEM_KEYS = {"problem_id", "prompt", "answer"}


BLANK = " \t\r"


def load_problems(path: Path) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    seen: set[str] = set()
    # Only U+000A separates records; str.splitlines would also split inside JSON strings.
    for line_number, line in enumerate(read_utf8(path).split("\n"), 1):
        if not line.strip(BLANK):
            continue
        try:
            row = strict_loads(line)
        except ValueError as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc
        if type(row) is not dict or not set(row) <= PROBLEM_KEYS or not {"problem_id", "prompt"} <= set(row):
            raise ValueError(f"line {line_number}: a problem has exactly problem_id, prompt, and optional answer")
        if type(row["problem_id"]) is not str or not row["problem_id"] or type(row["prompt"]) is not str:
            raise ValueError(f"line {line_number}: invalid problem_id or prompt")
        answer = row.get("answer")
        if answer is not None and type(answer) is not str:
            raise ValueError(f"line {line_number}: answer must be a string or null")
        if row["problem_id"] in seen:
            raise ValueError(f"line {line_number}: duplicate problem_id")
        seen.add(row["problem_id"])
        problems.append({"problem_id": row["problem_id"], "prompt": row["prompt"], "answer": answer})
    if not problems:
        raise ValueError("problems file must be non-empty")
    return problems
