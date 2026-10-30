"""Offline audit of a finished output directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import load_json, read_utf8, strict_loads
from .checkpoint import validate_checkpoint
from .metrics import build_metrics
from .problems import BLANK
from .settings import build_config


def same_json(actual: Any, expected: Any) -> bool:
    """Compare parsed JSON with exact types; key order is irrelevant, list order is not."""
    if type(expected) is dict:
        return type(actual) is dict and set(actual) == set(expected) and all(same_json(actual[key], expected[key]) for key in expected)
    if type(expected) is list:
        return type(actual) is list and len(actual) == len(expected) and all(same_json(a, e) for a, e in zip(actual, expected))
    if type(expected) is float:
        return type(actual) in (int, float) and actual == expected
    return type(actual) is type(expected) and actual == expected


def load_jsonl(path: Path) -> list[Any]:
    lines = read_utf8(path).split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [strict_loads(line) for line in lines if line.strip(BLANK)]


def audit(output: Path, problems: list[dict[str, Any]], settings: dict[str, Any], input_hash: str, run_fingerprint: str) -> None:
    """Accept only a finished directory whose artifacts equal what a successful run publishes."""
    for name in ("config.json", "generations.jsonl", "metrics.json", "checkpoint.json"):
        if not (output / name).is_file():
            raise ValueError(f"audit: {name} is missing")
    records, _, complete = validate_checkpoint(load_json(output / "checkpoint.json"), run_fingerprint, problems, settings)
    if not complete:
        raise ValueError("audit: the run is not complete")
    if not same_json(load_json(output / "config.json"), build_config(problems, input_hash, settings)):
        raise ValueError("audit: config.json does not match the run")
    if not same_json(load_jsonl(output / "generations.jsonl"), records):
        raise ValueError("audit: generations.jsonl does not match the committed records")
    expected_metrics = build_metrics(records, problems, settings["samples_per_problem"])
    if not same_json(load_json(output / "metrics.json"), expected_metrics):
        raise ValueError("audit: metrics.json does not match the committed records")
