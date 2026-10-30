"""Command line settings, their bounds, and the published run configuration."""

from __future__ import annotations

import argparse
import math
from typing import Any


SETTING_KEYS = (
    "samples_per_problem", "max_turns", "temperature", "top_p", "max_tokens", "seed",
    "tool_bonus_per_call", "tool_bonus_cap", "length_max", "length_cache",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run deterministic agentic evaluation")
    result.add_argument("--problems", required=True)
    result.add_argument("--engine-module", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--samples-per-problem", type=int, default=3)
    result.add_argument("--max-turns", type=int, default=4)
    result.add_argument("--temperature", type=float, default=0.7)
    result.add_argument("--top-p", type=float, default=0.95)
    result.add_argument("--max-tokens", type=int, default=256)
    result.add_argument("--seed", type=int, default=1000)
    result.add_argument("--tool-bonus-per-call", type=float, default=0.1)
    result.add_argument("--tool-bonus-cap", type=float, default=0.3)
    result.add_argument("--length-max", type=int, default=1000)
    result.add_argument("--length-cache", type=int, default=200)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--audit", action="store_true")
    result.add_argument("--call-timeout", type=float, default=None)
    return result


def _is_int(value: object) -> bool:
    return type(value) is int


def _finite(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def settings_from(args: argparse.Namespace) -> dict[str, Any]:
    settings = {key: getattr(args, key) for key in SETTING_KEYS}
    for key in ("samples_per_problem", "max_turns", "max_tokens", "length_max", "length_cache"):
        if not _is_int(settings[key]) or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not _is_int(settings["seed"]):
        raise ValueError("seed must be an integer")
    for key in ("temperature", "top_p", "tool_bonus_per_call", "tool_bonus_cap"):
        settings[key] = _finite(settings[key], key)
    if settings["temperature"] < 0 or not 0 < settings["top_p"] <= 1:
        raise ValueError("invalid sampling settings")
    if settings["tool_bonus_per_call"] < 0 or settings["tool_bonus_cap"] <= 0:
        raise ValueError("invalid reward settings")
    if settings["length_cache"] > settings["length_max"]:
        raise ValueError("length_cache cannot exceed length_max")
    return settings


def call_timeout_from(args: argparse.Namespace) -> float | None:
    """The per-call limit is operational: it is validated here but never recorded or fingerprinted."""
    limit = args.call_timeout
    if limit is None:
        return None
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("call timeout must be a finite positive number of seconds")
    return limit


def build_config(problems: list[dict[str, Any]], input_hash: str, settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1, "problem_ids": [problem["problem_id"] for problem in problems],
        "input_sha256": input_hash, **settings, "engine": "module_adapter",
        "selection_mode": "agentic_tir_rollout",
    }
