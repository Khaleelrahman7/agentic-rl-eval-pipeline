"""Transcript-derived per-problem and aggregate metrics."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from .core import difficulty_bucket, group_normalized_advantages, pass_at_k
from .parsing import normalize_answer


FIELDS = (
    "average_at_k", "pass_at_k", "maj_at_k", "mean_tool_calls", "mean_response_chars",
    "tool_success_rate", "mean_reward",
)
BUCKETS = ("easy", "medium", "hard", "discarded")


def _problem_metrics(records: list[dict[str, Any]], reference: str | None) -> dict[str, Any]:
    n = len(records)
    c = sum(record["correct"] is True for record in records)
    candidates = [
        (index, normalize_answer(record["final_answer"]))
        for index, record in enumerate(records)
        if record["final_answer"] is not None
    ]
    counts = Counter(answer for _, answer in candidates)
    majority: str | None = None
    if counts:
        highest = max(counts.values())
        majority = next(answer for _, answer in candidates if counts[answer] == highest)
    normalized_reference = normalize_answer(reference)
    majority_correct = majority is not None and normalized_reference is not None and majority == normalized_reference
    tool_messages = [
        item["content"]
        for record in records
        for item in record["transcript"]
        if item["role"] == "tool"
    ]
    successes = sum(not content.startswith("ERROR:") for content in tool_messages)
    rewards = [float(record["reward"]) for record in records]
    average = c / n
    return {
        "average_at_k": average,
        "pass_at_k": pass_at_k(n, c, n),
        "maj_at_k": float(majority_correct),
        "mean_tool_calls": statistics.fmean(record["num_tool_calls"] for record in records),
        "mean_response_chars": statistics.fmean(record["response_chars"] for record in records),
        "tool_success_rate": successes / len(tool_messages) if tool_messages else 1.0,
        "mean_reward": statistics.fmean(rewards),
        "advantages": group_normalized_advantages(rewards),
        "difficulty": difficulty_bucket(average),
    }


def build_metrics(
    records: list[dict[str, Any]], problems: list[dict[str, Any]], samples_per_problem: int
) -> dict[str, Any]:
    summaries = []
    for index, problem in enumerate(problems):
        selected = records[index * samples_per_problem:(index + 1) * samples_per_problem]
        if len(selected) != samples_per_problem or any(row["problem_id"] != problem["problem_id"] for row in selected):
            raise ValueError("records do not form complete sample groups")
        summaries.append({"problem_id": problem["problem_id"], **_problem_metrics(selected, problem["answer"])})
    aggregate: dict[str, Any] = {field: statistics.fmean(summary[field] for summary in summaries) for field in FIELDS}
    counts = dict.fromkeys(BUCKETS, 0)
    for summary in summaries:
        counts[summary["difficulty"] or "discarded"] += 1
    aggregate["difficulty_counts"] = counts
    return {
        "schema_version": 1,
        "num_problems": len(problems),
        "num_generations": len(records),
        "problems": summaries,
        "aggregate": aggregate,
    }
