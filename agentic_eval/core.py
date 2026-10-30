"""Validated numerical mechanisms used by the evaluator."""

from __future__ import annotations

import math
import statistics


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def group_normalized_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    if not isinstance(rewards, list) or not rewards:
        raise ValueError("rewards must be a non-empty list")
    values = [_number(value, "reward") for value in rewards]
    epsilon = _number(eps, "eps")
    if epsilon <= 0:
        raise ValueError("eps must be positive")
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    if std == 0:
        return [0.0 for _ in values]
    return [(value - mean) / (std + epsilon) for value in values]


def _surrogate(ratio: float, advantage: float, eps_low: float, eps_high: float) -> float:
    clipped = min(max(ratio, 1.0 - eps_low), 1.0 + eps_high)
    return min(ratio * advantage, clipped * advantage)


def _validate_surrogate(
    token_ratios: list[list[float]], advantages: list[float], eps_low: float, eps_high: float
) -> tuple[list[list[float]], list[float], float, float]:
    if not isinstance(token_ratios, list) or not isinstance(advantages, list):
        raise ValueError("ratios and advantages must be lists")
    if not token_ratios or len(token_ratios) != len(advantages):
        raise ValueError("ratios and advantages must have equal non-zero length")
    low = _number(eps_low, "eps_low")
    high = _number(eps_high, "eps_high")
    if low < 0 or low >= 1 or high <= 0:
        raise ValueError("invalid clipping bounds")
    checked: list[list[float]] = []
    for rollout in token_ratios:
        if not isinstance(rollout, list) or not rollout:
            raise ValueError("every rollout must be a non-empty list")
        row = [_number(value, "ratio") for value in rollout]
        if any(value <= 0 for value in row):
            raise ValueError("ratios must be strictly positive")
        checked.append(row)
    return checked, [_number(value, "advantage") for value in advantages], low, high


def token_level_surrogate(
    token_ratios: list[list[float]], advantages: list[float], eps_low: float, eps_high: float
) -> float:
    ratios, advs, low, high = _validate_surrogate(token_ratios, advantages, eps_low, eps_high)
    values = [
        _surrogate(ratio, advantage, low, high)
        for rollout, advantage in zip(ratios, advs)
        for ratio in rollout
    ]
    return statistics.fmean(values)


def sequence_level_surrogate(
    token_ratios: list[list[float]], advantages: list[float], eps_low: float, eps_high: float
) -> float:
    ratios, advs, low, high = _validate_surrogate(token_ratios, advantages, eps_low, eps_high)
    values = []
    for rollout, advantage in zip(ratios, advs):
        sequence_ratio = math.exp(statistics.fmean(math.log(value) for value in rollout))
        values.append(_surrogate(sequence_ratio, advantage, low, high))
    return statistics.fmean(values)


def outcome_tool_reward(correct: bool, calls: int, per_call: float, cap: float) -> float:
    if not isinstance(correct, bool):
        raise ValueError("correct must be boolean")
    count = _integer(calls, "calls", 0)
    bonus = _number(per_call, "per_call")
    ceiling = _number(cap, "cap")
    if bonus < 0 or ceiling <= 0:
        raise ValueError("invalid reward policy")
    return (1.0 if correct else -1.0) + min(count * bonus, ceiling)


def overlong_penalty(length: int, maximum: int, cache: int) -> float:
    size = _integer(length, "length", 0)
    limit = _integer(maximum, "maximum", 1)
    window = _integer(cache, "cache", 1)
    if window > limit:
        raise ValueError("cache cannot exceed maximum")
    safe = limit - window
    if size <= safe:
        return 0.0
    if size <= limit:
        return (safe - size) / window
    return -1.0


def pass_at_k(n: int, c: int, k: int) -> float:
    total = _integer(n, "n", 1)
    correct = _integer(c, "c", 0)
    chosen = _integer(k, "k", 1)
    if correct > total or chosen > total:
        raise ValueError("invalid pass@k bounds")
    if total - correct < chosen:
        return 1.0
    product = 1.0
    for index in range(chosen):
        product *= (total - correct - index) / (total - index)
    return 1.0 - product


def difficulty_bucket(accuracy: float) -> str | None:
    value = _number(accuracy, "accuracy")
    if value < 0 or value > 1:
        raise ValueError("accuracy must be in [0, 1]")
    if value == 0 or value == 1:
        return None
    if value >= 0.75:
        return "easy"
    if value <= 0.25:
        return "hard"
    return "medium"
