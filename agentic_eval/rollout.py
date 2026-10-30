"""Multi-turn engine execution and record validation."""

from __future__ import annotations

import os
import sys
import threading
from typing import Any, Callable

from .core import outcome_tool_reward, overlong_penalty
from .parsing import extract_first_python, extract_last_boxed, normalize_answer


TIMEOUT_STATUS = 3


def _expire(description: str) -> None:
    # A hung call cannot be interrupted reliably from Python, and an engine may swallow
    # any exception raised into it, so the only dependable response is to end the process.
    # Committed state is already durable because every commit is an atomic replacement.
    try:
        sys.stderr.write(f"error: {description} exceeded the call timeout\n")
        sys.stderr.flush()
    finally:
        os._exit(TIMEOUT_STATUS)


def limited(call: Callable[[], Any], limit: float | None, description: str) -> Any:
    """Run one engine call under its own time limit; the limit never spans two calls."""
    if limit is None:
        return call()
    watchdog = threading.Timer(limit, _expire, args=(description,))
    watchdog.daemon = True
    watchdog.start()
    try:
        return call()
    finally:
        watchdog.cancel()


def make_record(
    engine: Any,
    problem: dict[str, Any],
    sample_index: int,
    settings: dict[str, Any],
    call_offset: int,
    call_timeout: float | None = None,
) -> tuple[dict[str, Any], int]:
    # Construct a new transcript for every invocation, including replay after a crash.
    # No sample history is stored in a default argument or shared module state.
    history: list[dict[str, str]] = list()
    start_offset = call_offset
    final_answer: str | None = None
    for turn_index in range(settings["max_turns"]):
        seed = settings["seed"] + call_offset
        call_offset += 1
        text = limited(lambda: engine.generate_turn(
            problem["prompt"],
            [dict(item) for item in history],
            problem_id=problem["problem_id"],
            sample_index=sample_index,
            turn_index=turn_index,
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            max_tokens=settings["max_tokens"],
            seed=seed,
        ), call_timeout, "generate_turn")
        if not isinstance(text, str) or not text:
            raise ValueError("generate_turn must return a non-empty string")
        history.append({"role": "assistant", "content": text})
        final_answer = extract_last_boxed(text)
        if final_answer is not None:
            break
        code = extract_first_python(text)
        if code is None:
            break
        tool_seed = settings["seed"] + call_offset
        call_offset += 1
        result = limited(lambda: engine.run_tool(
            code,
            problem_id=problem["problem_id"],
            sample_index=sample_index,
            turn_index=turn_index,
            seed=tool_seed,
        ), call_timeout, "run_tool")
        if not isinstance(result, str):
            raise ValueError("run_tool must return a string")
        history.append({"role": "tool", "content": result})
    reference = problem["answer"]
    correct = None if reference is None or final_answer is None else normalize_answer(reference) == final_answer
    tool_calls = sum(item["role"] == "tool" for item in history)
    turns = sum(item["role"] == "assistant" for item in history)
    response_chars = sum(len(item["content"]) for item in history if item["role"] == "assistant")
    reward = outcome_tool_reward(correct is True, tool_calls, settings["tool_bonus_per_call"], settings["tool_bonus_cap"])
    reward += overlong_penalty(response_chars, settings["length_max"], settings["length_cache"])
    return {
        "problem_id": problem["problem_id"],
        "sample_index": sample_index,
        "num_turns": turns,
        "num_tool_calls": tool_calls,
        "final_answer": final_answer,
        "correct": correct,
        "response_chars": response_chars,
        "reward": reward,
        "call_seed_start": settings["seed"] + start_offset,
        "call_seed_end_exclusive": settings["seed"] + call_offset,
        "transcript": history,
    }, call_offset


def run_remaining(
    engine: Any,
    problems: list[dict[str, Any]],
    settings: dict[str, Any],
    records: list[dict[str, Any]],
    call_offset: int,
    commit: Callable[[list[dict[str, Any]], int], None],
    call_timeout: float | None = None,
) -> tuple[list[dict[str, Any]], int]:
    total = len(problems) * settings["samples_per_problem"]
    for flat_index in range(len(records), total):
        problem = problems[flat_index // settings["samples_per_problem"]]
        sample_index = flat_index % settings["samples_per_problem"]
        record, call_offset = make_record(engine, problem, sample_index, settings, call_offset, call_timeout)
        records.append(record)
        commit(records, call_offset)
    return records, call_offset
