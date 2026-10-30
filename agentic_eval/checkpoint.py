"""Exact validation of a checkpoint against the rollout rules."""

from __future__ import annotations

import math
from typing import Any

from .core import outcome_tool_reward, overlong_penalty
from .parsing import extract_first_python, extract_last_boxed, normalize_answer
from .settings import _is_int


RECORD_KEYS = {
    "problem_id", "sample_index", "num_turns", "num_tool_calls", "final_answer", "correct",
    "response_chars", "reward", "call_seed_start", "call_seed_end_exclusive", "transcript",
}


CHECKPOINT_KEYS = {"schema_version", "fingerprint", "records", "next_call_offset", "complete"}


def transcript_is_reachable(transcript: list[Any], max_turns: int) -> bool:
    """Replay the rollout rules over the recorded texts; accept only the exact sequence they produce."""
    cursor = 0
    for _ in range(max_turns):
        if cursor >= len(transcript):
            return False  # the rollout would have generated another turn
        item = transcript[cursor]
        if item["role"] != "assistant" or item["content"] == "":
            return False
        cursor += 1
        if extract_last_boxed(item["content"]) is not None or extract_first_python(item["content"]) is None:
            return cursor == len(transcript)
        if cursor >= len(transcript) or transcript[cursor]["role"] != "tool":
            return False
        cursor += 1
    return cursor == len(transcript)


def _final_answer(transcript: list[dict[str, str]]) -> str | None:
    last_assistant = next(item for item in reversed(transcript) if item["role"] == "assistant")
    return extract_last_boxed(last_assistant["content"])


def validate_checkpoint(
    value: Any, expected_fingerprint: str, problems: list[dict[str, Any]], settings: dict[str, Any]
) -> tuple[list[dict[str, Any]], int, bool]:
    if type(value) is not dict or set(value) != CHECKPOINT_KEYS:
        raise ValueError("checkpoint does not have exactly the required keys")
    if not _is_int(value["schema_version"]) or value["schema_version"] != 1:
        raise ValueError("unsupported checkpoint schema version")
    if type(value["fingerprint"]) is not str or value["fingerprint"] != expected_fingerprint:
        raise ValueError("checkpoint fingerprint mismatch")
    records, offset, complete = value["records"], value["next_call_offset"], value["complete"]
    total = len(problems) * settings["samples_per_problem"]
    if type(records) is not list or len(records) > total or not _is_int(offset) or type(complete) is not bool:
        raise ValueError("invalid checkpoint envelope")
    expected_offset = 0
    for flat_index, record in enumerate(records):
        problem = problems[flat_index // settings["samples_per_problem"]]
        sample_index = flat_index % settings["samples_per_problem"]
        if type(record) is not dict or set(record) != RECORD_KEYS:
            raise ValueError("checkpoint record does not have exactly the required keys")
        if type(record["problem_id"]) is not str or record["problem_id"] != problem["problem_id"]:
            raise ValueError("checkpoint records are not the required prefix")
        if not _is_int(record["sample_index"]) or record["sample_index"] != sample_index:
            raise ValueError("checkpoint records are not the required prefix")
        transcript = record["transcript"]
        if type(transcript) is not list:
            raise ValueError("invalid checkpoint transcript")
        for item in transcript:
            if type(item) is not dict or set(item) != {"role", "content"}:
                raise ValueError("invalid transcript item")
            if item["role"] not in ("assistant", "tool") or type(item["role"]) is not str or type(item["content"]) is not str:
                raise ValueError("invalid transcript item")
        if not transcript_is_reachable(transcript, settings["max_turns"]):
            raise ValueError("transcript is not what the rollout rules produce")
        turns = sum(item["role"] == "assistant" for item in transcript)
        tools = sum(item["role"] == "tool" for item in transcript)
        chars = sum(len(item["content"]) for item in transcript if item["role"] == "assistant")
        answer = _final_answer(transcript)
        correct = None if problem["answer"] is None or answer is None else normalize_answer(problem["answer"]) == answer
        reward = outcome_tool_reward(correct is True, tools, settings["tool_bonus_per_call"], settings["tool_bonus_cap"])
        reward += overlong_penalty(chars, settings["length_max"], settings["length_cache"])
        start_seed = settings["seed"] + expected_offset
        expected_offset += len(transcript)
        integers = {
            "num_turns": turns, "num_tool_calls": tools, "response_chars": chars,
            "call_seed_start": start_seed, "call_seed_end_exclusive": settings["seed"] + expected_offset,
        }
        if any(not _is_int(record[key]) or record[key] != expected for key, expected in integers.items()):
            raise ValueError("checkpoint record is internally inconsistent")
        if not (record["final_answer"] is None or type(record["final_answer"]) is str) or record["final_answer"] != answer:
            raise ValueError("checkpoint record is internally inconsistent")
        if not (record["correct"] is None or type(record["correct"]) is bool) or record["correct"] is not correct:
            raise ValueError("checkpoint record is internally inconsistent")
        if type(record["reward"]) not in (int, float) or not math.isfinite(record["reward"]) or record["reward"] != reward:
            raise ValueError("checkpoint record is internally inconsistent")
    # A crash after the last sample commit but before metrics publication leaves
    # every record with complete=false; that state is resumable, the converse is not.
    if offset != expected_offset or (complete and len(records) != total):
        raise ValueError("checkpoint progress is inconsistent")
    return records, offset, complete
