"""Command line orchestration of runs, recovery, and audits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, atomic_write_jsonl, file_sha256, fingerprint, load_json
from .audit import audit
from .checkpoint import validate_checkpoint
from .engine import load_engine
from .metrics import build_metrics
from .problems import load_problems
from .rollout import run_remaining
from .settings import build_config, call_timeout_from, parser, settings_from


def execute(args: argparse.Namespace) -> None:
    settings = settings_from(args)
    call_timeout = call_timeout_from(args)
    problem_path = Path(args.problems)
    problems = load_problems(problem_path)
    input_hash = file_sha256(problem_path)
    run_fingerprint = fingerprint(problems, settings)
    output = Path(args.output_dir)
    if args.audit:
        if args.resume:
            raise ValueError("--audit cannot be combined with --resume")
        audit(output, problems, settings, input_hash, run_fingerprint)
        return
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.json"
    if args.resume:
        if not checkpoint_path.is_file():
            raise ValueError("--resume requires an existing checkpoint")
        records, call_offset, complete = validate_checkpoint(load_json(checkpoint_path), run_fingerprint, problems, settings)
    else:
        records, call_offset, complete = [], 0, False
    atomic_write_json(output / "config.json", build_config(problems, input_hash, settings))

    def save(committed: list[dict[str, Any]], offset: int, done: bool = False) -> None:
        atomic_write_jsonl(output / "generations.jsonl", committed)
        atomic_write_json(checkpoint_path, {
            "schema_version": 1, "fingerprint": run_fingerprint, "records": committed,
            "next_call_offset": offset, "complete": done,
        })

    save(records, call_offset, complete)
    if len(records) < len(problems) * settings["samples_per_problem"]:
        engine = load_engine(args.engine_module)
        records, call_offset = run_remaining(engine, problems, settings, records, call_offset, save, call_timeout)
    metrics = build_metrics(records, problems, settings["samples_per_problem"])
    atomic_write_json(output / "metrics.json", metrics)
    save(records, call_offset, True)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        execute(args)
    except BaseException as exc:  # noqa: BLE001 - an engine may raise SystemExit or KeyboardInterrupt
        # Status 0 is reserved for a finished, fully published run, whatever was raised.
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
