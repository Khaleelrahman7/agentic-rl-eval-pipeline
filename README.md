# Agentic RL Evaluation Pipeline

A deterministic, standard-library Python pipeline for evaluating multi-turn,
tool-using agent rollouts. It preserves exact prompt and transcript state,
tracks run-wide call seeds, computes reward and GRPO-style statistics, and
publishes crash-safe artifacts that can be resumed or audited.

This standalone repository was extracted from a Terminal-Bench task I authored
and has been developed and iterated primarily with Claude Code.

## What it includes

- Multi-turn assistant/tool rollout execution through a pluggable engine module
- Exact transcript, identity, sampling-parameter, and call-seed accounting
- Nested `\\boxed{...}` answer parsing and fenced Python tool extraction
- Group-normalized advantages and token-/sequence-level clipped surrogates
- Outcome, tool-use, and response-length reward shaping
- Per-problem and aggregate pass@k, majority, tool-success, reward, and
  difficulty metrics
- Atomic JSON/JSONL publication, checkpoint validation, safe resume, and a
  read-only audit mode
- Optional per-engine-call watchdog timeouts
- No model weights, network calls, or third-party runtime dependencies

The full behavioral contract is documented in [PROTOCOL.md](PROTOCOL.md).

## Requirements

- Python 3.11 or newer

The runtime uses only the Python standard library.

## Quick start

Run the included deterministic example:

```bash
python bin/run_eval.py \
  --problems fixtures/visible_problems.jsonl \
  --engine-module fixtures/visible_engine.py \
  --output-dir results \
  --samples-per-problem 2 \
  --max-turns 3 \
  --temperature 1.0 \
  --top-p 0.6 \
  --max-tokens 512 \
  --seed 410
```

The command writes:

- `results/config.json`
- `results/generations.jsonl`
- `results/metrics.json`
- `results/checkpoint.json`

Audit a completed run without loading or calling the engine:

```bash
python bin/run_eval.py \
  --problems fixtures/visible_problems.jsonl \
  --engine-module fixtures/visible_engine.py \
  --output-dir results \
  --samples-per-problem 2 \
  --max-turns 3 \
  --temperature 1.0 \
  --top-p 0.6 \
  --max-tokens 512 \
  --seed 410 \
  --audit
```

## Engine adapter

An engine is a Python module exposing two functions:

```python
def generate_turn(
    prompt,
    history,
    *,
    problem_id,
    sample_index,
    turn_index,
    temperature,
    top_p,
    max_tokens,
    seed,
):
    ...


def run_tool(code, *, problem_id, sample_index, turn_index, seed):
    ...
```

Both functions return strings. The included
[`fixtures/visible_engine.py`](fixtures/visible_engine.py) is a minimal,
deterministic example.

## Tests

```bash
python -m unittest discover -s tests -v
```

