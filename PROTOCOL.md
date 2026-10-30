# Evaluation protocol

The service uses only the Python standard library. JSON object keys may have any order, but array and JSONL record order is significant.

## Conventions

These rules apply to every input the service reads, every artifact it writes, and every public function, wherever the situation arises.

A newline is the single character U+000A. Nothing else terminates a line or a JSONL record: not U+000D, U+0085, U+2028, U+2029, or any other character. A blank line contains only spaces, tabs, or carriage returns.

JSON means RFC 8259 text encoded as UTF-8. Bytes that are not valid UTF-8, the tokens `NaN`, `Infinity`, and `-Infinity`, and an object that repeats a key are all malformed. Every artifact the service writes must itself satisfy this rule.

Types are exact. A boolean is never a number, an integer is never a float (`1.0` is not `1`, and `true` is not `1`), and a string is never a number. An object described by a list of keys has exactly those keys, with none missing and none added, at every level. Integer parameters of the public functions follow the same rule.

A string is a sequence of Unicode code points and need not be well-formed text. An engine or a problems file may supply control characters, line and paragraph separators, or an unpaired surrogate. Every such string must reach the engine and every artifact unchanged, so that parsing the artifact as JSON yields the identical string.

Arithmetic is IEEE 754 double precision, but a result must not depend on the order or relative size of the terms being combined. Wherever this contract takes a sum or a mean, the result must equal the exact real-number sum or mean of the given doubles to within a relative error of 1e-9, however much the terms differ in magnitude or cancel one another.

Parsing cost is linear. Extracting a final answer, normalizing an answer, and finding a code block must each take time proportional to the length of the text, whatever it contains: megabytes of unclosed or nested `\boxed{` markers, braces, quotation marks, whitespace, or fences must be handled in seconds, not minutes.

The process exits with status 0 only when the run finished and all four artifacts were published, or when an audit accepted the output directory. Every other outcome exits non-zero and leaves previously committed state intact, whatever kind of exception or exit an engine module raises while it is loaded or called.

## Input and engine contract

The problems file is non-empty JSONL. Each row is an object with a unique non-empty string `problem_id`, a string `prompt`, and an optional `answer` that is either a string or null, and no other key. Blank lines are allowed. Any malformed row must fail before an engine call.

The engine module exports `generate_turn(prompt, history, *, problem_id, sample_index, turn_index, temperature, top_p, max_tokens, seed)` and `run_tool(code, *, problem_id, sample_index, turn_index, seed)`. Both return strings; an empty assistant response is invalid, while an empty tool result is valid. The prompt, complete accumulated history, identities, and sampling values must be passed unchanged. Each call receives its own copy of the history; an engine that mutates its arguments must not alter the recorded transcript or any later call.

Process problems in file order and sample indices in ascending order. Each sample starts with an empty history. Use one run-wide call offset beginning at zero. Every attempted generation or tool call uses `base_seed + call_offset`. Advance the offset immediately before making that call. A resumed incomplete sample replays from its last committed sample boundary and therefore reuses that incomplete sample's offsets.

After a generation, append its exact text as `{"role":"assistant","content":text}`. Search it first for final answers and otherwise for code. A final answer is the content, possibly empty, of the last `\boxed{...}` expression, parsed with balanced nested braces. Every `{` and every `}` counts toward the balance, including one preceded by a backslash. When boxed expressions nest, the last one is the one whose `\boxed{` begins last. Remove surrounding whitespace and matching single or double quotation marks repeatedly. Finding an answer ends the sample. If none exists, execute the first fenced block whose opening fence is lowercase `python` followed immediately by a newline; preserve the code between fences exactly, even when it is empty. An opening fence with no later closing fence is not a block. Append the exact tool result with role `tool`, then continue. A turn with neither form ends the sample. A tool request on the last allowed turn is still executed and recorded.

The package `agentic_eval.parsing` exposes these rules as public functions: `extract_last_boxed(text)` returns the normalized final answer or null, `extract_first_python(text)` returns the code of the first block or null, and `normalize_answer(value)` normalizes a string and maps null to null. The mathematical functions below are public in `agentic_eval.core`.

If a reference string is present, compare normalized reference and final answer. If either the reference is absent/null or no final answer exists, `correct` is null. For reward purposes only, null correctness is treated as false.

## Mathematical contract

All numeric inputs below reject booleans and non-finite values. Malformed collections, wrongly typed arguments, and invalid bounds raise `ValueError`.

`group_normalized_advantages(rewards, eps=1e-6)` uses the population mean and population standard deviation, returning `(reward - mean) / (std + eps)` for each value. Rewards must be a non-empty list and `eps` must be positive. A group whose values are all equal returns exact zeros, even when a floating-point mean of those values is not exact.

For clipped surrogates, every importance ratio is strictly positive, advantages and rollout groups have equal non-zero length, every rollout is non-empty, `0 <= eps_low < 1`, and `eps_high > 0`. Let `lower=1-eps_low`, `upper=1+eps_high`, and `s(r,A)=min(r*A, clamp(r,lower,upper)*A)`. `token_level_surrogate` averages `s` over every token in the group. `sequence_level_surrogate` computes each rollout ratio as `exp(mean(log(token_ratio)))`, applies `s` once per rollout, and averages across rollouts.

`outcome_tool_reward(correct, calls, per_call, cap)` returns `(1 if correct else -1) + min(calls*per_call, cap)`. Correctness is boolean, calls is a non-negative integer, `per_call >= 0`, and `cap > 0`.

`overlong_penalty(length, maximum, cache)` is zero through `maximum-cache`, changes linearly as `((maximum-cache)-length)/cache` through `maximum`, and is `-1` above it. Length is a non-negative integer, maximum is a positive integer, and cache is an integer with `0 < cache <= maximum`.

`pass_at_k(n,c,k)` is `1 - product((n-c-i)/(n-i), i=0..k-1)`, or `1` when `n-c < k`. The arguments are integers with `n>=1`, `0<=c<=n`, and `1<=k<=n`. `difficulty_bucket(accuracy)` takes a number in `[0, 1]`, discards exact zero and one with null, returns `easy` at or above 0.75, `hard` at or below 0.25, and `medium` otherwise.

## Output and recovery contract

The CLI accepts `--problems`, `--engine-module`, `--output-dir`, `--samples-per-problem`, `--max-turns`, `--temperature`, `--top-p`, `--max-tokens`, `--seed`, `--tool-bonus-per-call`, `--tool-bonus-cap`, `--length-max`, `--length-cache`, `--resume`, `--audit`, and `--call-timeout`. Counts and token/length budgets are positive integers; the seed is any integer; temperature is finite and non-negative; `0 < top-p <= 1`; reward policy bounds follow the mathematical contract. No other restriction applies to a setting.

`config.json` is an object with exactly `schema_version` (integer 1), `problem_ids` (in file order), `input_sha256` (of the problems file bytes), the ten settings `samples_per_problem`, `max_turns`, `temperature`, `top_p`, `max_tokens`, `seed`, `tool_bonus_per_call`, `tool_bonus_cap`, `length_max`, and `length_cache`, `engine` (`"module_adapter"`), and `selection_mode` (`"agentic_tir_rollout"`).

Each `generations.jsonl` row is an object with exactly `problem_id`, zero-based integer `sample_index`, integer `num_turns`, integer `num_tool_calls`, `final_answer` (string or null), `correct` (boolean or null), integer `response_chars`, numeric `reward`, integers `call_seed_start` and `call_seed_end_exclusive`, and the exact ordered `transcript`, whose items have exactly `role` and `content`. `num_turns` counts assistant messages, `num_tool_calls` counts tool messages, and `response_chars` is the sum of Python string lengths of assistant-message contents only. The seed fields are the base-seed-adjusted half-open interval; its width equals the number of assistant plus tool calls. Reward uses the declared outcome/tool policy plus the declared overlong penalty.

`metrics.json` is an object with exactly `schema_version` (integer 1), integers `num_problems` and `num_generations`, ordered `problems`, and `aggregate`. Here `k` is `samples-per-problem`. Each problem entry has `problem_id` and, like the aggregate, `average_at_k`, `pass_at_k`, `maj_at_k`, `mean_tool_calls`, `mean_response_chars`, `tool_success_rate`, and `mean_reward`. `average_at_k` is the fraction correct; `pass_at_k` uses the formula above with all `n=k` generated samples; `maj_at_k` is numeric 1 or 0. Null correctness counts as not correct. Majority considers non-null normalized answers and breaks count ties toward the answer occurring at the lowest sample index. A majority answer is correct only when it equals the normalized non-null reference. Tool success is the fraction of tool messages not beginning exactly with `ERROR:`; it is 1 when there are no tool calls. `mean_reward` is the arithmetic mean of the sample rewards. Aggregate fields are arithmetic means across problems.

Each problem entry additionally has `advantages`, the `group_normalized_advantages` of its sample rewards in sample order with the default `eps`, and `difficulty`, the `difficulty_bucket` of its `average_at_k`. In their place the aggregate has `difficulty_counts`, an object with exactly the keys `easy`, `medium`, `hard`, and `discarded` whose integer values count the problems in each bucket, where a null bucket counts as `discarded`.

`checkpoint.json` is an object with exactly `schema_version` (integer 1), `fingerprint` (a string), `records` (the complete committed record list), integer `next_call_offset`, and boolean `complete`. The fingerprint covers the normalized problems and the settings recorded in `config.json`: files that parse to the same normalized problems (an omitted answer normalizes to null) share a fingerprint, while any change to a problem ID, prompt, answer, problem order, or setting changes it. Commit only complete samples, in required prefix order, and commit each sample as soon as it completes: after any interruption `generations.jsonl` and the checkpoint hold exactly the samples that finished before it.

With `--resume`, a valid matching checkpoint must exist and is authoritative; skip committed samples and continue at its offset. Reject a missing, corrupt, mismatched, non-prefix, or internally inconsistent checkpoint without modifying it. A committed record is valid only if it satisfies the exact `generations.jsonl` row contract, every derived field equals the value derived from its transcript, and the transcript is exactly what the rollout rules above produce, under the run's settings, for the assistant and tool texts it contains: it could not have ended earlier, continued further, or contained a different sequence of roles. A checkpoint may hold every record while not yet complete; resuming it makes no engine calls and only finishes publication. Without `--resume`, begin a fresh run.

With `--audit`, the service inspects a finished output directory and changes nothing: it makes no engine call, never loads the engine module, and creates, modifies, or removes no file. It exits 0 only if `checkpoint.json` is a valid matching checkpoint under every resume rule for the given problems and settings and is complete, and each of `config.json`, `generations.jsonl`, and `metrics.json` parses to exactly the value a successful run publishes for those records. Object key order and whitespace between JSON tokens do not matter. Every other difference does, including a missing artifact, a missing, extra, or reordered record or list element, a missing or extra key, a changed value, and an integer written as a float or a boolean. Any other state, and `--audit` combined with `--resume`, exits non-zero.

`--call-timeout SECONDS` is optional and is not a setting: it must be a finite positive number, it is not recorded in `config.json`, and it does not affect the fingerprint, so a run may be resumed or audited with a different limit or with none. When it is given, every `generate_turn` or `run_tool` call has its own limit measured from the start of that call; time between calls never counts, and several calls that each finish in time may together take longer than the limit. A call still running when its limit expires ends the process with non-zero status within ten seconds, under the exit-status rule above and whatever the engine does, including an engine that catches or ignores every exception or signal delivered to it. Without the option, calls are not limited.

Result and checkpoint writes use a temporary file in the destination directory, flush and fsync it, and replace the target atomically. Stale temporary files must not affect a run. A successful run leaves all four files mutually consistent and marks the checkpoint complete.
