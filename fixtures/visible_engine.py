"""Small deterministic engine supplied for the required visible run."""

from __future__ import annotations


def generate_turn(prompt, history, *, problem_id, sample_index, turn_index, temperature, top_p, max_tokens, seed):
    if problem_id == "visible-add":
        if turn_index == 0:
            return "I will calculate.\n```python\nprint(2 + 2)\n```"
        return f"The tool returned {history[-1]['content']}. Therefore \\boxed{{4}}"
    if problem_id == "visible-nested":
        return "An earlier decoy is \\boxed{0}; the reduced value is \\boxed{\\frac{1}{2}}."
    return "The sentinel is \\boxed{'moon'}."


def run_tool(code, *, problem_id, sample_index, turn_index, seed):
    if code == "print(2 + 2)\n":
        return "4\n"
    return "ERROR: unsupported visible program"
