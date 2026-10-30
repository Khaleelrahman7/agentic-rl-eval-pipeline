from __future__ import annotations

import math
import unittest

from agentic_eval.core import (
    difficulty_bucket,
    group_normalized_advantages,
    overlong_penalty,
    sequence_level_surrogate,
    token_level_surrogate,
)
from agentic_eval.parsing import extract_first_python, extract_last_boxed, normalize_answer


class MechanismTests(unittest.TestCase):
    def test_population_advantages_and_singleton(self):
        self.assertEqual(group_normalized_advantages([7.0]), [0.0])
        got = group_normalized_advantages([1.0, 3.0], eps=1e-6)
        self.assertAlmostEqual(got[0], -1 / 1.000001)
        self.assertAlmostEqual(got[1], 1 / 1.000001)

    def test_clipping_direction_depends_on_advantage(self):
        self.assertAlmostEqual(token_level_surrogate([[2.0]], [1.0], .2, .2), 1.2)
        self.assertAlmostEqual(token_level_surrogate([[2.0]], [-1.0], .2, .2), -2.0)
        self.assertAlmostEqual(sequence_level_surrogate([[1.0, 4.0]], [1.0], .2, .2), 1.2)

    def test_overlong_and_boundaries(self):
        self.assertEqual(overlong_penalty(80, 100, 20), 0.0)
        self.assertEqual(overlong_penalty(90, 100, 20), -0.5)
        self.assertEqual(overlong_penalty(101, 100, 20), -1.0)
        self.assertEqual(difficulty_bucket(.75), "easy")
        self.assertEqual(difficulty_bucket(.25), "hard")

    def test_nested_last_box_and_normalization(self):
        text = r"first \boxed{bad}; then \boxed{ '\frac{1}{2}' }"
        self.assertEqual(extract_last_boxed(text), r"\frac{1}{2}")
        self.assertEqual(normalize_answer("  '\"x\"'  "), "x")

    def test_python_fence_is_exact(self):
        self.assertEqual(extract_first_python("x```python\na < b\n```y"), "a < b\n")
        self.assertIsNone(extract_first_python("```Python\npass\n```"))

    def test_invalid_nonfinite(self):
        with self.assertRaises(ValueError):
            group_normalized_advantages([math.inf])


if __name__ == "__main__":
    unittest.main()
