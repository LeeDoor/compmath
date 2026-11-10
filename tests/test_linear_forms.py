import unittest

import numpy as np

from linear_programming.math_tasks.canonical_form import (
    make_canonical,
    run_canonical_algorithm,
)
from linear_programming.math_tasks.standard_form import to_standard_form


class LinearFormTests(unittest.TestCase):
    def test_canonical_form_uses_constraint_signs_and_free_variables(self):
        result = run_canonical_algorithm(
            [[1, 2], [-1, 1], [2, 1]],
            [4, -2, 3],
            [3, 1],
            "max",
            [">=", "<=", "="],
            ["free", "nonnegative"],
        )

        self.assertIn("x1 -> x'1 - x''1", result)
        self.assertIn(">= 4  =>", result)
        self.assertIn("<= -2", result)
        self.assertIn("- x", result)
        self.assertIn("ИТОГОВАЯ КАНОНИЧЕСКАЯ ФОРМА", result)
        self.assertIn("x'1, x''1, x2, x3, x4 >= 0", result)
        final_form = result.split("ИТОГОВАЯ КАНОНИЧЕСКАЯ ФОРМА:")[-1]
        self.assertIn("f(x) = 3*x'1 - 3*x''1 + x2 -> max", final_form)

    def test_make_canonical_returns_the_same_canonicalization(self):
        A, b, c, decision_count = make_canonical(
            [[1, 2], [-1, 1], [2, 1]],
            [4, -2, 3],
            [3, 1],
            "max",
            [">=", "<=", "="],
            ["free", "nonnegative"],
        )

        np.testing.assert_array_equal(
            A,
            [[1, -1, 2, -1, 0], [-1, 1, 1, 0, 1], [2, -2, 1, 0, 0]],
        )
        np.testing.assert_array_equal(b, [4, -2, 3])
        np.testing.assert_array_equal(c, [3, -3, 1, 0, 0])
        self.assertEqual(decision_count, 3)

    def test_standard_form_uses_constraint_signs_and_free_variables(self):
        result = to_standard_form(
            [[1, 2], [-1, 1], [3, 4]],
            [5, -2, 1],
            [2, -3],
            "min",
            ["<=", ">=", "="],
            ["free", "nonnegative"],
        )

        self.assertIn("x1 -> x1' - x1''", result)
        self.assertIn("=>", result)
        self.assertIn("<= 2", result)
        self.assertIn("<= -1", result)
        self.assertIn("f(x) = -2*x1' + 2*x1'' + 3*x2 -> max", result)
        self.assertIn("ИТОГОВАЯ СТАНДАРТНАЯ ФОРМА", result)

    def test_default_options_and_invalid_option_dimensions(self):
        self.assertIn(
            "x1 + x2 = 2", run_canonical_algorithm([[1]], [2], [3])
        )
        self.assertIn("x1 <= 2", to_standard_form([[1]], [2], [3]))

        with self.assertRaisesRegex(ValueError, "constraint_signs"):
            run_canonical_algorithm([[1]], [2], [3], constraint_signs=[])
        with self.assertRaisesRegex(ValueError, "variable_constraints"):
            to_standard_form(
                [[1]], [2], [3], variable_constraints=["free", "free"]
            )


if __name__ == "__main__":
    unittest.main()
