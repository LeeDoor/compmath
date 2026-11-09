import unittest

from linear_programming.math_tasks.canonical_form import to_canonical_form
from linear_programming.math_tasks.standard_form import to_standard_form


class LinearFormTests(unittest.TestCase):
    def test_canonical_form_uses_constraint_signs_and_free_variables(self):
        result = to_canonical_form(
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
        self.assertIn("x1, ..., x5 >= 0", result)

    def test_standard_form_uses_constraint_signs_and_free_variables(self):
        result = to_standard_form(
            [[1, 2], [-1, 1], [3, 4]],
            [5, -2, 1],
            [2, -3],
            "min",
            ["<=", ">=", "="],
            ["free", "nonnegative"],
        )

        self.assertIn("x1 -> x'1 - x''1", result)
        self.assertIn("=>", result)
        self.assertIn("<= 2", result)
        self.assertIn("<= -1", result)
        self.assertIn("f(x) = -2.00000000000000*x1", result)
        self.assertIn("2.00000000000000*x2 + 3.00000000000000*x3 -> max", result)
        self.assertIn("ИТОГОВАЯ СТАНДАРТНАЯ ФОРМА", result)

    def test_default_options_and_invalid_option_dimensions(self):
        self.assertIn("1*x1 + x2 = 2", to_canonical_form([[1]], [2], [3]))
        self.assertIn("1*x1 <= 2", to_standard_form([[1]], [2], [3]))

        with self.assertRaisesRegex(ValueError, "constraint_signs"):
            to_canonical_form([[1]], [2], [3], constraint_signs=[])
        with self.assertRaisesRegex(ValueError, "variable_constraints"):
            to_standard_form(
                [[1]], [2], [3], variable_constraints=["free", "free"]
            )


if __name__ == "__main__":
    unittest.main()
