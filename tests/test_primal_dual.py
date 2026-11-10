import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from linear_programming.math_tasks.artificial_basis import solve_artificial_basis
from linear_programming.math_tasks.primal_dual import solve_primal_dual
from linear_programming.math_tasks.simplex import solve_simplex


class PrimalDualTests(unittest.TestCase):
    def test_solves_primal_and_dual_from_simplex_result(self):
        result, steps = solve_primal_dual(
            [[1, 1], [1, 0], [0, 1]],
            [4, 2, 3],
            [3, 2],
        )

        self.assertIn("x* = (x1 = 2, x2 = 2)", result)
        self.assertIn("Способ 1 (система):   y* = (y1 = 2, y2 = 1, y3 = 0)", result)
        self.assertIn("Способ 2 (обратная):  y* = (y1 = 2, y2 = 1, y3 = 0)", result)
        self.assertIn("f(x*) = g(y*) = 10", result)
        self.assertIn("СИМПЛЕКС-МЕТОД. ИТЕРАЦИЯ №1", steps)

    def test_simplex_default_result_still_includes_slack_variables(self):
        output = StringIO()
        with redirect_stdout(output):
            solution, steps = solve_simplex([[1]], [2], [1])

        self.assertEqual(solution.shape, (1, 2))
        self.assertEqual(solution.tolist(), [[2.0, 0.0]])
        self.assertIn("СИМПЛЕКС-МЕТОД. ИТЕРАЦИЯ №1", steps)
        self.assertEqual(output.getvalue(), "")

    def test_artificial_basis_returns_steps_without_printing(self):
        output = StringIO()
        with redirect_stdout(output):
            solution, steps = solve_artificial_basis([[1]], [2], [1])

        np.testing.assert_array_equal(solution, [[2]])
        self.assertIn("ЭТАП 1. ВСПОМОГАТЕЛЬНАЯ ЗАДАЧА", steps)
        self.assertIn("ЭТАП 2. РЕШЕНИЕ ИСХОДНОЙ ЗАДАЧИ", steps)
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
