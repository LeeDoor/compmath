import unittest
from contextlib import redirect_stdout
from io import StringIO

from linear_programming.math_tasks.primal_dual import solve_primal_dual
from linear_programming.math_tasks.simplex import solve_simplex


class PrimalDualTests(unittest.TestCase):
    def test_solves_primal_and_dual_from_simplex_result(self):
        output = StringIO()
        with redirect_stdout(output):
            result = solve_primal_dual(
                [[1, 1], [1, 0], [0, 1]],
                [4, 2, 3],
                [3, 2],
            )

        self.assertIn("x* = (x1 = 2, x2 = 2)", result)
        self.assertIn("Способ 1 (система):   y* = (y1 = 2, y2 = 1, y3 = 0)", result)
        self.assertIn("Способ 2 (обратная):  y* = (y1 = 2, y2 = 1, y3 = 0)", result)
        self.assertIn("f(x*) = g(y*) = 10", result)

    def test_simplex_default_result_still_includes_slack_variables(self):
        output = StringIO()
        with redirect_stdout(output):
            solution = solve_simplex([[1]], [2], [1])

        self.assertEqual(solution.shape, (1, 2))
        self.assertEqual(solution.tolist(), [[2.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
