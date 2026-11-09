import unittest
from contextlib import redirect_stdout
from io import StringIO
import re

import numpy as np

from linear_programming.math_tasks.transport import solve_transport


class TransportTests(unittest.TestCase):
    def run_transport(self, costs, supply, demand):
        output = StringIO()
        with redirect_stdout(output):
            plan = solve_transport(costs, supply, demand)
        return np.asarray(plan), output.getvalue()

    def test_lecture_example_reaches_optimum_with_potentials(self):
        plan, output = self.run_transport(
            [[4, 8, 10, 5], [1, 6, 2, 3], [4, 4, 6, 5]],
            [160, 30, 90],
            [100, 40, 80, 60],
        )

        np.testing.assert_array_equal(
            plan,
            [[100, 0, 0, 60], [0, 0, 30, 0], [0, 40, 50, 0]],
        )
        self.assertIn("Минимальная стоимость перевозок: 1220", output)
        self.assertIn("оценка Δc=-4", output)
        self.assertIn("Все оценки Δc[i,j] неотрицательны: план оптимален.", output)
        self.assertNotIn("| Запасы ", output)
        self.assertEqual(output.count("Метод северо-западного угла"), 1)
        self.assertIn("\033[38;2;128;0;0m-4\033[0m", output)
        self.assertIn("Входящая клетка: (1, 4)", output)
        self.assertIn("Цикл перераспределения: +(1,4) -(3,4) +(3,3) -(1,3)", output)
        self.assertNotIn("\033[1m", output)
        plain_output = re.sub(r"\033\[[0-9;]*m", "", output)
        self.assertIn("100 (4)", plain_output)
        self.assertIn("- (3)", plain_output)
        self.assertNotIn("0 (3)", plain_output)
        initial_table = output.split("Шаг 2.", 1)[0]
        self.assertNotIn("\033[48;", initial_table)
        first_cycle_table = output.split("Шаг 2.", 1)[1].split("Шаг 3.", 1)[0]
        self.assertGreaterEqual(first_cycle_table.count("\033[48;2;255;242;179m"), 16)
        self.assertNotIn("100 | 4", plain_output)

    def test_unbalanced_problem_adds_dummy_supplier_or_consumer(self):
        plan, output = self.run_transport([[5, 8]], [3], [1, 4])
        np.testing.assert_array_equal(plan, [[1, 2], [0, 2]])
        self.assertIn("добавлен фиктивный склад", output)
        self.assertIn("Минимальная стоимость перевозок: 21", output)

        plan, output = self.run_transport([[5], [8]], [1, 4], [3])
        np.testing.assert_array_equal(plan, [[1, 0], [2, 2]])
        self.assertIn("добавлен фиктивный потребитель", output)
        self.assertIn("Минимальная стоимость перевозок: 21", output)

    def test_degenerate_northwest_plan_is_completed_and_optimized(self):
        plan, output = self.run_transport([[4, 1], [2, 3]], [1, 1], [1, 1])

        np.testing.assert_array_equal(plan, [[0, 1], [1, 0]])
        self.assertIn("Все оценки Δc[i,j] неотрицательны: план оптимален.", output)


if __name__ == "__main__":
    unittest.main()
