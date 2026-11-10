import unittest

from linear_programming.math_tasks.dual_simplex import solve_dual_simplex
from server import solve_task


class DualSimplexTests(unittest.TestCase):
    def run_solver(self, *args, **kwargs):
        return solve_dual_simplex(*args, **kwargs)

    def test_dual_pivots_by_maximum_delta_over_negative_row_coefficient(self):
        result, output = self.run_solver(
            [[1, 1]],
            [2],
            [-2, -1],
            "max",
            [">="],
            ["nonnegative", "nonnegative"],
        )

        self.assertIn("применяем двойственный симплекс-метод", output)
        self.assertIn("ведущий столбец: x2", output)
        self.assertIn("x* = (0, 2)", result)
        self.assertIn("Значение целевой функции: -2", result)
        canonical_system = output.split(
            "Система после приведения к канонической форме:", 1
        )[1].split("После нормализации базисных переменных", 1)[0]
        equations = [
            line for line in canonical_system.splitlines() if line.endswith(" = -2")
        ]
        self.assertEqual(len(equations), 1)

    def test_initial_table_rows_are_basis_and_columns_are_nonbasis(self):
        _, output = self.run_solver(
            [[1, 1]],
            [2],
            [-2, -1],
            "max",
            [">="],
            ["nonnegative", "nonnegative"],
        )
        system_position = output.index(
            "Система после приведения к канонической форме:"
        )
        table_position = output.index("Симплекс-таблица, итерация 1:")
        self.assertLess(system_position, table_position)

        initial_table = output.split(
            "Симплекс-таблица, итерация 1:",
            1,
        )[1].split("Ведущая строка:", 1)[0]
        header = initial_table.splitlines()[1]
        first_row = initial_table.splitlines()[2]

        self.assertIn("x1", header)
        self.assertIn("x2", header)
        self.assertNotIn("x3", header)
        self.assertIn("x3", first_row)

    def test_nonnegative_basic_rhs_requests_regular_simplex(self):
        result, output = self.run_solver([[1]], [4], [3])

        self.assertIn("Система после приведения к канонической форме:", output)
        self.assertIn(
            "Нет нужды в двойственном симплексе. Используйте обычный симплекс-метод.",
            result,
        )
        self.assertNotIn("Симплекс-таблица", output)

    def test_minimization_objective_is_restored_in_final_result(self):
        result, output = self.run_solver([[1]], [2], [1], "min", [">="])

        self.assertIn("применяем двойственный симплекс-метод", output)
        self.assertIn("x* = (2)", result)
        self.assertIn("Значение целевой функции: 2", result)

    def test_free_variable_uses_canonical_matrix_and_restores_original_value(self):
        result, output = self.run_solver(
            [[1]],
            [2],
            [-1],
            "max",
            [">="],
            ["free"],
        )

        self.assertIn("Система после приведения к канонической форме:", output)
        self.assertIn("x'1", output)
        self.assertIn("x''1", output)
        self.assertIn("x* = (2)", result)
        self.assertIn("Значение целевой функции: -2", result)

    def test_primal_simplex_finishes_when_feasibility_is_restored(self):
        result, output = self.run_solver(
            [[1], [1]],
            [2, 5],
            [1],
            "max",
            [">=", "<="],
            ["nonnegative"],
        )

        self.assertIn("завершаем оптимизацию обычным симплекс-шагом", output)
        self.assertIn("x* = (5)", result)
        self.assertIn("Значение целевой функции: 5", result)

    def test_nonnegative_equality_basis_requests_regular_simplex(self):
        result, output = self.run_solver([[2]], [4], [1], "max", ["="])

        self.assertIn("x1 = 2", output)
        self.assertIn("Нет нужды в двойственном симплексе", result)

    def test_missing_row_basis_requires_artificial_basis_method(self):
        with self.assertRaisesRegex(ValueError, "Нет базиса. Используйте МИБ"):
            solve_dual_simplex(
                [[1, 1], [1, 1]],
                [1, 1],
                [1, 2],
                "max",
                ["=", "="],
            )

    def test_web_solver_passes_constraint_options(self):
        result = solve_task(
            {
                "task_id": "dual_simplex",
                "A": [[1]],
                "b": [2],
                "c": [-1],
                "sense": "max",
                "constraint_signs": [">="],
                "variable_constraints": ["nonnegative"],
            }
        )

        self.assertIn("применяем двойственный симплекс-метод", result)
        self.assertIn("x* = (2)", result)
        self.assertNotIn("Заглушка", result)


if __name__ == "__main__":
    unittest.main()
