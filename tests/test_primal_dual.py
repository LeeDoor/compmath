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

    def test_falls_back_to_artificial_basis_and_continues_primal_dual(self):
        result, steps = solve_primal_dual(
            [[1], [1]],
            [1, 3],
            [1],
            constraint_signs=[">=", "<="],
        )

        self.assertIn(
            "Начинаем использовать метод искусственного базиса (МИБ).",
            steps,
        )
        self.assertEqual(
            steps.count("ПРИВЕДЕНИЕ К КАНОНИЧЕСКОЙ ФОРМЕ"),
            1,
        )
        self.assertIn("МИБ. ПЕРВАЯ ФАЗА. ИТЕРАЦИЯ №1", steps)
        self.assertIn("x* = (x1 = 3)", result)
        self.assertIn("Способ 1 (система):   y* = (y1 = 0, y2 = 1)", result)
        self.assertIn("Способ 2 (обратная):  y* = (y1 = 0, y2 = 1)", result)
        self.assertIn("f(x*) = g(y*) = 3", result)

    def test_mib_fallback_restores_minimization_dual_values(self):
        result, steps = solve_primal_dual(
            [[1], [1]],
            [1, 3],
            [1],
            sense="min",
            constraint_signs=[">=", "<="],
        )

        self.assertIn(
            "Начинаем использовать метод искусственного базиса (МИБ).",
            steps,
        )
        self.assertIn("x* = (x1 = 1)", result)
        self.assertIn("Способ 1 (система):   y* = (y1 = 1, y2 = 0)", result)
        self.assertIn("Способ 2 (обратная):  y* = (y1 = 1, y2 = 0)", result)
        self.assertIn("f(x*) = g(y*) = 1", result)

    def test_simplex_default_result_still_includes_slack_variables(self):
        output = StringIO()
        with redirect_stdout(output):
            solution, steps = solve_simplex([[1]], [2], [1])

        self.assertEqual(solution.shape, (1, 2))
        self.assertEqual(solution.tolist(), [[2.0, 0.0]])
        self.assertIn("СИМПЛЕКС-МЕТОД. ИТЕРАЦИЯ №1", steps)
        self.assertEqual(output.getvalue(), "")

    def test_simplex_table_delta_row_uses_dual_simplex_layout(self):
        _, steps = solve_simplex([[1]], [1], [-1])
        delta_row = next(
            line for line in steps.splitlines() if line.strip().startswith("Δ")
        )

        cells = delta_row.split()
        self.assertEqual(len(cells), 3)
        self.assertEqual(cells[0], "Δ")
        self.assertEqual(cells[1], "1")
        self.assertEqual(cells[2], "0")

    def test_simplex_canonicalizes_and_flips_negative_rhs_equation(self):
        solution, steps = solve_simplex([[-1]], [-2], [-1])

        self.assertEqual(solution.tolist(), [[2.0, 0.0]])
        self.assertIn("ПРИВЕДЕНИЕ К КАНОНИЧЕСКОЙ ФОРМЕ", steps)
        self.assertIn("Уравнение 1 умножено на -1.", steps)
        self.assertIn("x1 - x2 = 2", steps)

    def test_simplex_uses_canonical_signs_and_free_variable_substitution(self):
        solution, steps = solve_simplex(
            [[1]],
            [2],
            [-1],
            constraint_signs=[">="],
            variable_constraints=["free"],
        )

        self.assertEqual(solution.tolist(), [[2.0, 0.0, 0.0]])
        self.assertIn("x'1 - x''1 - x2 = 2", steps)
        self.assertNotIn("Найдено альтернативное оптимальное решение", steps)

    def test_artificial_basis_returns_steps_without_printing(self):
        output = StringIO()
        with redirect_stdout(output):
            solution, steps = solve_artificial_basis([[1]], [2], [1])

        np.testing.assert_array_equal(solution, [[2]])
        self.assertIn("Система после приведения к канонической форме:", steps)
        self.assertTrue(steps.formatted_systems)
        self.assertTrue(steps.formatted_functions)
        self.assertNotIn("1) Умножаем", steps)
        self.assertIn("ИТОГОВЫЙ РЕЗУЛЬТАТ", steps)
        self.assertEqual(output.getvalue(), "")

    def test_artificial_basis_runs_two_simplex_phases(self):
        solution, steps = solve_artificial_basis(
            [[1], [1]],
            [1, 3],
            [1],
            constraint_signs=[">=", "<="],
        )

        np.testing.assert_array_equal(solution, [[3]])
        self.assertIn("Вспомогательная целевая функция: F1 = -w1 -> max", steps)
        self.assertIn("МИБ. ПЕРВАЯ ФАЗА. ИТЕРАЦИЯ №1", steps)
        self.assertIn("Искусственные переменные удалены из таблицы.", steps)
        self.assertIn("ПЕРЕХОД К ИСХОДНОЙ ЦЕЛЕВОЙ ФУНКЦИИ", steps)
        self.assertIn("СИМПЛЕКС-МЕТОД. ИТЕРАЦИЯ №1", steps)

    def test_artificial_basis_outputs_system_after_adding_artificial(self):
        _, steps = solve_artificial_basis(
            [[1], [1]],
            [1, 3],
            [1],
            constraint_signs=[">=", "<="],
        )

        self.assertIn("Система в МИБ:", steps)
        self.assertIn("x1 - x2 + w1 = 1", steps)
        self.assertIn("x1 + x3 = 3", steps)

    def test_artificial_basis_normalizes_rhs_after_canonicalization(self):
        solution, steps = solve_artificial_basis(
            [[-1]],
            [-2],
            [-1],
            constraint_signs=["<="],
        )

        np.testing.assert_array_equal(solution, [[2]])
        canonical_form_index = steps.index(
            "Система после приведения к канонической форме:"
        )
        normalization_index = steps.index("Система после нормализации b:")
        self.assertLess(canonical_form_index, normalization_index)
        self.assertIn("-x1 + x2 = -2", steps)
        self.assertIn("x1 - x2 = 2", steps)
        self.assertEqual(len(steps.formatted_systems), 2)

    def test_artificial_basis_reports_infeasible_problem(self):
        solution, steps = solve_artificial_basis(
            [[1], [1]],
            [2, 1],
            [1],
            constraint_signs=[">=", "<="],
        )

        self.assertEqual(solution.shape, (1, 0))
        self.assertIn("Задача несовместна", steps)

    def test_artificial_basis_removes_redundant_constraint(self):
        solution, steps = solve_artificial_basis(
            [[1, 1], [2, 2]],
            [2, 4],
            [1, 1],
            constraint_signs=["=", "="],
        )

        self.assertAlmostEqual(float(solution[0, 0] + solution[0, 1]), 2)
        self.assertIn("линейно зависима и удаляется", steps)


if __name__ == "__main__":
    unittest.main()
