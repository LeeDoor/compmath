import unittest

from linear_programming.math_tasks.optimality import check_optimality
from linear_programming.problem import create_problem


class OptimalityTests(unittest.TestCase):
    def test_rejects_a_primal_infeasible_solution(self):
        problem = create_problem([[1]], [1], [1], "max")

        result = check_optimality(problem, [2])

        self.assertIn("нарушены ограничения стандартной формы", result)
        self.assertIn("план НЕ является оптимальным", result)

    def test_rejects_a_negative_nonnegative_variable(self):
        problem = create_problem([[1]], [0], [-1], "max")

        result = check_optimality(problem, [-1])

        self.assertIn("нарушены условия неотрицательности", result)
        self.assertIn("план НЕ является оптимальным", result)

    def test_finds_optimality_witness_when_least_squares_would_be_negative(self):
        problem = create_problem([[1], [-1]], [1, -1], [1], "max")

        result = check_optimality(problem, [1])

        self.assertIn("Найдено решение системы", result)
        self.assertIn("план x ЯВЛЯЕТСЯ оптимальным", result)

    def test_checks_optimality_when_complementarity_system_is_empty(self):
        problem = create_problem([[1]], [0], [-1], "max")

        result = check_optimality(problem, [0])

        self.assertIn("Система равенств пуста.", result)
        self.assertIn("план x ЯВЛЯЕТСЯ оптимальным", result)

    def test_matches_the_plans_from_the_lecture_example(self):
        problem = create_problem(
            [[1, -8], [-1, 1], [-1, 2]],
            [3, 1, 4],
            [-1, 2],
            "max",
        )

        optimal_result = check_optimality(problem, [4, 4])
        nonoptimal_result = check_optimality(problem, [0, 1])

        self.assertIn("план x ЯВЛЯЕТСЯ оптимальным", optimal_result)
        self.assertIn("план НЕ является оптимальным", nonoptimal_result)


if __name__ == "__main__":
    unittest.main()
