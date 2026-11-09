import unittest
from unittest.mock import patch

from server import TASK_BY_ID, _form_payload, render_task_page, solve_task


class WebApiTests(unittest.TestCase):
    def test_linear_task_returns_algorithm_output(self):
        result = solve_task(
            {
                "task_id": "simplex",
                "A": [[1, 1], [1, 0], [0, 1]],
                "b": [4, 2, 3],
                "c": [3, 2],
                "sense": "max",
            }
        )
        self.assertIn("Получено оптимальное решение.", result)
        self.assertIn("x* = (2, 2", result)

    def test_transport_task_returns_algorithm_output(self):
        result = solve_task(
            {
                "task_id": "transport",
                "costs": [[1, 2], [3, 1]],
                "supply": [2, 3],
                "demand": [2, 3],
            }
        )
        self.assertIn("Минимальная стоимость перевозок:", result)

    def test_rejects_mismatched_linear_dimensions(self):
        with self.assertRaisesRegex(ValueError, "Размеры"):
            solve_task(
                {
                    "task_id": "simplex",
                    "A": [[1, 2]],
                    "b": [3, 4],
                    "c": [1, 1],
                }
            )

    def test_optimality_task_requires_a_solution_vector(self):
        with self.assertRaisesRegex(ValueError, "solution"):
            solve_task(
                {
                    "task_id": "optimality",
                    "A": [[1]],
                    "b": [2],
                    "c": [1],
                    "sense": "max",
                }
            )

    def test_selected_dimensions_generate_matching_linear_fields(self):
        page = render_task_page(
            TASK_BY_ID["simplex"],
            {"variables": "3", "constraints": "4"},
        )
        self.assertIn('name="a_3_2"', page)
        self.assertIn('name="b_3"', page)
        self.assertIn('class="back-button" href="/"', page)
        self.assertIn('class="equation-table"', page)
        self.assertIn('class="operator plus"', page)
        self.assertIn('class="operator relation"', page)
        self.assertIn('name="sign_0"', page)
        self.assertIn('name="x_constraint_2"', page)
        self.assertIn('data-variable-index="2"', page)
        self.assertNotIn('name="a_4_0"', page)
        self.assertIn('data-resize-url="/task/simplex"', page)
        self.assertNotIn("Обновить систему", page)

    def test_constraint_controls_are_not_added_to_optimality_task(self):
        page = render_task_page(
            TASK_BY_ID["optimality"],
            {"variables": "2", "constraints": "2"},
        )
        self.assertNotIn('name="sign_0"', page)
        self.assertNotIn('name="x_constraint_0"', page)

    def test_transport_form_uses_compact_table_class(self):
        page = render_task_page(TASK_BY_ID["transport"], {})
        self.assertIn('class="transport-table"', page)

    def test_html_form_payload_includes_signs_and_variable_constraints(self):
        payload = _form_payload(
            {
                "task_id": "simplex",
                "variables": "2",
                "constraints": "2",
                "a_0_0": "1",
                "a_0_1": "1",
                "a_1_0": "1",
                "a_1_1": "0",
                "b_0": "4",
                "b_1": "2",
                "c_0": "3",
                "c_1": "2",
                "sense": "max",
                "sign_0": ">=",
                "sign_1": "=",
                "x_constraint_0": "free",
                "x_constraint_1": "nonnegative",
            }
        )
        self.assertEqual(payload["constraint_signs"], [">=", "="])
        self.assertEqual(
            payload["variable_constraints"], ["free", "nonnegative"]
        )

    def test_solver_receives_constraint_arguments(self):
        payload = {
            "task_id": "simplex",
            "A": [[1]],
            "b": [2],
            "c": [1],
            "sense": "max",
            "constraint_signs": [">="],
            "variable_constraints": ["free"],
        }
        with patch("server.solve_simplex", return_value="result") as solver:
            self.assertEqual(solve_task(payload), "result")
        solver.assert_called_once_with(
            [[1.0]], [2.0], [1.0], "max", [">="], ["free"]
        )

    def test_constraint_arguments_are_validated_for_solver_dimensions(self):
        with self.assertRaisesRegex(ValueError, "constraint_signs"):
            solve_task(
                {
                    "task_id": "simplex",
                    "A": [[1], [2]],
                    "b": [3, 4],
                    "c": [1],
                    "constraint_signs": ["="],
                    "variable_constraints": ["nonnegative"],
                }
            )
        with self.assertRaisesRegex(ValueError, "variable_constraints"):
            solve_task(
                {
                    "task_id": "simplex",
                    "A": [[1]],
                    "b": [2],
                    "c": [1],
                    "constraint_signs": ["<="],
                    "variable_constraints": ["unrestricted"],
                }
            )

    def test_html_form_dimensions_are_sent_to_solver(self):
        values = {
            "task_id": "simplex",
            "variables": "2",
            "constraints": "2",
            "a_0_0": "1",
            "a_0_1": "1",
            "a_1_0": "1",
            "a_1_1": "0",
            "b_0": "4",
            "b_1": "2",
            "c_0": "3",
            "c_1": "2",
            "sense": "max",
        }
        result = solve_task(_form_payload(values))
        self.assertIn("Получено оптимальное решение.", result)


if __name__ == "__main__":
    unittest.main()
