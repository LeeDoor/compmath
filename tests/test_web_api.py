import unittest

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
        self.assertNotIn('name="a_4_0"', page)
        self.assertIn('data-resize-url="/task/simplex"', page)
        self.assertNotIn("Обновить систему", page)

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
