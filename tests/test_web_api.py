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
        self.assertIn('data-storage-key="simplex"', page)
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

    def test_numeric_fields_default_to_zero(self):
        linear_page = render_task_page(TASK_BY_ID["simplex"], {})
        transport_page = render_task_page(TASK_BY_ID["transport"], {})
        optimality_page = render_task_page(TASK_BY_ID["optimality"], {})

        self.assertIn('name="a_0_0" value="0"', linear_page)
        self.assertIn('name="b_0" value="0"', linear_page)
        self.assertIn('name="c_0" value="0"', linear_page)
        self.assertIn('name="cost_0_0" value="0"', transport_page)
        self.assertIn('name="supply_0" value="0"', transport_page)
        self.assertIn('name="demand_0" value="0"', transport_page)
        self.assertIn('name="solution_0" value="0"', optimality_page)

    def test_task_forms_use_distinct_storage_keys(self):
        simplex_page = render_task_page(TASK_BY_ID["simplex"], {})
        canonical_page = render_task_page(TASK_BY_ID["canonical_form"], {})
        transport_page = render_task_page(TASK_BY_ID["transport"], {})

        self.assertIn('data-storage-key="simplex"', simplex_page)
        self.assertIn('data-storage-key="canonical_form"', canonical_page)
        self.assertIn('data-storage-key="transport"', transport_page)

    def test_constraint_controls_are_added_to_optimality_task(self):
        page = render_task_page(
            TASK_BY_ID["optimality"],
            {"variables": "2", "constraints": "2"},
        )
        self.assertIn('name="sign_0"', page)
        self.assertIn('name="x_constraint_0"', page)
        self.assertIn('data-variable-index="0"', page)

    def test_transport_form_uses_compact_table_class(self):
        page = render_task_page(TASK_BY_ID["transport"], {})
        self.assertIn('class="transport-table"', page)

    def test_transport_result_uses_html_text_and_tables(self):
        result = (
            "Задача закрыта: добавлен фиктивный склад с запасом 1 и нулевыми тарифами.\n"
            "Условие закрытой задачи:\n"
            "Тарифы: [[1.0, 2.0]]\n"
            "Запасы: [2.0]\n"
            "Потребности: [2.0, 3.0]\n"
            "Шаг 1. Метод северо-западного угла\n"
            "+----------+----------+\n"
            "|          | b[1]=2   |\n"
            "+----------+----------+\n"
            "| a[1]=2   | \x1b[48;2;255;242;179m2 (1)\x1b[0m |\n"
            "+----------+----------+\n"
            "Шаг 2. Метод потенциалов\n"
            "Входящая клетка: (1, 2), оценка Δc=-1.\n"
            "Цикл перераспределения: +(1,2) -(1,1)\n"
            "Величина перераспределения Δx=2.\n"
            "Все оценки Δc[i,j] неотрицательны: план оптимален.\n"
            "Оптимальный план перевозок:\n"
            "[[2. 0.]\n"
            " [0. 3.]]\n"
            "Минимальная стоимость перевозок: 17\n"
            "\n"
            "[[2. 0.]\n"
            " [0. 3.]]"
        )
        page = render_task_page(TASK_BY_ID["transport"], {}, result=result)

        self.assertIn(
            '<h3 class="transport-result-heading">Метод северо-западного угла</h3>',
            page,
        )
        self.assertIn(
            '<h3 class="transport-result-heading">Итерация 1. Метод потенциалов</h3>',
            page,
        )
        self.assertNotIn("Шаг 1.", page)
        self.assertNotIn("Шаг 2.", page)
        self.assertIn(
            '<p class="transport-balance-notice">Задача закрыта: добавлен '
            "фиктивный склад с запасом 1 и нулевыми тарифами.</p>",
            page,
        )
        for hidden_text in (
            "Условие закрытой задачи",
            "Тарифы:",
            "Запасы:",
            "Потребности:",
            "Входящая клетка:",
            "Цикл перераспределения:",
            "Величина перераспределения",
            "план оптимален",
        ):
            self.assertNotIn(hidden_text, page)
        self.assertIn('<table class="transport-result-table">', page)
        self.assertIn("<th scope=\"col\">b[1]=2</th>", page)
        self.assertIn('<td class="selected-cell">2 (1)</td>', page)
        self.assertIn('<section class="transport-solution">', page)
        self.assertIn('<table class="transport-solution-table">', page)
        self.assertIn("<th scope=\"col\">Потребитель 1</th>", page)
        self.assertIn("<td>3</td>", page)
        self.assertIn(
            '<p class="transport-result-cost">Минимальная стоимость перевозок: '
            "<strong>17</strong></p>",
            page,
        )
        self.assertNotIn("[[2.", page)
        self.assertNotIn("<textarea", page)

    def test_simplex_result_uses_html_headings_and_table(self):
        result = solve_task(
            {
                "task_id": "simplex",
                "A": [[1]],
                "b": [1],
                "c": [-1],
                "sense": "max",
            }
        )
        page = render_task_page(TASK_BY_ID["simplex"], {}, result=result)

        self.assertTrue(result.formatted_systems)
        self.assertIn('<div class="equation-system"', page)
        self.assertIn('<div class="math-function"', page)
        self.assertIn(
            '<h3 class="algorithm-result-heading">'
            "СИМПЛЕКС-МЕТОД. ИТЕРАЦИЯ №1</h3>",
            page,
        )
        self.assertIn('<table class="algorithm-result-table">', page)
        self.assertIn('<th scope="col">Базис</th>', page)
        self.assertNotIn("<textarea", page)

    def test_simplex_pivot_is_highlighted_and_basis_is_listed(self):
        result = solve_task(
            {
                "task_id": "simplex",
                "A": [[1]],
                "b": [2],
                "c": [1],
            }
        )
        page = render_task_page(TASK_BY_ID["simplex"], {}, result=result)

        self.assertIn("Базисные переменные: x2", result)
        self.assertIn('class="pivot-column"', page)
        self.assertIn('class="pivot-row"', page)
        self.assertEqual(page.count('<table class="algorithm-result-table">'), 2)
        self.assertNotIn("Из базиса выходит", result)
        self.assertNotIn("Система уравнений текущего базиса", result)

    def test_function_expressions_use_html_math_card(self):
        page = render_task_page(
            TASK_BY_ID["standard_form"],
            {},
            result="f(x) = 2*x1 - x2 -> max",
        )

        self.assertIn('<div class="math-function"', page)
        self.assertIn('<var>x</var><sub>1</sub>', page)
        self.assertIn('<var>x</var><sub>2</sub>', page)

    def test_function_outputs_use_math_card_across_html_tasks(self):
        requests = [
            {
                "task_id": "simplex",
                "A": [[1]],
                "b": [2],
                "c": [1],
            },
            {
                "task_id": "primal_dual",
                "A": [[1, 1], [1, 0], [0, 1]],
                "b": [4, 2, 3],
                "c": [3, 2],
            },
            {
                "task_id": "dual_simplex",
                "A": [[1, 1]],
                "b": [2],
                "c": [-2, -1],
                "constraint_signs": [">="],
            },
            {
                "task_id": "optimality",
                "A": [[1]],
                "b": [2],
                "c": [1],
                "solution": [2],
            },
            {
                "task_id": "canonical_form",
                "A": [[1]],
                "b": [2],
                "c": [1],
            },
            {
                "task_id": "standard_form",
                "A": [[1]],
                "b": [2],
                "c": [1],
            },
        ]

        for request in requests:
            with self.subTest(task=request["task_id"]):
                result = solve_task(request)
                page = render_task_page(
                    TASK_BY_ID[request["task_id"]],
                    {},
                    result=result,
                )
                self.assertIn('<div class="math-function"', page)

    def test_simplex_normalized_system_uses_math_html_formatter(self):
        result = solve_task(
            {
                "task_id": "simplex",
                "A": [[-1]],
                "b": [-2],
                "c": [-1],
            }
        )
        page = render_task_page(TASK_BY_ID["simplex"], {}, result=result)

        self.assertIn("Уравнение 1 умножено на -1.", result)
        self.assertTrue(result.formatted_systems)
        self.assertIn('<div class="equation-system"', page)
        self.assertGreaterEqual(page.count('<div class="equation-system"'), 2)
        self.assertIn("Система после нормализации b", page)

    def test_linear_equation_system_uses_math_html_formatting(self):
        result = "\n".join(
            [
                "Система уравнений текущего базиса:",
                "x'1 = 2 - 0.5*x''1",
                "x2 = 3 & 4",
            ]
        )
        page = render_task_page(TASK_BY_ID["simplex"], {}, result=result)

        self.assertIn('<div class="equation-system"', page)
        self.assertIn('<var>x</var><sup>′</sup><sub>1</sub>', page)
        self.assertIn('<var>x</var><sup>″</sup><sub>1</sub>', page)
        self.assertIn("3 &amp; 4", page)
        self.assertEqual(page.count('class="equation-system-row"'), 2)

    def test_simplex_explanatory_sections_are_not_added_to_table(self):
        result = "\n".join(
            [
                "СИМПЛЕКС-МЕТОД. ИТЕРАЦИЯ №1",
                "Симплекс-таблица:",
                f"{'Базис':>8}{'x3':>12}{'x4':>12}{'b':>12}",
                f"{'x1':>8}{0.09259:12.4g}{-0.01852:12.4g}{0.07407:12.4g}",
                f"{'x2':>8}{-0.01852:12.4g}{0.2037:12.4g}{0.1852:12.4g}",
                f"{'Δ':>8}{0.1296:12.4g}{0.5741:12.4g}",
                "Оценки Δ всех переменных:",
                "x1: 0.0000    x2: 0.0000    x3: 0.1296    x4: 0.5741",
                "Система уравнений текущего базиса:",
                "x1 = 0.07407",
            ]
        )
        page = render_task_page(TASK_BY_ID["simplex"], {}, result=result)
        table = page.split(
            '<table class="algorithm-result-table">', 1
        )[1].split("</table>", 1)[0]

        self.assertNotIn("Оценки", table)
        self.assertNotIn("Система уравнений", table)
        self.assertIn(
            '<h3 class="algorithm-result-heading">Оценки Δ всех переменных</h3>',
            page,
        )
        self.assertIn(
            '<h3 class="algorithm-result-heading">Система уравнений текущего базиса</h3>',
            page,
        )

    def test_artificial_basis_uses_html_result_renderer(self):
        page = render_task_page(
            TASK_BY_ID["artificial_basis"], {}, result="Результат"
        )

        self.assertIn('<section class="algorithm-result">', page)
        self.assertNotIn("<textarea", page)

    def test_artificial_basis_canonical_result_uses_html_formatters(self):
        result = solve_task(
            {
                "task_id": "artificial_basis",
                "A": [[1]],
                "b": [2],
                "c": [1],
                "constraint_signs": ["="],
            }
        )
        page = render_task_page(
            TASK_BY_ID["artificial_basis"], {}, result=result
        )

        self.assertTrue(result.formatted_systems)
        self.assertTrue(result.formatted_functions)
        self.assertIn('<div class="equation-system"', page)
        self.assertIn('<div class="math-function"', page)
        self.assertNotIn("Приведение к канонической форме:", page)
        self.assertNotIn("1) Умножаем", page)

    def test_dual_simplex_result_uses_html_table(self):
        result = solve_task(
            {
                "task_id": "dual_simplex",
                "A": [[1, 1]],
                "b": [2],
                "c": [-2, -1],
                "sense": "max",
                "constraint_signs": [">="],
                "variable_constraints": ["nonnegative", "nonnegative"],
            }
        )
        page = render_task_page(TASK_BY_ID["dual_simplex"], {}, result=result)

        self.assertIn(
            '<h3 class="algorithm-result-heading">'
            "Симплекс-таблица, итерация 1</h3>",
            page,
        )
        self.assertIn('<div class="equation-system"', page)
        self.assertIn('<table class="algorithm-result-table">', page)
        self.assertIn('class="pivot-column"', page)
        self.assertIn('class="pivot-row"', page)
        self.assertNotIn("<textarea", page)

    def test_form_result_text_uses_html_and_escapes_contents(self):
        page = render_task_page(
            TASK_BY_ID["canonical_form"],
            {},
            result=(
                "Приведение к канонической форме:\n"
                "1) Преобразование <x>."
            ),
        )

        self.assertIn(
            '<h3 class="algorithm-result-heading">'
            "Приведение к канонической форме</h3>",
            page,
        )
        self.assertIn(
            '<h3 class="algorithm-result-heading">1) Преобразование &lt;x&gt;.</h3>',
            page,
        )
        self.assertNotIn("<textarea", page)

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

    def test_optimality_form_payload_includes_constraint_choices(self):
        payload = _form_payload(
            {
                "task_id": "optimality",
                "variables": "1",
                "constraints": "2",
                "a_0_0": "1",
                "a_1_0": "1",
                "b_0": "1",
                "b_1": "3",
                "c_0": "1",
                "sense": "max",
                "solution_0": "3",
                "sign_0": ">=",
                "sign_1": "<=",
                "x_constraint_0": "free",
            }
        )
        self.assertEqual(payload["constraint_signs"], [">=", "<="])
        self.assertEqual(payload["variable_constraints"], ["free"])

    def test_optimality_solver_receives_constraint_choices(self):
        payload = {
            "task_id": "optimality",
            "A": [[1], [1]],
            "b": [1, 3],
            "c": [1],
            "sense": "max",
            "solution": [3],
            "constraint_signs": [">=", "<="],
            "variable_constraints": ["free"],
        }
        with patch("server.check_optimality", return_value="result") as solver:
            self.assertEqual(solve_task(payload), "result")
        args, _ = solver.call_args
        self.assertEqual(
            args[1:],
            ([3.0], [">=", "<="], ["free"]),
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
