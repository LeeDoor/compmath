"""Локальный HTTP-сервер для веб-интерфейса линейного программирования."""

from __future__ import annotations

import argparse
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import mimetypes
from pathlib import Path
import re
import sys
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from linear_programming.math_tasks.artificial_basis import solve_artificial_basis
from linear_programming.math_tasks.canonical_form import run_canonical_algorithm
from linear_programming.math_tasks.dual_simplex import solve_dual_simplex
from linear_programming.math_tasks.optimality import check_optimality
from linear_programming.math_tasks.primal_dual import solve_primal_dual
from linear_programming.math_tasks.simplex import solve_simplex
from linear_programming.math_tasks.steps import (
    FormattedFunction,
    FormattedSystem,
    ReportText,
    SimplexTableStyle,
)
from linear_programming.math_tasks.standard_form import to_standard_form
from linear_programming.math_tasks.transport import solve_transport
from linear_programming.problem import create_problem
from linear_programming.display_methods import (
    format_math_function,
    format_system_of_equations,
)


WEB_ROOT = Path(__file__).parent / "web"
MAX_BODY_SIZE = 1_000_000
MAX_DIMENSION = 20
MAX_FORM_DIMENSION = 8

TASKS = [
    {
        "id": "primal_dual",
        "title": "Одновременное решение прямой и двойственной задач",
        "input": "linear",
    },
    {"id": "simplex", "title": "Симплекс-метод", "input": "linear"},
    {"id": "transport", "title": "Транспортная задача", "input": "transport"},
    {"id": "dual_simplex", "title": "Двойственный симплекс-метод", "input": "linear"},
    {
        "id": "artificial_basis",
        "title": "Метод искусственного базиса",
        "input": "linear",
    },
    {
        "id": "optimality",
        "title": "Проверка на оптимальность",
        "input": "linear",
        "requires_solution": True,
    },
    {
        "id": "canonical_form",
        "title": "Приведение к канонической форме",
        "input": "linear",
    },
    {
        "id": "standard_form",
        "title": "Приведение к стандартной форме",
        "input": "linear",
    },
]
TASK_BY_ID = {task["id"]: task for task in TASKS}
CONSTRAINT_TASKS = frozenset(
    {
        "primal_dual",
        "simplex",
        "dual_simplex",
        "artificial_basis",
        "canonical_form",
        "standard_form",
    }
)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _validate_vector(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or not value or len(value) > MAX_DIMENSION:
        raise ValueError(f"{name}: ожидается список из 1–{MAX_DIMENSION} чисел.")
    return [_validate_number(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _validate_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: значение должно быть числом.")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name}: значение должно быть конечным числом.") from None
    if not math.isfinite(number):
        raise ValueError(f"{name}: значение должно быть конечным числом.")
    return number


def _validate_matrix(value: Any, name: str) -> list[list[float]]:
    if (
        not isinstance(value, list)
        or not value
        or len(value) > MAX_DIMENSION
        or not all(isinstance(row, list) for row in value)
    ):
        raise ValueError(f"{name}: ожидается непустая матрица.")
    width = len(value[0])
    if not width or width > MAX_DIMENSION or any(len(row) != width for row in value):
        raise ValueError(f"{name}: строки матрицы должны иметь одинаковую длину.")
    return [
        [
            _validate_number(item, f"{name}[{row_index}][{column_index}]")
            for column_index, item in enumerate(row)
        ]
        for row_index, row in enumerate(value)
    ]


def _validate_choices(
    value: Any,
    name: str,
    allowed: set[str],
    expected_length: int,
    default: str,
) -> list[str]:
    if value is None:
        return [default] * expected_length
    if (
        not isinstance(value, list)
        or len(value) != expected_length
        or any(not isinstance(item, str) or item not in allowed for item in value)
    ):
        options = ", ".join(sorted(allowed))
        raise ValueError(f"{name}: ожидается {expected_length} значений из {options}.")
    return value


def _run_solver(function: Any, *args: Any) -> str:
    result = function(*args)
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[1], str)
    ):
        value, steps = result
        returned = value if isinstance(value, str) else (
            str(value) if value is not None else ""
        )
        text = "\n\n".join(part for part in (steps, returned) if part)
        return ReportText(
            text,
            formatted_systems=getattr(steps, "formatted_systems", ()),
            formatted_functions=getattr(steps, "formatted_functions", ()),
            simplex_table_styles=getattr(steps, "simplex_table_styles", ()),
        )

    returned = result if isinstance(result, str) else (
        str(result) if result is not None else ""
    )
    return returned


def solve_task(payload: Any) -> str:
    """Validate an API request and return the selected algorithm's output."""
    if not isinstance(payload, dict):
        raise ValueError("Тело запроса должно быть JSON-объектом.")

    task_id = payload.get("task_id")
    task = TASK_BY_ID.get(task_id) if isinstance(task_id, str) else None
    if task is None:
        raise ValueError("Неизвестный тип задачи.")

    if task["input"] == "transport":
        costs = _validate_matrix(payload.get("costs"), "costs")
        supply = _validate_vector(payload.get("supply"), "supply")
        demand = _validate_vector(payload.get("demand"), "demand")
        if len(costs) != len(supply) or len(costs[0]) != len(demand):
            raise ValueError(
                "Размеры не совпадают: запасы соответствуют строкам, "
                "а потребности — столбцам матрицы тарифов."
            )
        return _run_solver(solve_transport, costs, supply, demand)

    matrix = _validate_matrix(payload.get("A"), "A")
    right_hand_side = _validate_vector(payload.get("b"), "b")
    objective = _validate_vector(payload.get("c"), "c")
    if len(matrix) != len(right_hand_side) or len(matrix[0]) != len(objective):
        raise ValueError("Размеры A, b и c не соответствуют друг другу.")
    sense = payload.get("sense", "max")
    if sense not in ("max", "min"):
        raise ValueError("Направление целевой функции должно быть max или min.")

    if task_id == "optimality":
        solution = _validate_vector(payload.get("solution"), "solution")
        if len(solution) != len(objective):
            raise ValueError("Длина вектора решения должна совпадать с числом неизвестных.")
        problem = create_problem(matrix, right_hand_side, objective, sense)
        return _run_solver(check_optimality, problem, solution)

    solver_args: list[Any] = [matrix, right_hand_side, objective, sense]
    if task_id in CONSTRAINT_TASKS:
        constraint_signs = _validate_choices(
            payload.get("constraint_signs"),
            "constraint_signs",
            {"<=", ">=", "="},
            len(matrix),
            "<=",
        )
        variable_constraints = _validate_choices(
            payload.get("variable_constraints"),
            "variable_constraints",
            {"nonnegative", "free"},
            len(objective),
            "nonnegative",
        )
        solver_args.extend([constraint_signs, variable_constraints])

    solver = {
        "primal_dual": solve_primal_dual,
        "simplex": solve_simplex,
        "dual_simplex": solve_dual_simplex,
        "artificial_basis": solve_artificial_basis,
        "canonical_form": run_canonical_algorithm,
        "standard_form": to_standard_form,
    }[task_id]
    return _run_solver(solver, *solver_args)


def _form_value(values: dict[str, str], name: str, default: str = "0") -> str:
    return values.get(name, default)


def _form_number(values: dict[str, str], name: str, label: str) -> float:
    raw_value = values.get(name, "")
    try:
        value = float(raw_value)
    except ValueError:
        raise ValueError(f"Заполните поле «{label}» числом.") from None
    if not math.isfinite(value):
        raise ValueError(f"Поле «{label}» должно быть конечным числом.")
    return value


def _form_dimension(values: dict[str, str], name: str, label: str) -> int:
    try:
        value = int(values.get(name, "2"))
    except ValueError:
        raise ValueError(f"Количество {label} должно быть от 1 до 8.") from None
    if not 1 <= value <= MAX_FORM_DIMENSION:
        raise ValueError(f"Количество {label} должно быть от 1 до 8.")
    return value


def _select(name: str, label: str, selected: int) -> str:
    options = "".join(
        f'<option value="{number}"{" selected" if number == selected else ""}>'
        f"{number}</option>"
        for number in range(1, MAX_FORM_DIMENSION + 1)
    )
    return (
        f'<label>{label} <select name="{name}">{options}</select></label>'
    )


def _number_input(name: str, value: str, label: str) -> str:
    return (
        f'<input type="number" step="any" name="{html.escape(name)}" '
        f'value="{html.escape(value, quote=True)}" aria-label="{html.escape(label, quote=True)}" required>'
    )


def _render_transport_result(result: str) -> str:
    """Render transport solver output as HTML text and semantic tables."""
    lines = result.splitlines()
    rendered: list[str] = []
    index = 0
    solution_rendered = False

    def is_table_line(line: str) -> bool:
        return line.startswith("|") or re.fullmatch(r"\+(?:-+\+)+", line) is not None

    def parse_matrix(start: int) -> tuple[list[list[str]], int] | None:
        while start < len(lines) and not lines[start].strip():
            start += 1
        if start == len(lines) or not lines[start].lstrip().startswith("["):
            return None

        matrix_rows = []
        bracket_depth = 0
        number_pattern = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
        while start < len(lines):
            line = lines[start]
            if "[" not in line and "]" not in line:
                return None
            bracket_depth += line.count("[") - line.count("]")
            values = number_pattern.findall(line)
            if values:
                matrix_rows.append([f"{float(value):g}" for value in values])
            start += 1
            if bracket_depth == 0:
                break

        if (
            bracket_depth != 0
            or not matrix_rows
            or any(len(row) != len(matrix_rows[0]) for row in matrix_rows)
        ):
            return None
        return matrix_rows, start

    def render_solution_table(matrix_rows: list[list[str]]) -> str:
        headers = "".join(
            f"<th scope=\"col\">Потребитель {column + 1}</th>"
            for column in range(len(matrix_rows[0]))
        )
        body = "".join(
            "<tr>"
            f"<th scope=\"row\">Склад {row + 1}</th>"
            + "".join(f"<td>{html.escape(value)}</td>" for value in values)
            + "</tr>"
            for row, values in enumerate(matrix_rows)
        )
        return (
            '<section class="transport-solution">'
            '<h3 class="transport-result-heading">Оптимальный план перевозок</h3>'
            '<div class="transport-solution-scroll"><table class="transport-solution-table">'
            f"<thead><tr><th scope=\"col\">Склад</th>{headers}</tr></thead>"
            f"<tbody>{body}</tbody></table></div></section>"
        )

    while index < len(lines):
        if is_table_line(lines[index]):
            table_lines = []
            while index < len(lines) and is_table_line(lines[index]):
                table_lines.append(lines[index])
                index += 1

            rows = [
                line.split("|")[1:-1]
                for line in table_lines
                if line.startswith("|")
            ]
            if rows:
                header_index = next(
                    (
                        row_index
                        for row_index, row in enumerate(rows)
                        if row and not row[0].strip()
                    ),
                    None,
                )
                table = ['<table class="transport-result-table">']
                if header_index is not None:
                    headers = rows.pop(header_index)
                    table.append("<thead><tr>")
                    table.extend(
                        f"<th scope=\"col\">{html.escape(re.sub(_ANSI_ESCAPE, '', cell).strip())}</th>"
                        for cell in headers
                    )
                    table.append("</tr></thead>")
                table.append("<tbody>")
                for row in rows:
                    cells = []
                    for column, cell in enumerate(row):
                        selected = "\x1b[48;2;255;242;179m" in cell
                        text = html.escape(re.sub(_ANSI_ESCAPE, "", cell).strip())
                        css_class = ' class="selected-cell"' if selected else ""
                        if column == 0:
                            cells.append(
                                f'<th scope="row"{css_class}>{text}</th>'
                            )
                        else:
                            cells.append(f"<td{css_class}>{text}</td>")
                    table.append("<tr>" + "".join(cells) + "</tr>")
                table.append("</tbody></table>")
                rendered.append("".join(table))
            continue

        text = re.sub(_ANSI_ESCAPE, "", lines[index]).strip()
        northwest_heading = "Шаг 1. Метод северо-западного угла"
        potential_iteration = re.fullmatch(r"Шаг (\d+)\. Метод потенциалов", text)
        if text == "Оптимальный план перевозок:":
            parsed_matrix = parse_matrix(index + 1)
            if parsed_matrix is not None:
                matrix_rows, index = parsed_matrix
                rendered.append(render_solution_table(matrix_rows))
                solution_rendered = True
                continue
            rendered.append(
                '<p class="transport-result-text">Оптимальный план перевозок:</p>'
            )
            index += 1
            continue
        if solution_rendered and lines[index].lstrip().startswith("["):
            parsed_matrix = parse_matrix(index)
            if parsed_matrix is not None:
                _, index = parsed_matrix
                continue
        total_cost = re.fullmatch(r"Минимальная стоимость перевозок: (.+)", text)
        if total_cost:
            rendered.append(
                '<p class="transport-result-cost">Минимальная стоимость перевозок: '
                f"<strong>{html.escape(total_cost.group(1))}</strong></p>"
            )
            index += 1
            continue
        if text.startswith("Задача закрыта: добавлен фиктивный"):
            rendered.append(
                f'<p class="transport-balance-notice">{html.escape(text)}</p>'
            )
            index += 1
            continue
        if text == northwest_heading:
            rendered.append(
                '<h3 class="transport-result-heading">Метод северо-западного угла</h3>'
            )
            index += 1
            continue
        if potential_iteration:
            rendered.append(
                '<h3 class="transport-result-heading">'
                f"Итерация {int(potential_iteration.group(1)) - 1}. "
                "Метод потенциалов</h3>"
            )
            index += 1
            continue

        while index < len(lines) and not is_table_line(lines[index]):
            text = re.sub(_ANSI_ESCAPE, "", lines[index]).strip()
            if (
                text == "Шаг 1. Метод северо-западного угла"
                or text == "Оптимальный план перевозок:"
                or text.startswith("Задача закрыта: добавлен фиктивный")
                or re.fullmatch(r"Шаг \d+\. Метод потенциалов", text)
                or re.fullmatch(r"Минимальная стоимость перевозок: .+", text)
                or (solution_rendered and lines[index].lstrip().startswith("["))
            ):
                break
            index += 1

    return (
        '<section class="transport-result">'
        + "".join(rendered)
        + "</section>"
    )


def _render_algorithm_result(
    result: str,
    formatted_systems: tuple[FormattedSystem, ...] = (),
    formatted_functions: tuple[FormattedFunction, ...] = (),
    simplex_table_styles: tuple[SimplexTableStyle, ...] = (),
) -> str:
    """Render plain solver output as semantic HTML and tables."""
    lines = [_ANSI_ESCAPE.sub("", line).rstrip() for line in result.splitlines()]
    rendered: list[str] = []
    index = 0

    def split_row(line: str) -> list[str]:
        return [cell.strip() for cell in re.split(r"\s{2,}", line.strip())]

    def render_table(
        header: list[str],
        rows: list[list[str]],
        delta: list[str],
        style: SimplexTableStyle | None,
    ) -> str:
        table = [
            '<div class="algorithm-result-scroll">'
            '<table class="algorithm-result-table"><thead><tr>'
        ]
        for cell in header:
            css_class = (
                ' class="pivot-column"'
                if style is not None and cell == style.pivot_column
                else ""
            )
            table.append(
                f'<th scope="col"{css_class}>{html.escape(cell)}</th>'
            )
        table.append("</tr></thead><tbody>")
        for row in rows:
            pivot_row = style is not None and row[0] == style.pivot_row
            table.append(
                '<tr class="pivot-row">' if pivot_row else "<tr>"
            )
            table.append(f'<th scope="row">{html.escape(row[0])}</th>')
            for index, cell in enumerate(row[1:], start=1):
                css_class = (
                    ' class="pivot-column"'
                    if style is not None
                    and index < len(header)
                    and header[index] == style.pivot_column
                    else ""
                )
                table.append(f"<td{css_class}>{html.escape(cell)}</td>")
            table.append("</tr>")
        table.append("</tbody>")
        if delta:
            table.append("<tfoot><tr>")
            table.append(f'<th scope="row">{html.escape(delta[0])}</th>')
            table.extend(
                f"<td>{html.escape(cell)}</td>" for cell in delta[1:]
            )
            table.append("</tr></tfoot>")
        table.append("</table></div>")
        return "".join(table)

    def is_rule(line: str) -> bool:
        return re.fullmatch(r"[-=]{5,}", line.strip()) is not None

    def is_heading(line: str) -> bool:
        text = line.strip()
        return (
            (text.endswith(":") and len(text) <= 180)
            or re.match(r"^\d+\)\s+", text) is not None
            or "ИТЕРАЦИЯ" in text.upper()
            or "СИМПЛЕКСНАЯ ТАБЛИЦА" in text.upper()
            or text.upper() in ("ИТОГОВЫЙ РЕЗУЛЬТАТ", "ИТОГОВОЕ РЕШЕНИЕ")
        )

    def is_system_heading(line: str) -> bool:
        text = line.strip().rstrip(":")
        return (
            text.startswith("Система уравнений")
            or text.startswith("Система после приведения")
            or text.startswith("Система после нормализации")
            or text.startswith("Система в МИБ")
            or text in ("Ограничения", "Добавляем базисные переменные")
        )

    def is_equation(line: str) -> bool:
        return "=" in line and "->" not in line

    function_pattern = re.compile(r"(?:f\(x\)|g\(y\)|[Ff])\s*=", re.IGNORECASE)

    def render_function(expression: str) -> str:
        return next(
            (
                function.html
                for function in formatted_functions
                if function.expression == expression
            ),
            format_math_function(expression),
        )

    table_style_index = 0
    while index < len(lines):
        text = lines[index].strip()
        if not text or is_rule(text):
            index += 1
            continue

        header = split_row(lines[index])
        if header and header[0] == "Базис":
            rows = []
            delta = []
            cursor = index + 1
            while cursor < len(lines):
                row_text = lines[cursor].strip()
                if not row_text:
                    cursor += 1
                    continue
                if is_rule(row_text):
                    cursor += 1
                    continue

                cells = split_row(lines[cursor])
                if cells and cells[0] == "Δ":
                    if len(cells) == len(header) - 1:
                        cells.append("")
                    if len(cells) == len(header):
                        delta = cells
                        cursor += 1
                        continue
                if len(cells) == len(header):
                    rows.append(cells)
                    cursor += 1
                    continue
                break

            if rows:
                style = (
                    simplex_table_styles[table_style_index]
                    if table_style_index < len(simplex_table_styles)
                    else None
                )
                rendered.append(render_table(header, rows, delta, style))
                table_style_index += 1
                index = cursor
                continue

        function_match = function_pattern.search(text)
        if function_match:
            prefix = text[:function_match.start()].strip()
            expression = text[function_match.start():].strip()
            if prefix:
                if is_heading(prefix):
                    rendered.append(
                        '<h3 class="algorithm-result-heading">'
                        f"{html.escape(prefix.rstrip(':'))}</h3>"
                    )
                else:
                    rendered.append(
                        f'<p class="algorithm-result-text">{html.escape(prefix)}</p>'
                    )
            rendered.append(render_function(expression))
            index += 1
            continue

        if is_heading(text):
            rendered.append(
                '<h3 class="algorithm-result-heading">'
                f"{html.escape(text.rstrip(':'))}</h3>"
            )
            index += 1
            if not is_system_heading(text):
                continue

            equations = []
            while index < len(lines):
                equation = lines[index].strip()
                if not equation:
                    index += 1
                    continue
                if equation.startswith("F =") and "->" in equation:
                    rendered.append(render_function(equation))
                    index += 1
                    continue
                if not is_equation(equation):
                    break
                equations.append(equation)
                index += 1
            if equations:
                formatted = next(
                    (
                        system.html
                        for system in formatted_systems
                        if system.equations == tuple(equations)
                    ),
                    None,
                )
                rendered.append(
                    formatted or format_system_of_equations(equations)
                )
            continue
        else:
            rendered.append(
                f'<p class="algorithm-result-text">{html.escape(text)}</p>'
            )
        index += 1

    return '<section class="algorithm-result">' + "".join(rendered) + "</section>"


def render_task_page(
    task: dict[str, Any],
    values: dict[str, str],
    result: str | None = None,
    error: str | None = None,
) -> str:
    """Render a plain HTML condition form, optionally with its result."""
    task_id = task["id"]
    storage_key = task_id
    form = [
        f'<form method="post" action="/solve-page" data-storage-key="{storage_key}" '
        f'data-resize-url="/task/{task_id}">',
        f'<input type="hidden" name="task_id" value="{task_id}">',
    ]

    if task["input"] == "transport":
        warehouses = _form_dimension(values, "warehouses", "складов")
        customers = _form_dimension(values, "customers", "потребителей")
        form.append(
            '<p class="dimensions">'
            + _select("warehouses", "Складов:", warehouses)
            + _select("customers", "Потребителей:", customers)
            + "</p>"
        )
        columns = "".join(
            f'<th>Потребитель {column + 1}<br>'
            f'{_number_input(f"demand_{column}", _form_value(values, f"demand_{column}"), f"Потребность потребителя {column + 1}")}'
            "</th>"
            for column in range(customers)
        )
        rows = []
        for row in range(warehouses):
            costs = "".join(
                f'<td>{_number_input(f"cost_{row}_{column}", _form_value(values, f"cost_{row}_{column}"), f"Тариф со склада {row + 1} потребителю {column + 1}")}</td>'
                for column in range(customers)
            )
            stock = _number_input(
                f"supply_{row}",
                _form_value(values, f"supply_{row}"),
                f"Запас склада {row + 1}",
            )
            rows.append(
                f"<tr><th>Склад {row + 1}<br>{stock}</th>{costs}</tr>"
            )
        form.append(
            '<table class="transport-table"><thead><tr><th></th>'
            + columns
            + "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
    else:
        variables = _form_dimension(values, "variables", "неизвестных")
        constraints = _form_dimension(values, "constraints", "ограничений")
        has_variable_constraints = task_id in CONSTRAINT_TASKS
        form.append(
            '<p class="dimensions">'
            + _select("variables", "Неизвестных:", variables)
            + _select("constraints", "Ограничений:", constraints)
            + "</p>"
        )
        rows = []
        for row in range(constraints):
            if has_variable_constraints:
                relation = _form_value(values, f"sign_{row}", "<=")
                relation_cell = (
                    f'<select name="sign_{row}" aria-label="Знак ограничения {row + 1}">'
                    + "".join(
                        f'<option value="{html.escape(sign, quote=True)}"'
                        f'{" selected" if relation == sign else ""}>'
                        f'{html.escape(sign)}</option>'
                        for sign in ("<=", ">=", "=")
                    )
                    + "</select>"
                )
            else:
                relation_cell = "≤"
            coefficients = '<td class="operator plus">+</td>'.join(
                "<td>"
                + _number_input(
                    f"a_{row}_{column}",
                    _form_value(values, f"a_{row}_{column}"),
                    f"Коэффициент x{column + 1}, ограничение {row + 1}",
                )
                + "</td>"
                for column in range(variables)
            )
            rows.append(
                f"<tr><th>Огр. {row + 1}</th>{coefficients}"
                f'<td class="operator relation">{relation_cell}</td><td>{_number_input(f"b_{row}", _form_value(values, f"b_{row}"), f"Правая часть ограничения {row + 1}")}</td></tr>'
            )
        form.append(
            '<table class="equation-table"><thead><tr><th></th>'
            + "".join(
                f"<th>x{column + 1}</th>"
                + (
                    '<th class="operator plus"></th>'
                    if column < variables - 1
                    else ""
                )
                for column in range(variables)
            )
            + (
                '<th class="operator relation">Знак</th>'
                if has_variable_constraints
                else '<th class="operator relation">≤</th>'
            )
            + "<th>b</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
        if has_variable_constraints:
            variable_buttons = []
            for column in range(variables):
                constraint = _form_value(
                    values, f"x_constraint_{column}", "nonnegative"
                )
                label = (
                    f"∀ x{column + 1}"
                    if constraint == "free"
                    else f"x{column + 1} ≥ 0"
                )
                variable_buttons.append(
                    f'<input type="hidden" name="x_constraint_{column}" '
                    f'value="{html.escape(constraint, quote=True)}">'
                    f'<button type="button" class="variable-constraint" '
                    f'data-variable-index="{column}">{label}</button>'
                )
            form.append(
                '<div class="variable-constraints" aria-label="Ограничения переменных">'
                + "".join(variable_buttons)
                + "</div>"
            )
        objective = "".join(
            (
                f'<label>{_number_input(f"c_{column}", _form_value(values, f"c_{column}"), f"Коэффициент функции при x{column + 1}")}'
                f" x{column + 1}</label>"
                + (" + " if column < variables - 1 else "")
            )
            for column in range(variables)
        )
        form.append(
            f'<p>f(x) = {objective} '
            '<label>Направление '
            f'<select name="sense"><option value="max"{" selected" if _form_value(values, "sense", "max") == "max" else ""}>max</option>'
            f'<option value="min"{" selected" if _form_value(values, "sense", "max") == "min" else ""}>min</option>'
            "</select></label></p>"
        )
        if task.get("requires_solution"):
            solution = "".join(
                f'<label>x{column + 1} '
                f'{_number_input(f"solution_{column}", _form_value(values, f"solution_{column}"), f"Компонент решения x{column + 1}")}</label>'
                for column in range(variables)
            )
            form.append(f"<p>Решение: {solution}</p>")

    form.append('<p><button type="submit">Решить</button></p></form>')
    if error:
        form.append(f'<p class="error">{html.escape(error)}</p>')
    if result is not None:
        form.append('<h2 class="result-heading">Результат</h2>')
        if task_id == "transport":
            form.append(_render_transport_result(result))
        else:
            form.append(
                _render_algorithm_result(
                    result,
                    getattr(result, "formatted_systems", ()),
                    getattr(result, "formatted_functions", ()),
                    getattr(result, "simplex_table_styles", ()),
                )
            )

    return (
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(task['title'])}</title>"
        '<link rel="stylesheet" href="/styles.css">'
        '<script src="/storage.js" defer></script></head><body><main>'
        f'<a class="back-button" href="/">← Назад к задачам</a>'
        f'<h1>{html.escape(task["title"])}</h1>'
        + "".join(form)
        + "</main></body></html>"
    )


def _form_payload(values: dict[str, str]) -> dict[str, Any]:
    task_id = values.get("task_id", "")
    task = TASK_BY_ID.get(task_id)
    if task is None:
        raise ValueError("Неизвестный тип задачи.")

    if task["input"] == "transport":
        warehouses = _form_dimension(values, "warehouses", "складов")
        customers = _form_dimension(values, "customers", "потребителей")
        return {
            "task_id": task_id,
            "costs": [
                [
                    _form_number(values, f"cost_{row}_{column}", "тариф")
                    for column in range(customers)
                ]
                for row in range(warehouses)
            ],
            "supply": [
                _form_number(values, f"supply_{row}", "запас")
                for row in range(warehouses)
            ],
            "demand": [
                _form_number(values, f"demand_{column}", "потребность")
                for column in range(customers)
            ],
        }

    variables = _form_dimension(values, "variables", "неизвестных")
    constraints = _form_dimension(values, "constraints", "ограничений")
    payload: dict[str, Any] = {
        "task_id": task_id,
        "A": [
            [
                _form_number(values, f"a_{row}_{column}", "коэффициент")
                for column in range(variables)
            ]
            for row in range(constraints)
        ],
        "b": [
            _form_number(values, f"b_{row}", "правая часть")
            for row in range(constraints)
        ],
        "c": [
            _form_number(values, f"c_{column}", "коэффициент функции")
            for column in range(variables)
        ],
        "sense": values.get("sense", "max"),
    }
    if task_id in CONSTRAINT_TASKS:
        payload["constraint_signs"] = [
            values.get(f"sign_{row}", "<=") for row in range(constraints)
        ]
        payload["variable_constraints"] = [
            values.get(f"x_constraint_{column}", "nonnegative")
            for column in range(variables)
        ]
    if task.get("requires_solution"):
        payload["solution"] = [
            _form_number(values, f"solution_{column}", "компонент решения")
            for column in range(variables)
        ]
    return payload


class CompmathHandler(BaseHTTPRequestHandler):
    server_version = "CompmathHTTP/1.0"

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", body)

    def do_GET(self) -> None:
        parsed_url = urlsplit(self.path)
        path = parsed_url.path
        if path == "/api/tasks":
            self._send_json(200, {"tasks": TASKS})
            return
        if path.startswith("/api/"):
            self._send_json(404, {"error": "API-маршрут не найден."})
            return
        if path.startswith("/task/"):
            task_id = unquote(path.removeprefix("/task/"))
            query = {
                name: entries[-1]
                for name, entries in parse_qs(
                    parsed_url.query, keep_blank_values=True
                ).items()
            }
            task = TASK_BY_ID.get(task_id) or TASK_BY_ID.get(
                query.get("task_id", "")
            )
            if task is None:
                self._send(404, "text/plain; charset=utf-8", b"Task not found")
                return
            values = query
            try:
                page = render_task_page(task, values).encode("utf-8")
            except ValueError as error:
                page = render_task_page(task, {}, error=str(error)).encode("utf-8")
            self._send(200, "text/html; charset=utf-8", page)
            return

        relative_path = "index.html" if path == "/" else unquote(path.lstrip("/"))
        file_path = (WEB_ROOT / relative_path).resolve()
        try:
            file_path.relative_to(WEB_ROOT.resolve())
        except ValueError:
            self._send(404, "text/plain; charset=utf-8", b"Not found")
            return
        if not file_path.is_file():
            self._send(404, "text/plain; charset=utf-8", b"Not found")
            return

        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in (
            "application/javascript",
            "image/svg+xml",
        ):
            content_type += "; charset=utf-8"
        self._send(200, content_type, file_path.read_bytes())

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path not in ("/api/solve", "/solve-page"):
            self._send_json(404, {"error": "API-маршрут не найден."})
            return

        raw_length = self.headers.get("Content-Length")
        try:
            content_length = int(raw_length or "")
        except ValueError:
            self._send_json(400, {"error": "Не указан корректный Content-Length."})
            return
        if content_length < 0 or content_length > MAX_BODY_SIZE:
            self._send_json(413, {"error": "Размер запроса превышает допустимый."})
            return

        if path == "/solve-page":
            self._handle_form_post(content_length)
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
            result = solve_task(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "Тело запроса должно быть корректным JSON."})
            return
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
            return
        except Exception:
            logging.exception("Ошибка при решении задачи")
            self._send_json(500, {"error": "Не удалось выполнить задачу."})
            return
        self._send_json(200, {"result": result})

    def _handle_form_post(self, content_length: int) -> None:
        values: dict[str, str] = {}
        try:
            encoded = self.rfile.read(content_length).decode("utf-8")
            values = {
                name: entries[-1]
                for name, entries in parse_qs(
                    encoded, keep_blank_values=True
                ).items()
            }
            task = TASK_BY_ID.get(values.get("task_id", ""))
            if task is None:
                raise ValueError("Неизвестный тип задачи.")
            payload = _form_payload(values)
            result = solve_task(payload)
            page = render_task_page(task, values, result=result)
            self._send(200, "text/html; charset=utf-8", page.encode("utf-8"))
        except ValueError as error:
            task = TASK_BY_ID.get(values.get("task_id", ""))
            if task is None:
                self._send(400, "text/plain; charset=utf-8", str(error).encode("utf-8"))
                return
            try:
                page = render_task_page(task, values, error=str(error))
                self._send(400, "text/html; charset=utf-8", page.encode("utf-8"))
            except ValueError:
                self._send(400, "text/plain; charset=utf-8", str(error).encode("utf-8"))
        except Exception:
            logging.exception("Ошибка при решении задачи из формы")
            self._send(
                500,
                "text/plain; charset=utf-8",
                "Не удалось выполнить задачу.".encode("utf-8"),
            )

    def log_message(self, format_string: str, *args: Any) -> None:
        sys.stderr.write(
            "%s - - [%s] %s\n"
            % (self.address_string(), self.log_date_time_string(), format_string % args)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Запустить веб-интерфейс Compmath.")
    parser.add_argument("--port", type=int, default=8000, help="порт сервера (по умолчанию: 8000)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("порт должен быть в диапазоне от 1 до 65535")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), CompmathHandler)
    print(f"Веб-интерфейс: http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
