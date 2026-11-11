"""Решение задачи линейного программирования методом искусственного базиса."""

import numpy as np
import numpy.typing as npt

from ..display_methods import format_math_function, format_system_of_equations
from ..problem import Sense, create_problem
from .canonical_form import format_equation, make_canonical
from .simplex import (
    ConstraintSign,
    VariableConstraint,
    _run_simplex_tableau_impl,
    append_simplex_table,
)
from .steps import StepReport


def _pivot(tableau: np.ndarray, row: int, column: int, eps: float) -> None:
    """Шаг симплекса (пересчет таблицы)."""
    pivot_value = tableau[row, column]
    if abs(pivot_value) <= eps:
        raise ValueError("Невозможно выполнить преобразование: разрешающий элемент равен нулю.")

    tableau[row, :] /= pivot_value
    for i in range(tableau.shape[0] - 1):
        if i == row:
            continue
        factor = tableau[i, column]
        if abs(factor) > eps:
            tableau[i, :] -= factor * tableau[row, :]


def _recompute_objective_row(
    tableau: np.ndarray,
    basis: list[int],
    costs: np.ndarray,
    eps: float,
) -> None:
    """Пересчитать строку оценок Δ = c_B·A - c и значение F = c_B·b."""
    row_count = tableau.shape[0] - 1
    basic_costs = np.array([costs[index] for index in basis])
    tableau[-1, :-1] = basic_costs @ tableau[:row_count, :-1] - costs
    tableau[-1, -1] = float(basic_costs @ tableau[:row_count, -1])
    tableau[np.abs(tableau) < eps] = 0.0


def _find_initial_basis(A: np.ndarray, eps: float) -> tuple[list[int], list[int]]:
    """Найти единичные столбцы; для остальных строк понадобятся искусственные переменные."""
    row_count, variable_count = A.shape
    basis: list[int | None] = [None] * row_count
    used_columns: set[int] = set()

    for column in range(variable_count):
        values = A[:, column]
        ones = np.where(np.abs(values - 1.0) < eps)[0]
        zeros = np.where(np.abs(values) < eps)[0]
        if len(ones) != 1 or len(zeros) != row_count - 1:
            continue
        row = int(ones[0])
        if basis[row] is not None or column in used_columns:
            continue
        basis[row] = column
        used_columns.add(column)

    artificial_rows = [row for row, value in enumerate(basis) if value is None]
    artificial_start = variable_count
    for offset, row in enumerate(artificial_rows):
        basis[row] = artificial_start + offset

    return [int(value) for value in basis], artificial_rows


def _solve_artificial_basis_impl(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
    constraint_signs: list[ConstraintSign] | None = None,
    variable_constraints: list[VariableConstraint] | None = None,
    *,
    report: StepReport,
    return_duals: bool = False,
) -> np.matrix | tuple[np.matrix, np.ndarray, np.ndarray]:
    eps = 1e-9
    problem = create_problem(A, b, c, sense)
    A_original = np.asarray(problem.A, dtype=float)
    b_original = np.asarray(problem.b, dtype=float).reshape(-1)
    c_original = np.asarray(problem.c, dtype=float).reshape(-1)
    original_variable_count = A_original.shape[1]
    if variable_constraints is None:
        variable_constraints = ["nonnegative"] * original_variable_count

    A_can, b_can, c_can, decision_count = make_canonical(
        A_original,
        b_original,
        c_original,
        sense,
        constraint_signs,
        variable_constraints,
    )
    A_can = np.asarray(A_can, dtype=float)
    b_can = np.asarray(b_can, dtype=float).reshape(-1)
    c_can = np.asarray(c_can, dtype=float).reshape(-1)
    variable_count = A_can.shape[1]
    canonical_names = []
    for index, option in enumerate(variable_constraints):
        if option == "free":
            canonical_names.extend((f"x'{index + 1}", f"x''{index + 1}"))
        else:
            canonical_names.append(f"x{index + 1}")
    canonical_names.extend(
        f"x{index}"
        for index in range(
            original_variable_count + 1,
            original_variable_count + variable_count - decision_count + 1,
        )
    )

    report.add("\nПРИВЕДЕНИЕ К КАНОНИЧЕСКОЙ ФОРМЕ")
    canonical_function = (
        f"F = {format_equation(c_can, canonical_names)} -> max"
    )
    report.add_formatted_function(
        canonical_function,
        format_math_function(canonical_function),
    )
    canonical_equations = [
        f"{format_equation(row, canonical_names)} = {value:g}"
        for row, value in zip(A_can, b_can)
    ]
    report.add_formatted_system(
        canonical_equations,
        format_system_of_equations(canonical_equations),
        heading="Система после приведения к канонической форме:",
    )

    negative_rhs_rows = np.flatnonzero(b_can < -eps)
    retained_rows = list(range(len(b_can)))
    row_signs = [1.0] * len(b_can)
    if len(negative_rhs_rows):
        for row in negative_rhs_rows:
            A_can[row, :] *= -1
            b_can[row] *= -1
            row_signs[row] = -1.0
    b_can[np.abs(b_can) < eps] = 0.0
    A_can[np.abs(A_can) < eps] = 0.0
    if len(negative_rhs_rows):
        normalized_equations = [
            f"{format_equation(row, canonical_names)} = {value:g}"
            for row, value in zip(A_can, b_can)
        ]
        report.add_formatted_system(
            normalized_equations,
            format_system_of_equations(normalized_equations),
            heading="Система после нормализации b:",
        )

    basis, artificial_rows = _find_initial_basis(A_can, eps)
    artificial_count = len(artificial_rows)
    artificial_columns = set(range(variable_count, variable_count + artificial_count))

    if artificial_count:
        artificial_matrix = np.zeros((A_can.shape[0], artificial_count))
        for index, row in enumerate(artificial_rows):
            artificial_matrix[row, index] = 1.0
        A_phase_one = np.column_stack((A_can, artificial_matrix))
    else:
        A_phase_one = A_can.copy()

    total_cols = A_phase_one.shape[1]
    variable_names = canonical_names + [
        f"w{i + 1}" for i in range(artificial_count)
    ]

    tableau = np.zeros((A_can.shape[0] + 1, total_cols + 1))
    tableau[:-1, :-1] = A_phase_one
    tableau[:-1, -1] = b_can

    if artificial_columns:
        phase_one_costs = np.zeros(total_cols)
        phase_one_costs[list(artificial_columns)] = -1.0
        artificial_names = [
            variable_names[column] for column in sorted(artificial_columns)
        ]
        report.add(
            "\nНачальный базис неполный. Добавлены искусственные переменные: "
            + ", ".join(artificial_names)
            + "."
        )
        artificial_equations = [
            f"{format_equation(row, variable_names)} = {value:g}"
            for row, value in zip(A_phase_one, b_can)
        ]
        report.add_formatted_system(
            artificial_equations,
            format_system_of_equations(artificial_equations),
            heading="Система в МИБ:",
        )
        report.add(
            "\nПЕРВАЯ ФАЗА: МАКСИМИЗАЦИЯ ОТРИЦАТЕЛЬНОЙ СУММЫ "
            "ИСКУССТВЕННЫХ ПЕРЕМЕННЫХ"
        )
        report.add(
            "Вспомогательная целевая функция: F1 = -"
            + " - ".join(artificial_names)
            + " -> max"
        )
        _recompute_objective_row(tableau, basis, phase_one_costs, eps)
        tableau, basis, _ = _run_simplex_tableau_impl(
            tableau,
            basis,
            phase_one_costs,
            report,
            variable_names,
            title="МИБ. Первая фаза",
            eps=eps,
        )

        if tableau[-1, -1] < -eps:
            raise ValueError(
                "Задача несовместна: искусственные переменные не удалось "
                "вывести из базиса."
            )

        report.add(
            "\nОптимум первой фазы достигнут; выводим искусственные "
            "переменные из базиса."
        )
        while True:
            row_idx = next(
                (
                    row
                    for row, variable in enumerate(basis)
                    if variable in artificial_columns
                ),
                None,
            )
            if row_idx is None:
                break

            if abs(tableau[row_idx, -1]) > eps:
                raise ValueError(
                    "Задача несовместна: искусственная базисная переменная "
                    "имеет ненулевое значение."
                )
            tableau[row_idx, -1] = 0.0
            candidates = [
                column
                for column in range(variable_count)
                if column not in basis and abs(tableau[row_idx, column]) > eps
            ]
            if candidates:
                entering = candidates[0]
                _pivot(tableau, row_idx, entering, eps)
                basis[row_idx] = entering
                _recompute_objective_row(tableau, basis, phase_one_costs, eps)
                append_simplex_table(
                    report,
                    tableau,
                    basis,
                    variable_names,
                    "Таблица после вывода искусственной переменной из базиса",
                )
            else:
                report.add(
                    f"Строка {row_idx + 1} линейно зависима и удаляется."
                )
                tableau = np.delete(tableau, row_idx, axis=0)
                basis.pop(row_idx)
                retained_rows.pop(row_idx)
                row_signs.pop(row_idx)

        report.add("\nИскусственные переменные удалены из таблицы.")
        keep_columns = [
            column
            for column in range(total_cols)
            if column not in artificial_columns
        ]
        tableau = np.column_stack((tableau[:, keep_columns], tableau[:, -1]))
        variable_names = canonical_names
    else:
        tableau = np.zeros((A_can.shape[0] + 1, variable_count + 1))
        tableau[:-1, :-1] = A_can
        tableau[:-1, -1] = b_can

    if artificial_columns:
        report.add("\nПЕРЕХОД К ИСХОДНОЙ ЦЕЛЕВОЙ ФУНКЦИИ")
    else:
        report.add("\nНачальный базис найден. Выполняем симплекс-метод.")
    tableau, basis, _ = _run_simplex_tableau_impl(
        tableau,
        basis,
        c_can,
        report,
        variable_names,
        title="Симплекс-метод",
        eps=eps,
    )

    # Восстанавливаем решение
    canonical_solution = np.zeros(variable_count)
    for i, variable in enumerate(basis):
        canonical_solution[variable] = tableau[i, -1]
    canonical_solution[np.abs(canonical_solution) < eps] = 0.0

    solution = np.zeros(original_variable_count)
    position = 0
    for i, constraint in enumerate(variable_constraints):
        if constraint == "free":
            solution[i] = canonical_solution[position] - canonical_solution[position + 1]
            position += 2
        else:
            solution[i] = canonical_solution[position]
            position += 1

    objective_value = float(np.dot(c_original, solution))
    report.add("\nИТОГОВЫЙ РЕЗУЛЬТАТ")
    report.add("x* = (" + ", ".join(f"{value:g}" for value in solution) + ")")
    report.add(f"F{'max' if sense == 'max' else 'min'} = {objective_value:g}")

    solution_matrix = np.matrix(solution.reshape(1, -1))
    if not return_duals:
        return solution_matrix

    basis_matrix = A_can[retained_rows][:, basis]
    canonical_dual_system = np.linalg.solve(basis_matrix.T, c_can[basis])
    canonical_dual_inverse = c_can[basis] @ np.linalg.inv(basis_matrix)
    objective_sign = 1.0 if sense == "max" else -1.0
    dual_system = np.zeros(len(b_original))
    dual_inverse = np.zeros(len(b_original))
    for row, original_row in enumerate(retained_rows):
        multiplier = row_signs[row] * objective_sign
        dual_system[original_row] = canonical_dual_system[row] * multiplier
        dual_inverse[original_row] = canonical_dual_inverse[row] * multiplier
    dual_system[np.abs(dual_system) < eps] = 0.0
    dual_inverse[np.abs(dual_inverse) < eps] = 0.0

    return solution_matrix, dual_system, dual_inverse


def solve_artificial_basis(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
    constraint_signs: list[ConstraintSign] | None = None,
    variable_constraints: list[VariableConstraint] | None = None,
) -> tuple[np.matrix, str]:
    """Решить задачу методом искусственного базиса."""
    report = StepReport()
    try:
        result = _solve_artificial_basis_impl(
            A, b, c, sense, constraint_signs, variable_constraints, report=report
        )
    except Exception as e:
        report.add(f"\n\nОШИБКА: {e}")
        result = np.matrix([[]])
    return result, report.get_text()