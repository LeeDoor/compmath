"""Решение задачи линейного программирования методом искусственного базиса."""

import numpy as np
import numpy.typing as npt

from ..problem import Sense, create_problem
from .canonical_form import make_canonical, run_canonical_algorithm
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
) -> np.matrix:
    eps = 1e-9
    problem = create_problem(A, b, c, sense)
    A_original = np.asarray(problem.A, dtype=float).copy()
    b_original = np.asarray(problem.b, dtype=float).reshape(-1).copy()
    c_original = np.asarray(problem.c, dtype=float).reshape(-1)
    row_count, original_variable_count = A_original.shape

    if constraint_signs is None:
        signs = ["<="] * row_count
    else:
        signs = list(constraint_signs)

    if variable_constraints is None:
        variable_constraints = ["nonnegative"] * original_variable_count

    # Отрицательные b приводим к положительным
    for i in range(row_count):
        if b_original[i] < -eps:
            A_original[i, :] *= -1
            b_original[i] *= -1
            signs[i] = {"<=": ">=", ">=": "<=", "=": "="}[signs[i]]

    report.add(
        run_canonical_algorithm(
            A_original, b_original, c_original, sense, signs, variable_constraints
        )
    )

    A_can, b_can, c_can, _ = make_canonical(
        A_original, b_original, c_original, sense, signs, variable_constraints
    )
    A_can = np.asarray(A_can, dtype=float)
    b_can = np.asarray(b_can, dtype=float).reshape(-1)
    c_can = np.asarray(c_can, dtype=float).reshape(-1)
    b_can[np.abs(b_can) < eps] = 0.0

    basis, artificial_rows = _find_initial_basis(A_can, eps)
    artificial_count = len(artificial_rows)
    variable_count = A_can.shape[1]
    artificial_columns = set(range(variable_count, variable_count + artificial_count))

    if artificial_count:
        artificial_matrix = np.zeros((A_can.shape[0], artificial_count))
        for index, row in enumerate(artificial_rows):
            artificial_matrix[row, index] = 1.0
        A_phase_one = np.column_stack((A_can, artificial_matrix))
    else:
        A_phase_one = A_can.copy()

    total_cols = A_phase_one.shape[1]
    variable_names = [f"x{i + 1}" for i in range(variable_count)] + [
        f"w{i + 1}" for i in range(artificial_count)
    ]

    phase_one_costs = np.zeros(total_cols)
    for column in artificial_columns:
        phase_one_costs[column] = -1.0

    # --- ПЕРВЫЙ ЭТАП (Ручной, чтобы выводить таблицы как на лекции) ---
    if artificial_columns:
        tableau = np.zeros((A_can.shape[0] + 1, total_cols + 1))
        tableau[:-1, :-1] = A_phase_one
        tableau[:-1, -1] = b_can
        _recompute_objective_row(tableau, basis, phase_one_costs, eps)

        # Выводим начальную таблицу с W
        append_simplex_table(report, tableau, basis, variable_names, "Таблица 1")

        while artificial_columns:
            # Проверяем, не пора ли вывести W
            row_idx = next((i for i, var in enumerate(basis) if var in artificial_columns), None)
            if row_idx is None:
                break # Все W уже не в базисе

            # Делаем W ведущей строкой и "переворачиваем"
            candidates = [
                col for col in range(tableau.shape[1] - 1)
                if col not in artificial_columns and col not in basis and abs(tableau[row_idx, col]) > eps
            ]
            if candidates:
                entering = candidates[0]
                _pivot(tableau, row_idx, entering, eps)
                basis[row_idx] = entering

                # Пересчитываем оценки (пока W еще в таблице, как на Таблице 2)
                _recompute_objective_row(tableau, basis, phase_one_costs, eps)
                # Выводим Таблицу 2
                append_simplex_table(report, tableau, basis, variable_names, "Таблица 2")

                # СРАЗУ ВЫЧЕРКИВАЕМ столбец W
                report.add("\nВычеркиваем столбцы искусственных переменных.")
                keep_cols = [j for j in range(tableau.shape[1] - 1) if j not in artificial_columns]
                tableau = np.column_stack((tableau[:, keep_cols], tableau[:, -1]))

                # Обновляем неймспейс
                artificial_columns = set()
                variable_names = [n for i, n in enumerate(variable_names) if i not in artificial_columns]
                phase_one_costs = np.zeros(tableau.shape[1] - 1)

                # ВАЖНО: Пересчитываем оценки для ИСХОДНОЙ функции (c_can), чтобы в Таблице 3 были правильные Δ
                _recompute_objective_row(tableau, basis, c_can, eps)

                # Выводим Таблицу 3
                append_simplex_table(report, tableau, basis, variable_names, "Таблица 3")

            else:
                # Если нулевая строка, удаляем её
                tableau = np.delete(tableau, row_idx, axis=0)
                basis.pop(row_idx)
                break

        # Синхронизируем costs для второго этапа
        _recompute_objective_row(tableau, basis, c_can, eps)

    else:
        # Если W не было, просто строим таблицу
        tableau = np.zeros((A_can.shape[0] + 1, variable_count + 1))
        tableau[:-1, :-1] = A_can
        tableau[:-1, -1] = b_can
        _recompute_objective_row(tableau, basis, c_can, eps)

    # --- ВТОРОЙ ЭТАП (Используем ваш simplex.py) ---
    phase_two_names = [f"x{i + 1}" for i in range(variable_count)]

    # Запускаем ваш симплекс.
    result = _run_simplex_tableau_impl(
        tableau, basis, c_can, report, phase_two_names, title=""
    )

    # Распаковываем безопасно (вдруг вернется 3 или 6 элементов)
    is_unbounded = False
    unbounded_sol = None

    if len(result) == 3:
        tableau, basis, _ = result
    elif len(result) >= 5:
        tableau, basis, _, is_unbounded, unbounded_sol = result[:5]
        # Исправление ошибки "truth value of an array": проверяем строго на True
        if is_unbounded is True:
            solution = np.zeros(original_variable_count)
            position = 0
            for i, constraint in enumerate(variable_constraints):
                if constraint == "free":
                    solution[i] = unbounded_sol[position] - unbounded_sol[position + 1]
                    position += 2
                else:
                    solution[i] = unbounded_sol[position]
                    position += 1
            report.add("\nИТОГ: Задача не имеет конечного оптимума (НР2).")
            return np.matrix(solution.reshape(1, -1))
    else:
        raise ValueError("Неизвестный формат возврата из симплекс-метода.")

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

    return np.matrix(solution.reshape(1, -1))


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