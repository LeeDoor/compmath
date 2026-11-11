"""Решение задачи линейного программирования симплекс-методом."""

from itertools import combinations
from typing import Literal

import numpy as np
import numpy.typing as npt

from ..display_methods import format_math_function, format_system_of_equations
from ..problem import Sense, create_problem
from .steps import StepReport


ConstraintSign = Literal["<=", ">=", "="]
VariableConstraint = Literal["nonnegative", "free"]


def append_simplex_table(
    report: StepReport,
    tableau: np.ndarray,
    basis: list[int],
    variable_names: list[str],
    title: str,
    pivot_column: int | None = None,
    pivot_row: int | None = None,
) -> None:
    """Append a simplex tableau using the shared dual-simplex layout."""
    tableau[np.abs(tableau) < 1e-10] = 0.0
    variable_count = tableau.shape[1] - 1
    nonbasic = [
        column for column in range(variable_count) if column not in basis
    ]

    report.add_simplex_table_style(
        variable_names[pivot_column] if pivot_column is not None else None,
        variable_names[basis[pivot_row]] if pivot_row is not None else None,
    )
    report.add(f"\n{title}:")
    report.add(
        f"{'Базис':>8}"
        + "".join(f"{variable_names[column]:>12}" for column in nonbasic)
        + f"{'b':>12}"
    )
    for row in range(len(basis)):
        report.add(
            f"{variable_names[basis[row]]:>8}"
            + "".join(f"{tableau[row, column]:12.4g}" for column in nonbasic)
            + f"{tableau[row, -1]:12.4g}"
        )
    report.add(
        f"{'Δ':>8}"
        + "".join(f"{tableau[-1, column]:12.4g}" for column in nonbasic)
    )


def _run_simplex_tableau_impl(
    tableau: np.ndarray,
    basis: list[int],
    costs: np.ndarray,
    report: StepReport,
    variable_names: list[str] | None = None,
    title: str = "Симплекс-метод",
    eps: float = 1e-10,
    max_iterations: int = 1000,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Выполнить симплекс-итерации на уже подготовленной таблице."""
    tableau = tableau.copy()
    basis = basis.copy()
    costs = np.asarray(costs, dtype=float).reshape(-1)
    variable_count = len(costs)
    row_count = tableau.shape[0] - 1

    if tableau.shape[1] != variable_count + 1:
        raise ValueError("Размер таблицы не соответствует числу коэффициентов целевой функции.")
    if len(basis) != row_count:
        raise ValueError("Количество базисных переменных не соответствует числу ограничений.")
    if any(index < 0 or index >= variable_count for index in basis):
        raise ValueError("В базисе указана несуществующая переменная.")

    if variable_names is None:
        variable_names = [f"x{i + 1}" for i in range(variable_count)]

    non_basis = [j for j in range(variable_count) if j not in basis]

    def append_table(
        number: int,
        pivot_column: int | None = None,
        pivot_row: int | None = None,
    ) -> None:
        append_simplex_table(
            report,
            tableau,
            basis,
            variable_names,
            f"{title}. СИМПЛЕКСНАЯ ТАБЛИЦА №{number}",
            pivot_column,
            pivot_row,
        )

        report.add("\nОценки Δ всех переменных:")
        deltas = []
        for j in range(variable_count):
            delta = 0.0 if j in basis else tableau[-1, j]
            if abs(delta) < eps:
                delta = 0.0
            deltas.append(f"{variable_names[j]}: {delta:.4f}")
        report.add("    ".join(deltas))

    for iteration in range(1, max_iterations + 1):
        tableau[np.abs(tableau) < eps] = 0.0

        # Вычисление оценок Δ = z_j - c_j
        basic_costs = np.array([costs[index] for index in basis])
        reduced = basic_costs @ tableau[:row_count, :-1] - costs

        objective_value = float(basic_costs @ tableau[:row_count, -1])
        tableau[-1, :] = 0.0
        tableau[-1, :-1] = reduced
        tableau[-1, -1] = objective_value
        tableau[np.abs(tableau) < eps] = 0.0

        report.add(f"\n\n{title.upper()}. ИТЕРАЦИЯ №{iteration}")
        report.add(
            "Базисные переменные: "
            + ", ".join(variable_names[index] for index in basis)
        )

        candidates = [j for j in non_basis if tableau[-1, j] < -eps]
        if not candidates:
            append_table(iteration)
            report.add("\nВсе оценки Δ неотрицательны.")
            report.add("Получено оптимальное решение.")
            return tableau, basis, non_basis

        entering = min(candidates, key=lambda j: tableau[-1, j])

        ratios = []
        ratio_steps = []
        for i in range(row_count):
            coefficient = tableau[i, entering]
            if coefficient > eps:
                ratio = tableau[i, -1] / coefficient
                ratios.append((ratio, i))
                ratio_steps.append(
                    f"Строка {variable_names[basis[i]]}: "
                    f"{tableau[i, -1]:.4f} / {coefficient:.4f} = {ratio:.4f}"
                )
            else:
                ratio_steps.append(
                    f"Строка {variable_names[basis[i]]}: "
                    "не подходит (коэффициент <= 0)"
                )

        if not ratios:
            append_table(iteration, pivot_column=entering)
            report.add(f"\nРазрешающий столбец: {variable_names[entering]}")
            report.add(f"Оценка Δ = {tableau[-1, entering]:.4f}")
            report.add("\nВычисление отношений b_i / a_ij:")
            for step in ratio_steps:
                report.add(step)
            raise ValueError("Целевая функция не ограничена на допустимой области.")

        _, leaving = min(ratios)
        pivot = tableau[leaving, entering]
        leaving_variable = basis[leaving]

        append_table(iteration, pivot_column=entering, pivot_row=leaving)
        report.add(f"\nРазрешающий столбец: {variable_names[entering]}")
        report.add(f"Оценка Δ = {tableau[-1, entering]:.4f}")
        report.add("\nВычисление отношений b_i / a_ij:")
        for step in ratio_steps:
            report.add(step)
        report.add(
            f"\nРазрешающая строка: {variable_names[leaving_variable]}"
        )
        report.add(f"Разрешающий элемент: {pivot:.4f}")

        tableau[leaving, :] /= pivot
        for i in range(row_count + 1):
            if i == leaving:
                continue
            factor = tableau[i, entering]
            tableau[i, :] -= factor * tableau[leaving, :]

        entering_position = non_basis.index(entering)
        basis[leaving] = entering
        non_basis[entering_position] = leaving_variable

        tableau[np.abs(tableau) < eps] = 0
    raise ValueError("Превышено максимальное количество итераций.")


def _solve_simplex_impl(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
    constraint_signs: list[ConstraintSign] | None = None,
    variable_constraints: list[VariableConstraint] | None = None,
    *,
    report: StepReport,
    return_details: bool = False,
) -> np.matrix | tuple[np.matrix, list[int], np.ndarray, np.ndarray]:
    """Решить задачу линейного программирования обычным симплекс-методом."""
    eps = 1e-10

    from .canonical_form import format_equation, make_canonical

    problem = create_problem(A, b, c, sense)
    original_c = np.asarray(problem.c, dtype=float).reshape(-1)
    original_variable_count = len(original_c)
    canonical_matrix, canonical_rhs, canonical_costs, decision_count = (
        make_canonical(
            problem.A,
            problem.b,
            problem.c,
            sense,
            constraint_signs,
            variable_constraints,
        )
    )
    canonical_matrix = canonical_matrix.copy()
    canonical_rhs = canonical_rhs.copy()
    canonical_costs = canonical_costs.copy()
    row_count, variable_count = canonical_matrix.shape

    variable_options = (
        ["nonnegative"] * original_variable_count
        if variable_constraints is None
        else list(variable_constraints)
    )
    variable_names = []
    transformed_index = 0
    for index, option in enumerate(variable_options):
        if option == "free":
            variable_names.extend((f"x'{index + 1}", f"x''{index + 1}"))
            transformed_index += 2
        else:
            variable_names.append(f"x{index + 1}")
            transformed_index += 1
    variable_names.extend(
        f"x{index}"
        for index in range(
            original_variable_count + 1,
            original_variable_count + variable_count - decision_count + 1,
        )
    )

    report.add("\nПРИВЕДЕНИЕ К КАНОНИЧЕСКОЙ ФОРМЕ")
    canonical_function = (
        f"F = {format_equation(canonical_costs, variable_names)} -> max"
    )
    report.add_formatted_function(
        canonical_function,
        format_math_function(canonical_function),
    )
    canonical_equations = [
        f"{format_equation(row, variable_names)} = {value:g}"
        for row, value in zip(canonical_matrix, canonical_rhs)
    ]
    report.add_formatted_system(
        canonical_equations,
        format_system_of_equations(canonical_equations),
        heading="Система после приведения к канонической форме:",
    )

    negative_rhs_rows = np.flatnonzero(canonical_rhs < -eps)
    if len(negative_rhs_rows):
        report.add("\nПРИВЕДЕНИЕ ПРАВЫХ ЧАСТЕЙ К НЕОТРИЦАТЕЛЬНОМУ ВИДУ")
        for row in negative_rhs_rows:
            canonical_matrix[row, :] *= -1
            canonical_rhs[row] *= -1
            report.add(f"Уравнение {row + 1} умножено на -1.")
    else:
        report.add("\nВсе элементы вектора b неотрицательны.")

    canonical_rhs[np.abs(canonical_rhs) < eps] = 0
    canonical_matrix[np.abs(canonical_matrix) < eps] = 0
    normalized_equations = [
        f"{format_equation(row, variable_names)} = {value:g}"
        for row, value in zip(canonical_matrix, canonical_rhs)
    ]
    if len(negative_rhs_rows):
        report.add_formatted_system(
            normalized_equations,
            format_system_of_equations(normalized_equations),
            heading="Система после нормализации b:",
        )

    column_support = np.abs(canonical_matrix) > eps
    basis = []
    for row in range(row_count):
        candidates = [
            column
            for column in range(variable_count)
            if column_support[row, column]
            and column_support[:, column].sum() == 1
            and canonical_matrix[row, column] > eps
        ]
        candidates.sort(key=lambda column: column < decision_count)
        if not candidates:
            raise ValueError(
                "Нет допустимого начального базиса. Используйте МИБ."
            )
        basis.append(candidates[0])

    for row, basic_variable in enumerate(basis):
        pivot = canonical_matrix[row, basic_variable]
        canonical_matrix[row, :] /= pivot
        canonical_rhs[row] /= pivot

    initial_matrix = canonical_matrix.copy()
    tableau = np.zeros((row_count + 1, variable_count + 1))
    tableau[:row_count, :variable_count] = canonical_matrix
    tableau[:row_count, -1] = canonical_rhs

    report.add("\nИСХОДНАЯ ЗАДАЧА")
    report.add("\nЦелевая функция:")
    original_function = (
        "F = "
        + " + ".join(
            f"({original_c[j]:g})*x{j + 1}"
            for j in range(original_variable_count)
        )
        + f" -> {sense}"
    )
    report.add_formatted_function(
        original_function,
        format_math_function(original_function),
    )

    tableau, basis, non_basis = _run_simplex_tableau_impl(
        tableau,
        basis,
        canonical_costs,
        report,
        variable_names,
        title="Симплекс-метод",
        eps=eps,
    )

    solution = np.zeros(variable_count)
    for i, variable in enumerate(basis):
        solution[variable] = tableau[i, -1]
    solution[np.abs(solution) < eps] = 0.0

    variable_mapping = []
    transformed_index = 0
    for option in variable_options:
        if option == "free":
            variable_mapping.append((transformed_index, transformed_index + 1))
            transformed_index += 2
        else:
            variable_mapping.append((transformed_index, None))
            transformed_index += 1
    original_solution = np.asarray(
        [
            solution[first]
            if second is None
            else solution[first] - solution[second]
            for first, second in variable_mapping
        ]
    )
    objective_value = float(np.dot(original_c, original_solution))

    report.add("\n\nИТОГОВЫЙ РЕЗУЛЬТАТ")
    report.add("-" * 40)
    report.add("x* = (" + ", ".join(f"{value:g}" for value in solution) + ")")
    report.add(f"F{'max' if sense == 'max' else 'min'} = {objective_value:g}")

    alternate_solution = None
    zero_delta_found = False

    for j in range(variable_count):
        if j in basis or abs(tableau[-1, j]) >= eps:
            continue

        zero_delta_found = True
        ratios = [
            (tableau[i, -1] / tableau[i, j], i)
            for i in range(row_count)
            if tableau[i, j] > eps
        ]

        if ratios:
            _, leaving = min(ratios)
            alt_tableau = tableau.copy()
            alt_basis = basis.copy()
            pivot = alt_tableau[leaving, j]
            alt_tableau[leaving, :] /= pivot

            for i in range(row_count + 1):
                if i != leaving:
                    factor = alt_tableau[i, j]
                    alt_tableau[i, :] -= factor * alt_tableau[leaving, :]

            alt_basis[leaving] = j
            alt_tableau[np.abs(alt_tableau) < eps] = 0
            candidate = np.zeros(variable_count)
            for i, variable in enumerate(alt_basis):
                candidate[variable] = alt_tableau[i, -1]
        else:
            candidate = solution.copy()
            candidate[j] = 1
            for i, variable in enumerate(basis):
                candidate[variable] -= tableau[i, j]

        if np.all(candidate >= -eps):
            candidate_original = np.asarray(
                [
                    candidate[first]
                    if second is None
                    else candidate[first] - candidate[second]
                    for first, second in variable_mapping
                ]
            )
            if not np.allclose(candidate_original, original_solution, atol=eps):
                alternate_solution = candidate
                break

    if alternate_solution is not None:
        alternate_original = np.asarray(
            [
                alternate_solution[first]
                if second is None
                else alternate_solution[first] - alternate_solution[second]
                for first, second in variable_mapping
            ]
        )
        alternate_value = float(np.dot(original_c, alternate_original))
        report.add("\nНайдено альтернативное оптимальное решение:")
        report.add("x** = (" + ", ".join(f"{value:g}" for value in alternate_solution) + ")")
        report.add(f"F = {alternate_value:g}")
    elif zero_delta_found:
        report.add("\nОбнаружена нулевая оценка Δ, но другой допустимый вектор не найден.")
    else:
        report.add("\nАльтернативных оптимальных решений по нулевым оценкам Δ не обнаружено.")

    solution_matrix = np.matrix(original_solution.reshape(1, -1))
    if return_details:
        return (
            solution_matrix,
            basis,
            tableau[-1, :variable_count].copy(),
            initial_matrix,
        )
    return np.matrix(solution.reshape(1, -1))


def run_simplex_tableau(
    tableau: np.ndarray,
    basis: list[int],
    costs: np.ndarray,
    variable_names: list[str] | None = None,
    title: str = "Симплекс-метод",
    eps: float = 1e-10,
    max_iterations: int = 1000,
) -> tuple[tuple[np.ndarray, list[int], list[int]], str]:
    report = StepReport()
    result = _run_simplex_tableau_impl(
        tableau,
        basis,
        costs,
        report,
        variable_names,
        title,
        eps,
        max_iterations,
    )
    return result, report.get_text()


def solve_simplex(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
    constraint_signs: list[ConstraintSign] | None = None,
    variable_constraints: list[VariableConstraint] | None = None,
    *,
    return_details: bool = False,
) -> tuple[
    np.matrix | tuple[np.matrix, list[int], np.ndarray, np.ndarray], str
]:
    report = StepReport()
    result = _solve_simplex_impl(
        A,
        b,
        c,
        sense,
        constraint_signs,
        variable_constraints,
        report=report,
        return_details=return_details,
    )
    return result, report.get_text()
