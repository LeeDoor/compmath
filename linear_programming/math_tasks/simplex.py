"""Решение задачи линейного программирования симплекс-методом."""

from typing import Literal

import numpy as np
import numpy.typing as npt

from ..display_methods import format_system_of_equations
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
) -> None:
    """Append a simplex tableau using the shared dual-simplex layout."""
    variable_count = tableau.shape[1] - 1
    nonbasic = [
        column for column in range(variable_count) if column not in basis
    ]

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
    """Выполнить симплекс-итерации на уже подготовленной таблице.

    Функция НЕ создаёт начальную таблицу и НЕ формирует базис.
    Она получает готовые tableau и basis, поэтому подходит для обеих
    фаз метода искусственного базиса.
    """
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

    def append_table(number: int) -> None:
        append_simplex_table(
            report,
            tableau,
            basis,
            variable_names,
            f"{title}. СИМПЛЕКСНАЯ ТАБЛИЦА №{number}",
        )

        report.add("\nОценки Δ всех переменных:")
        deltas = []
        for j in range(variable_count):
            delta = 0.0 if j in basis else tableau[-1, j]
            deltas.append(f"{variable_names[j]}: {delta:.4f}")
        report.add("    ".join(deltas))

    def append_equations() -> None:
        equations = []
        for i in range(row_count):
            terms = []
            for j in range(variable_count):
                if j == basis[i]:
                    continue
                coefficient = tableau[i, j]
                if abs(coefficient) < eps:
                    continue
                terms.append(f"({coefficient:.4f})*{variable_names[j]}")

            equation = f"{variable_names[basis[i]]} = {tableau[i, -1]:.4f}"
            if terms:
                equation += " - " + " - ".join(terms)
            equations.append(equation)
        report.add_formatted_system(
            equations,
            format_system_of_equations(equations),
            heading="Система уравнений текущего базиса:",
        )

    for iteration in range(1, max_iterations + 1):
        basic_costs = np.array([costs[index] for index in basis])
        reduced = basic_costs @ tableau[:row_count, :-1] - costs

        objective_value = float(basic_costs @ tableau[:row_count, -1])
        tableau[-1, :] = 0
        tableau[-1, :-1] = reduced
        tableau[-1, -1] = objective_value

        report.add(f"\n\n{title.upper()}. ИТЕРАЦИЯ №{iteration}")
        append_table(iteration)
        append_equations()

        candidates = [j for j in non_basis if tableau[-1, j] < -eps]
        if not candidates:
            report.add("\nВсе оценки Δ неотрицательны.")
            report.add("Получено оптимальное решение.")
            return tableau, basis, non_basis

        entering = min(candidates, key=lambda j: tableau[-1, j])
        report.add(f"\nРазрешающий столбец: {variable_names[entering]}")
        report.add(f"Оценка Δ = {tableau[-1, entering]:.4f}")

        ratios = []
        report.add("\nВычисление отношений b_i / a_ij:")
        for i in range(row_count):
            coefficient = tableau[i, entering]
            if coefficient > eps:
                ratio = tableau[i, -1] / coefficient
                ratios.append((ratio, i))
                report.add(
                    f"Строка {variable_names[basis[i]]}: "
                    f"{tableau[i, -1]:.4f} / {coefficient:.4f} = {ratio:.4f}"
                )
            else:
                report.add(
                    f"Строка {variable_names[basis[i]]}: "
                    "не подходит (коэффициент <= 0)"
                )

        if not ratios:
            raise ValueError("Целевая функция не ограничена на допустимой области.")

        _, leaving = min(ratios)
        pivot = tableau[leaving, entering]
        leaving_variable = basis[leaving]

        report.add(
            f"\nРазрешающая строка: {variable_names[leaving_variable]}"
        )
        report.add(f"Разрешающий элемент: {pivot:.4f}")
        report.add(
            f"\nИз базиса выходит {variable_names[leaving_variable]}, "
            f"в базис входит {variable_names[entering]}."
        )

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
        report.add("\nТаблица после пересчёта:")
        append_table(iteration + 1)
        append_equations()

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

    problem = create_problem(A, b, c, sense)
    A = np.asarray(problem.A, dtype=float)
    b = np.asarray(problem.b, dtype=float).reshape(-1)
    original_c = np.asarray(problem.c, dtype=float).reshape(-1)
    m, n = A.shape

    if np.any(b < -eps):
        raise ValueError(
            "Для данного варианта симплекс-метода все b должны быть >= 0."
        )

    b[np.abs(b) < eps] = 0
    work_c = original_c.copy()
    if sense == "min":
        work_c = -work_c

    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:m, :n] = A
    tableau[:m, n:n + m] = np.eye(m)
    tableau[:m, -1] = b
    basis = list(range(n, n + m))
    variable_names = [f"x{i + 1}" for i in range(n + m)]

    report.add("\nИСХОДНАЯ ЗАДАЧА")
    report.add("\nЦелевая функция:")
    report.add(
        "F = "
        + " + ".join(f"({original_c[j]:g})*x{j + 1}" for j in range(n))
        + f" -> {sense}"
    )
    constraints = []
    for i in range(m):
        terms = " + ".join(f"({A[i, j]:g})*x{j + 1}" for j in range(n))
        constraints.append(f"{terms} <= {b[i]:g}")
    report.add_formatted_system(
        constraints,
        format_system_of_equations(constraints),
        heading="Ограничения:",
    )

    augmented_constraints = []
    for i in range(m):
        terms = " + ".join(f"({A[i, j]:g})*x{j + 1}" for j in range(n))
        augmented_constraints.append(f"{terms} + x{n + i + 1} = {b[i]:g}")
    report.add_formatted_system(
        augmented_constraints,
        format_system_of_equations(augmented_constraints),
        heading="Добавляем базисные переменные:",
    )

    full_costs = np.zeros(n + m)
    full_costs[:n] = work_c

    tableau, basis, non_basis = _run_simplex_tableau_impl(
        tableau,
        basis,
        full_costs,
        report,
        variable_names,
        title="Симплекс-метод",
        eps=eps,
    )

    solution = np.zeros(n + m)
    for i, variable in enumerate(basis):
        solution[variable] = tableau[i, -1]

    objective_value = float(np.dot(original_c, solution[:n]))

    report.add("\n\nИТОГОВЫЙ РЕЗУЛЬТАТ")
    report.add("-" * 40)
    report.add("x* = (" + ", ".join(f"{value:g}" for value in solution) + ")")
    report.add(f"F{'max' if sense == 'max' else 'min'} = {objective_value:g}")

    alternate_solution = None
    zero_delta_found = False

    for j in range(n + m):
        if j in basis or abs(tableau[-1, j]) >= eps:
            continue

        zero_delta_found = True
        ratios = [
            (tableau[i, -1] / tableau[i, j], i)
            for i in range(m)
            if tableau[i, j] > eps
        ]

        if ratios:
            _, leaving = min(ratios)
            alt_tableau = tableau.copy()
            alt_basis = basis.copy()
            pivot = alt_tableau[leaving, j]
            alt_tableau[leaving, :] /= pivot

            for i in range(m + 1):
                if i != leaving:
                    factor = alt_tableau[i, j]
                    alt_tableau[i, :] -= factor * alt_tableau[leaving, :]

            alt_basis[leaving] = j
            alt_tableau[np.abs(alt_tableau) < eps] = 0
            candidate = np.zeros(n + m)
            for i, variable in enumerate(alt_basis):
                candidate[variable] = alt_tableau[i, -1]
        else:
            candidate = solution.copy()
            candidate[j] = 1
            for i, variable in enumerate(basis):
                candidate[variable] -= tableau[i, j]

        if np.all(candidate >= -eps) and not np.allclose(candidate, solution, atol=eps):
            alternate_solution = candidate
            break

    if alternate_solution is not None:
        alternate_value = float(np.dot(original_c, alternate_solution[:n]))
        report.add("\nНайдено альтернативное оптимальное решение:")
        report.add("x** = (" + ", ".join(f"{value:g}" for value in alternate_solution) + ")")
        report.add(f"F = {alternate_value:g}")
    elif zero_delta_found:
        report.add("\nОбнаружена нулевая оценка Δ, но другой допустимый вектор не найден.")
    else:
        report.add("\nАльтернативных оптимальных решений по нулевым оценкам Δ не обнаружено.")

    solution_matrix = np.matrix(solution[:n].reshape(1, -1))
    if return_details:
        augmented_A = np.column_stack([A, np.eye(m)])
        return (
            solution_matrix,
            basis,
            tableau[-1, :n + m].copy(),
            augmented_A,
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
