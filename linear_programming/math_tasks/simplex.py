"""Решение задачи линейного программирования симплекс-методом."""

from itertools import combinations
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
    tableau[np.abs(tableau) < 1e-10] = 0.0
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
    # Выводим оценки для всех внебазисных столбцов И для столбца b (значение F)
    report.add(
        f"{'Δ':>8}"
        + "".join(f"{tableau[-1, column]:12.4g}" for column in nonbasic)
        + f"{tableau[-1, -1]:12.4g}"
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
            if abs(delta) < eps:
                delta = 0.0
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

            b_val = 0.0 if abs(tableau[i, -1]) < eps else tableau[i, -1]
            equation = f"{variable_names[basis[i]]} = {b_val:.4f}"
            if terms:
                equation += " - " + " - ".join(terms)
            equations.append(equation)
        report.add_formatted_system(
            equations,
            format_system_of_equations(equations),
            heading="Система уравнений текущего базиса:",
        )

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
            # Формируем текущее допустимое решение
            current_solution = np.zeros(variable_count)
            for i, var_idx in enumerate(basis):
                current_solution[var_idx] = tableau[i, -1]

            report.add(
                f"\nВ разрешающем столбце {variable_names[entering]} все коэффициенты "
                f"a_ij <= 0."
            )
            report.add(
                "Это означает, что при увеличении этой переменной целевая функция "
                "неограниченно растёт, а ограничения её не сдерживают."
            )
            report.add(
                f"Направление ухода в бесконечность: увеличиваем {variable_names[entering]}."
            )
            report.add(
                "Текущий базис: "
                + ", ".join(variable_names[i] for i in basis)
            )
            report.add(f"Текущее (допустимое) значение F = {objective_value:.6g}")
            report.add(
                "ВЫВОД: НР2"
            )

            # Возвращаем текущее решение, чтобы интерфейс мог его отобразить,
            # и сигнализируем об неограниченности через специальный флаг в конце.
            # (Мы не выбрасываем ошибку, чтобы можно было вернуть решение).
            return tableau, basis, non_basis, current_solution, objective_value, True

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

        tableau[np.abs(tableau) < eps] = 0.0
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
    A_mat = np.asarray(problem.A, dtype=float)
    b_vec = np.asarray(problem.b, dtype=float).reshape(-1)
    original_c = np.asarray(problem.c, dtype=float).reshape(-1)
    m, n = A_mat.shape

    if constraint_signs is None:
        constraint_signs = ["<="] * m
    elif len(constraint_signs) != m or any(
        sign not in ("<=", ">=", "=") for sign in constraint_signs
    ):
        raise ValueError(
            "constraint_signs должны задавать знак '<=', '>=' или '=' "
            "для каждого ограничения."
        )

    if any(sign == ">=" for sign in constraint_signs):
        raise ValueError("Ограничения со знаком '>=' этим вариантом симплекс-метода не поддерживаются.")

    free_vars = []
    if variable_constraints is not None:
        free_vars = [i for i, vc in enumerate(variable_constraints) if vc == "free"]

    if free_vars:
        report.add("\nОБРАБОТКА СВОБОДНЫХ ПЕРЕМЕННЫХ:")
        A_list = []
        c_list = []
        names_list = []
        for i in range(n):
            if i in free_vars:
                report.add(f"Переменная x{i + 1} свободна => замена: x{i + 1} = x'{i + 1} - x''{i + 1}")
                A_list.append(A_mat[:, i:i+1])
                A_list.append(-A_mat[:, i:i+1])
                c_list.extend([original_c[i], -original_c[i]])
                names_list.extend([f"x'{i+1}", f"x''{i+1}"])
            else:
                A_list.append(A_mat[:, i:i+1])
                c_list.append(original_c[i])
                names_list.append(f"x{i+1}")

        A_work = np.hstack(A_list)
        c_work = np.array(c_list)
    else:
        A_work = A_mat.copy()
        c_work = original_c.copy()
        names_list = [f"x{i + 1}" for i in range(n)]

    n_work = A_work.shape[1]

    b_vec[np.abs(b_vec) < eps] = 0.0
    work_c = c_work.copy()
    if sense == "min":
        work_c = -work_c

    if "=" in constraint_signs:
        slack_rows = [i for i, sign in enumerate(constraint_signs) if sign == "<="]
        slack_matrix = np.zeros((m, len(slack_rows)))
        for column, row in enumerate(slack_rows):
            slack_matrix[row, column] = 1.0

        A_augmented = np.column_stack((A_work, slack_matrix))
        total_variable_count = A_augmented.shape[1]
        full_costs = np.concatenate((work_c, np.zeros(len(slack_rows))))
        variable_names = names_list + [
            f"x{n_work + row + 1}" for row in slack_rows
        ]

        basis = None
        initial_values = None
        for candidate in combinations(range(total_variable_count), m):
            basis_matrix = A_augmented[:, candidate]
            try:
                values = np.linalg.solve(basis_matrix, b_vec)
            except np.linalg.LinAlgError:
                continue
            if np.all(values >= -eps):
                basis = list(candidate)
                initial_values = values
                break

        if basis is None:
            raise ValueError(
                "Не удалось подобрать допустимый начальный базис для ограничений '='. "
                "Без метода искусственного базиса задача может быть неразрешима этим способом."
            )

        basis_matrix = A_augmented[:, basis]
        canonical_A = np.linalg.solve(basis_matrix, A_augmented)
        canonical_b = np.linalg.solve(basis_matrix, b_vec)
        tableau = np.zeros((m + 1, total_variable_count + 1))
        tableau[:m, :-1] = canonical_A
        tableau[:m, -1] = canonical_b
    else:
        if np.any(b_vec < -eps):
            raise ValueError(
                "Для данного варианта симплекс-метода все b должны быть >= 0. "
                "Рекомендуется использовать метод двойственного симплекса."
            )

        A_augmented = np.column_stack((A_work, np.eye(m)))
        total_variable_count = n_work + m
        full_costs = np.zeros(total_variable_count)
        full_costs[:n_work] = work_c
        tableau = np.zeros((m + 1, total_variable_count + 1))
        tableau[:m, :n_work] = A_work
        tableau[:m, n_work:n_work + m] = np.eye(m)
        tableau[:m, -1] = b_vec
        basis = list(range(n_work, n_work + m))
        variable_names = names_list + [f"x{n_work + i + 1}" for i in range(m)]

    report.add("\nИСХОДНАЯ ЗАДАЧА")
    report.add("\nЦелевая функция:")
    report.add(
        "F = "
        + " + ".join(f"({original_c[j]:g})*x{j + 1}" for j in range(n))
        + f" -> {sense}"
    )

    tableau, basis, non_basis = _run_simplex_tableau_impl(
        tableau,
        basis,
        full_costs,
        report,
        variable_names,
        title="Симплекс-метод",
        eps=eps,
    )

    extended_solution = np.zeros(total_variable_count)
    for i, variable in enumerate(basis):
        extended_solution[variable] = tableau[i, -1]

    extended_solution[np.abs(extended_solution) < eps] = 0.0

    solution = np.zeros(n)
    idx = 0
    for i in range(n):
        if i in free_vars:
            solution[i] = extended_solution[idx] - extended_solution[idx + 1]
            idx += 2
        else:
            solution[i] = extended_solution[idx]
            idx += 1

    solution[np.abs(solution) < eps] = 0.0
    objective_value = float(np.dot(original_c, solution))
    if abs(objective_value) < eps:
        objective_value = 0.0

    report.add("\n\nИТОГОВЫЙ РЕЗУЛЬТАТ")
    report.add("-" * 40)
    report.add("x* = (" + ", ".join(f"{value:g}" for value in solution) + ")")
    report.add(f"F{'max' if sense == 'max' else 'min'} = {objective_value:g}")

    # Поиск столбцов с 0 в оценках (по конспекту)
    zero_delta_vars = [
        j for j in non_basis if abs(tableau[-1, j]) < eps
    ]

    if zero_delta_vars:
        report.add("\nТ.к. существует столбец с '0' в Δ, существует множество решений.")
        report.add("Выбрав разрешающим столбец, у которого '0' в Δ, получим другое опорное решение:")

        entering_zero = zero_delta_vars[0]
        ratios = []
        for i in range(m):
            coeff = tableau[i, entering_zero]
            if coeff > eps:
                ratios.append((tableau[i, -1] / coeff, i))

        if ratios:
            _, leaving_zero = min(ratios)
            alt_tableau = tableau.copy()
            alt_basis = basis.copy()
            pivot = alt_tableau[leaving_zero, entering_zero]

            alt_tableau[leaving_zero, :] /= pivot
            for i in range(m + 1):
                if i != leaving_zero:
                    factor = alt_tableau[i, entering_zero]
                    alt_tableau[i, :] -= factor * alt_tableau[leaving_zero, :]

            alt_basis[leaving_zero] = entering_zero
            alt_tableau[np.abs(alt_tableau) < eps] = 0.0

            alt_extended = np.zeros(total_variable_count)
            for i, var_idx in enumerate(alt_basis):
                alt_extended[var_idx] = alt_tableau[i, -1]

            alt_solution = np.zeros(n)
            alt_idx = 0
            for i in range(n):
                if i in free_vars:
                    alt_solution[i] = alt_extended[alt_idx] - alt_extended[alt_idx + 1]
                    alt_idx += 2
                else:
                    alt_solution[i] = alt_extended[alt_idx]
                    alt_idx += 1
            alt_solution[np.abs(alt_solution) < eps] = 0.0

            report.add(f"\nВторое базовое решение x~ = ({', '.join(f'{val:g}' for val in alt_solution)})")
            report.add("Тогда общее решение задаётся:")
            report.add("x = α * x* + (1 - α) * x~, где α ∈ [0; 1]")

    solution_matrix = np.matrix(solution.reshape(1, -1))
    if return_details:
        augmented_A = A_augmented
        return (
            solution_matrix,
            basis,
            tableau[-1, :total_variable_count].copy(),
            augmented_A,
        )
    return solution_matrix


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
