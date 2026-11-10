"""Решение задачи двойственным симплекс-методом."""

import numpy as np
import numpy.typing as npt

from ..problem import Sense
from .canonical_form import make_canonical
from .simplex import ConstraintSign, VariableConstraint


_TOLERANCE = 1e-10
_MAX_ITERATIONS = 1000


def solve_dual_simplex(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
    constraint_signs: list[ConstraintSign] | None = None,
    variable_constraints: list[VariableConstraint] | None = None,
) -> str:
    """Solve an LP with dual simplex when its canonical basis allows it."""
    (
        canonical_matrix,
        canonical_rhs,
        objective,
        decision_count,
    ) = make_canonical(
        A,
        b,
        c,
        sense,
        constraint_signs,
        variable_constraints,
    )
    constraint_count, variable_count = canonical_matrix.shape
    original_objective = np.asarray(c, dtype=float).reshape(-1)

    variable_options = (
        ["nonnegative"] * len(original_objective)
        if variable_constraints is None
        else list(variable_constraints)
    )
    variable_mapping: list[tuple[int, int | None]] = []
    names = []
    transformed_index = 0
    for index, option in enumerate(variable_options):
        if option == "free":
            variable_mapping.append((transformed_index, transformed_index + 1))
            names.extend((f"x'{index + 1}", f"x''{index + 1}"))
            transformed_index += 2
        else:
            variable_mapping.append((transformed_index, None))
            names.append(f"x{index + 1}")
            transformed_index += 1
    names.extend(
        f"x{index}"
        for index in range(
            len(original_objective) + 1,
            len(original_objective) + variable_count - decision_count + 1,
        )
    )

    column_support = np.abs(canonical_matrix) > _TOLERANCE
    basis = []
    for row in range(constraint_count):
        candidates = np.flatnonzero(
            column_support[row] & (column_support.sum(axis=0) == 1)
        )
        if not len(candidates):
            raise ValueError("Нет базиса. Используйте МИБ.")
        supplemental = candidates[candidates >= decision_count]
        basis.append(int(supplemental[0] if len(supplemental) else candidates[0]))

    for row, basic_variable in enumerate(basis):
        coefficient = canonical_matrix[row, basic_variable]
        canonical_matrix[row, :] /= coefficient
        canonical_rhs[row] /= coefficient

    row_count = constraint_count

    def format_expression(coefficients: np.ndarray, variable_names: list[str]) -> str:
        terms = []
        for coefficient, name in zip(coefficients, variable_names):
            value = float(coefficient)
            if abs(value) < _TOLERANCE:
                continue
            magnitude = abs(value)
            term = name if abs(magnitude - 1) < _TOLERANCE else f"{magnitude:g}*{name}"
            if not terms:
                terms.append(f"-{term}" if value < 0 else term)
            else:
                terms.append(f" - {term}" if value < 0 else f" + {term}")
        return "".join(terms) or "0"

    print("\nСистема после приведения к канонической форме:")
    print(f"F = {format_expression(objective, names)} -> max")
    for row, value in zip(canonical_matrix, canonical_rhs):
        print(f"{format_expression(row, names)} = {value:g}")

    dual_candidates = np.flatnonzero(canonical_rhs < -_TOLERANCE)
    if not len(dual_candidates):
        return "Нет нужды в двойственном симплексе. Используйте обычный симплекс-метод."

    print(
        "После нормализации базисных переменных найдены отрицательные b; "
        "применяем двойственный симплекс-метод."
    )
    tableau = np.zeros((row_count + 1, variable_count + 1))
    tableau[:row_count, :variable_count] = canonical_matrix
    tableau[:row_count, -1] = canonical_rhs
    tableau[-1, :variable_count] = -objective
    for row, basic_variable in enumerate(basis):
        tableau[-1, :] += (
            objective[basic_variable] * tableau[row, :]
        )
    tableau[np.abs(tableau) < _TOLERANCE] = 0

    def print_table(title: str) -> None:
        nonbasic = [
            column
            for column in range(variable_count)
            if column not in basis
        ]
        print(f"\n{title}:")
        print(
            f"{'Базис':>8}"
            + "".join(f"{names[column]:>12}" for column in nonbasic)
            + f"{'b':>12}"
        )
        for row in range(row_count):
            print(
                f"{names[basis[row]]:>8}"
                + "".join(f"{tableau[row, column]:12.4g}" for column in nonbasic)
                + f"{tableau[row, -1]:12.4g}"
            )
        print(
            f"{'Δ':>8}"
            + "".join(f"{tableau[-1, column]:12.4g}" for column in nonbasic)
        )

    def pivot(row: int, column: int) -> None:
        element = tableau[row, column]
        tableau[row, :] /= element
        for other_row in range(row_count + 1):
            if other_row == row:
                continue
            factor = tableau[other_row, column]
            if abs(factor) > _TOLERANCE:
                tableau[other_row, :] -= factor * tableau[row, :]
        basis[row] = column
        tableau[np.abs(tableau) < _TOLERANCE] = 0

    for iteration in range(1, _MAX_ITERATIONS + 1):
        print_table(f"Симплекс-таблица, итерация {iteration}")
        negative_rhs = np.flatnonzero(tableau[:row_count, -1] < -_TOLERANCE)
        negative_delta = np.flatnonzero(tableau[-1, :-1] < -_TOLERANCE)
        if not len(negative_rhs) and not len(negative_delta):
            break

        if len(negative_rhs):
            leaving = int(negative_rhs[np.argmin(tableau[negative_rhs, -1])])
            candidates = [
                column
                for column in range(variable_count)
                if tableau[leaving, column] < -_TOLERANCE
            ]
            if not candidates:
                raise ValueError(
                    "Задача несовместна: в ведущей строке нет подходящего "
                    "ведущего столбца."
                )
            ratios = {
                column: tableau[-1, column] / tableau[leaving, column]
                for column in candidates
            }
            entering = max(candidates, key=ratios.__getitem__)
            print(
                f"Ведущая строка: {leaving + 1}; ведущий столбец: "
                f"{names[entering]}; Δ/a = {ratios[entering]:g}."
            )
        else:
            entering = int(negative_delta[np.argmin(tableau[-1, negative_delta])])
            candidates = [
                row
                for row in range(row_count)
                if tableau[row, entering] > _TOLERANCE
            ]
            if not candidates:
                raise ValueError("Целевая функция не ограничена на допустимой области.")
            leaving = min(
                candidates,
                key=lambda row: tableau[row, -1] / tableau[row, entering],
            )
            print(
                "Все b неотрицательны; завершаем оптимизацию обычным "
                f"симплекс-шагом. Входит {names[entering]}."
            )

        pivot(leaving, entering)
    else:
        raise ValueError("Превышено максимальное количество итераций.")

    transformed_solution = np.zeros(variable_count)
    for row, variable in enumerate(basis):
        if variable < variable_count:
            transformed_solution[variable] = tableau[row, -1]
    transformed_solution = transformed_solution[:decision_count]
    original_solution = np.asarray(
        [
            transformed_solution[first]
            if second is None
            else transformed_solution[first] - transformed_solution[second]
            for first, second in variable_mapping
        ]
    )
    objective_value = float(np.dot(original_objective, original_solution))
    return (
        "Решение исходной задачи: x* = ("
        + ", ".join(f"{value:g}" for value in original_solution)
        + f")\nЗначение целевой функции: {objective_value:g}"
    )
