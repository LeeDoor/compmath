"""Приведение задачи к канонической форме."""

import numpy as np
import numpy.typing as npt

from ..problem import Sense, create_problem
from .simplex import ConstraintSign, VariableConstraint

CanonicalTable = tuple[np.ndarray, np.ndarray, np.ndarray, int]


def format_equation(row: npt.ArrayLike, var_names: list[str]) -> str:
    terms = []
    for coef, name in zip(row, var_names):
        if abs(coef) < 1e-9:
            continue

        if abs(coef - 1.0) < 1e-9:
            terms.append(name)
        elif abs(coef + 1.0) < 1e-9:
            terms.append(f"-{name}")
        else:
            terms.append(f"{coef:g}*{name}")

    if not terms:
        return "0"

    equation = terms[0]
    for term in terms[1:]:
        if term.startswith("-"):
            equation += f" - {term[1:]}"
        else:
            equation += f" + {term}"
    return equation


def _canonicalize(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense,
    constraint_signs: list[ConstraintSign] | None,
    variable_constraints: list[VariableConstraint] | None,
) -> tuple[CanonicalTable, list[str]]:
    problem = create_problem(A, b, c, sense)
    A_np = np.asarray(problem.A, dtype=float)
    b_np = np.asarray(problem.b, dtype=float).reshape(-1)
    c_np = np.asarray(problem.c, dtype=float).reshape(-1)

    m, n = A_np.shape
    signs = ["<="] * m if constraint_signs is None else list(constraint_signs)
    if len(signs) != m or any(sign not in ("<=", ">=", "=") for sign in signs):
        raise ValueError(
            "constraint_signs должны задавать знак '<=', '>=' или '=' "
            "для каждого ограничения."
        )

    if variable_constraints is None:
        variable_constraints = ["nonnegative"] * n
    if len(variable_constraints) != n or any(
        constraint not in ("nonnegative", "free")
        for constraint in variable_constraints
    ):
        raise ValueError(
            "variable_constraints должны задавать 'nonnegative' или 'free' "
            "для каждой переменной."
        )

    var_names = [f"x{i + 1}" for i in range(n)]
    output = ["Приведение к канонической форме:"]

    if problem.sense == "min":
        output.append(
            "1) Умножаем целевую функцию на -1, чтобы перейти к максимизации:"
        )
        c_np = -c_np
        output.append(f"   f(x) = {format_equation(c_np, var_names)} -> max")
    else:
        output.append(
            f"1) Задача уже на максимум: "
            f"f(x) = {format_equation(c_np, var_names)} -> max"
        )

    free_vars = [
        index
        for index, constraint in enumerate(variable_constraints)
        if constraint == "free"
    ]
    if free_vars:
        output.append("2) Замена свободных переменных (x = x' - x''):")
        for index in reversed(free_vars):
            A_np = np.insert(A_np, index + 1, -A_np[:, index], axis=1)
            c_np = np.insert(c_np, index + 1, -c_np[index])

            var_names[index] = f"x'{index + 1}"
            var_names.insert(index + 1, f"x''{index + 1}")
            output.append(f"   x{index + 1} -> x'{index + 1} - x''{index + 1}")
    else:
        output.append("2) Свободных переменных нет.")

    output.append("3) Преобразование ограничений в уравнения:")
    extra_columns = []
    extra_names = []
    variable_count = n

    for index, sign in enumerate(signs):
        row_str = format_equation(A_np[index], var_names)
        value = b_np[index]

        if sign == "<=":
            variable_count += 1
            column = np.zeros(m)
            column[index] = 1.0
            extra_columns.append(column)
            extra_names.append(f"x{variable_count}")
            output.append(
                f"   {row_str} <= {value:g}  =>  "
                f"{row_str} + x{variable_count} = {value:g}"
            )
        elif sign == ">=":
            variable_count += 1
            column = np.zeros(m)
            column[index] = -1.0
            extra_columns.append(column)
            extra_names.append(f"x{variable_count}")
            output.append(
                f"   {row_str} >= {value:g}  =>  "
                f"{row_str} - x{variable_count} = {value:g}"
            )
        else:
            output.append(f"   {row_str} = {value:g}   (уже уравнение)")

    decision_count = A_np.shape[1]
    if extra_columns:
        A_final = np.column_stack([A_np, *extra_columns])
        c_final = np.pad(c_np, (0, len(extra_columns)))
        var_names.extend(extra_names)
    else:
        A_final = A_np
        c_final = c_np

    output.append("4) Все переменные неотрицательны.")
    output.append("\nИТОГОВАЯ КАНОНИЧЕСКАЯ ФОРМА:")
    output.append(f"f(x) = {format_equation(c_final, var_names)} -> max")

    for index in range(m):
        output.append(
            f"  {format_equation(A_final[index], var_names)} = {b_np[index]:g}"
        )

    output.append(f"  {', '.join(var_names)} >= 0")

    table = (A_final, b_np, c_final, decision_count)
    return table, output


def make_canonical(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
    constraint_signs: list[ConstraintSign] | None = None,
    variable_constraints: list[VariableConstraint] | None = None,
) -> CanonicalTable:
    """Вернуть численные данные задачи в канонической форме."""
    table, _ = _canonicalize(
        A, b, c, sense, constraint_signs, variable_constraints
    )
    return table


def run_canonical_algorithm(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
    constraint_signs: list[ConstraintSign] | None = None,
    variable_constraints: list[VariableConstraint] | None = None,
) -> str:
    """Вернуть пошаговое текстовое описание приведения к канонической форме."""
    _, output = _canonicalize(
        A, b, c, sense, constraint_signs, variable_constraints
    )
    return "\n".join(output)
