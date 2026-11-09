"""Приведение задачи к канонической форме."""

import numpy.typing as npt
import numpy as np
from ..problem import Sense, create_problem
from ..display_methods.display_function import display_function
from .simplex import ConstraintSign, VariableConstraint


def to_canonical_form(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
    constraint_signs: list[ConstraintSign] | None = None,
    variable_constraints: list[VariableConstraint] | None = None,
) -> str:
    problem = create_problem(A, b, c, sense)
    A_np = np.asarray(problem.A, dtype=float)
    b_np = np.asarray(problem.b, dtype=float).reshape(-1)
    c_np = np.asarray(problem.c, dtype=float).reshape(-1)

    m, n = A_np.shape
    signs = (
        ["<="] * m if constraint_signs is None else list(constraint_signs)
    )
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
    free_vars = [
        index
        for index, constraint in enumerate(variable_constraints)
        if constraint == "free"
    ]

    result = ["Приведение к канонической форме:"]
    if problem.sense == "min":
        c_np = -c_np
        result.append("1) Умножаем f(x) на -1 для перехода к max.")
    else:
        result.append("1) Задача уже на максимум.")

    if free_vars:
        result.append("2) Замена свободных переменных (x = x' - x''):")
        for j in reversed(sorted(free_vars)):
            A_np = np.insert(A_np, j + 1, -A_np[:, j], axis=1)
            c_np = np.insert(c_np, j + 1, -c_np[j])
            result.append(f"   x{j+1} -> x'{j+1} - x''{j+1}")
        n = A_np.shape[1]
    else:
        result.append("2) Свободных переменных нет.")

    result.append("3) Преобразование ограничений в уравнения:")

    new_columns = [] 
    var_count = n    
    for i in range(m):
        row = A_np[i]
        val = b_np[i]
        sign = signs[i]
        
        row_str = " + ".join(f"{row[j]:g}*x{j+1}" for j in range(n)) or "0"
        
        if sign == "<=":
            var_count += 1
            col = np.zeros(m)
            col[i] = 1.0
            new_columns.append(col)
            result.append(f"   {row_str} <= {val:g}  =>  {row_str} + x{var_count} = {val:g}")

        elif sign == ">=":
            var_count += 1
            col = np.zeros(m)
            col[i] = -1.0
            new_columns.append(col)
            result.append(f"   {row_str} >= {val:g}  =>  {row_str} - x{var_count} = {val:g}")
        elif sign == "=":

            result.append(f"   {row_str} = {val:g}")

    A_final = (
        np.hstack([A_np, np.column_stack(new_columns)])
        if new_columns
        else A_np
    )
    c_final = np.pad(c_np, (0, len(new_columns)))
    result.append("\nИТОГОВАЯ КАНОНИЧЕСКАЯ ФОРМА:")
    result.append(f"f(x) = {display_function(c_final)} -> max")

    for i in range(m):
        terms = []
        for j in range(A_final.shape[1]):
            coef = A_final[i, j]
            if abs(coef) < 1e-9:
                continue
            if abs(coef - 1.0) < 1e-9:
                terms.append(f"x{j+1}")
            elif abs(coef + 1.0) < 1e-9:
                terms.append(f"- x{j+1}")
            else:
                terms.append(f"{coef:g}*x{j+1}")

        eq_str = terms[0] if terms else "0"
        for term in terms[1:]:
            if term.startswith("-"):
                eq_str += f" - {term[1:].strip()}"
            else:
                eq_str += f" + {term}"

        result.append(f"  {eq_str} = {b_np[i]:g}")

    result.append(f"  x1, ..., x{A_final.shape[1]} >= 0")

    return "\n".join(result)

def get_canonical_matrices(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
    constraint_signs: list[ConstraintSign] | None = None,
    variable_constraints: list[VariableConstraint] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Получить матрицы задачи в канонической форме."""

    problem = create_problem(A, b, c, sense)

    A_np = np.asarray(problem.A, dtype=float)
    b_np = np.asarray(problem.b, dtype=float).reshape(-1)
    c_np = np.asarray(problem.c, dtype=float).reshape(-1)

    m, n = A_np.shape

    signs = (
        ["<="] * m
        if constraint_signs is None
        else list(constraint_signs)
    )

    if len(signs) != m or any(
        sign not in ("<=", ">=", "=") for sign in signs
    ):
        raise ValueError("Некорректные знаки ограничений.")

    if variable_constraints is None:
        variable_constraints = ["nonnegative"] * n

    if len(variable_constraints) != n or any(
        value not in ("nonnegative", "free")
        for value in variable_constraints
    ):
        raise ValueError("Некорректные ограничения переменных.")

    if problem.sense == "min":
        c_np = -c_np

    # Свободные переменные заменяем разностью двух неотрицательных.
    free_vars = [
        i for i, value in enumerate(variable_constraints)
        if value == "free"
    ]

    for j in reversed(free_vars):
        A_np = np.insert(A_np, j + 1, -A_np[:, j], axis=1)
        c_np = np.insert(c_np, j + 1, -c_np[j])

    decision_count = A_np.shape[1]

    # Добавляем дополнительные переменные.
    extra_columns = []

    for i, sign in enumerate(signs):
        if sign == "=":
            continue

        column = np.zeros(m)

        if sign == "<=":
            column[i] = 1
        else:
            column[i] = -1

        extra_columns.append(column)

    if extra_columns:
        A_np = np.column_stack([A_np, *extra_columns])

    c_np = np.pad(c_np, (0, len(extra_columns)))

    return A_np, b_np, c_np, decision_count
