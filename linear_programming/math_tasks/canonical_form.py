"""Приведение задачи к канонической форме."""

import numpy.typing as npt
import numpy as np
from ..problem import Sense, create_problem
from .simplex import ConstraintSign, VariableConstraint


def format_equation(row, var_names):
    terms = []
    for coef, name in zip(row, var_names):
        if abs(coef) < 1e-9:
            continue
            
        if abs(coef - 1.0) < 1e-9:
            terms.append(f"{name}")
        elif abs(coef + 1.0) < 1e-9:
            terms.append(f"-{name}")
        else:
            terms.append(f"{coef:g}*{name}")
            
    if not terms:
        return "0"
        
    eq_str = terms[0]
    for term in terms[1:]:
        if term.startswith("-"):
            eq_str += f" - {term[1:]}"
        else:
            eq_str += f" + {term}"
    return eq_str


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
    original_n = n 
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

    free_vars = [
        index
        for index, constraint in enumerate(variable_constraints)
        if constraint == "free"
    ]
    
    var_names = [f"x{i+1}" for i in range(n)]

    result = ["Приведение к канонической форме:"]


    if problem.sense == "min":
            result.append("1) Умножаем целевую функцию на -1, чтобы перейти к максимизации:")
            c_np = -c_np
            result.append(f"   f(x) = {format_equation(c_np, var_names)} -> max")
    else:
        result.append(f"1) Задача уже на максимум: f(x) = {format_equation(c_np, var_names)} -> max")


    if free_vars:
        result.append("2) Замена свободных переменных (x = x' - x''):")
        for j in reversed(sorted(free_vars)):
            A_np = np.insert(A_np, j + 1, -A_np[:, j], axis=1)
            c_np = np.insert(c_np, j + 1, -c_np[j])
            
            var_names[j] = f"x{j+1}'"
            var_names.insert(j + 1, f"x{j+1}''")
            
            result.append(f"   x{j+1} -> x{j+1}' - x{j+1}''")
        n = A_np.shape[1]
    else:
        result.append("2) Свободных переменных нет.")


    result.append("3) Преобразование ограничений в уравнения:")
    new_columns = [] 
    var_count = original_n    
    
    for i in range(m):
        row = A_np[i]
        val = b_np[i]
        sign = signs[i]

        row_str = format_equation(row, var_names)
        
        if sign == "<=":
            var_count += 1
            col = np.zeros(m)
            col[i] = 1.0
            new_columns.append(col)
            var_names.append(f"x{var_count}")
            result.append(f"   {row_str} <= {val:g}  =>  {row_str} + x{var_count} = {val:g}")
            
        elif sign == ">=":
            var_count += 1
            col = np.zeros(m)
            col[i] = -1.0
            new_columns.append(col)
            var_names.append(f"x{var_count}")
            result.append(f"   {row_str} >= {val:g}  =>  {row_str} - x{var_count} = {val:g}")
            
        elif sign == "=":
            result.append(f"   {row_str} = {val:g}   (уже уравнение)")


    if new_columns:
        A_final = np.hstack([A_np, np.column_stack(new_columns)])
        c_final = np.pad(c_np, (0, len(new_columns)))
    else:
        A_final = A_np
        c_final = c_np

    result.append("4) Все переменные неотрицательны.")
    

    result.append("\nИТОГОВАЯ КАНОНИЧЕСКАЯ ФОРМА:")
    

    result.append(f"f(x) = {format_equation(c_final, var_names)} -> max")


    for i in range(m):
        eq_str = format_equation(A_final[i], var_names)
        result.append(f"  {eq_str} = {b_np[i]:g}")

    result.append(f"  {', '.join(var_names)} >= 0")

    return "\n".join(result)