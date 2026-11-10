"""Приведение задачи к стандартной форме."""

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


def to_standard_form(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
    constraint_signs: list[ConstraintSign] | None = None,
    variable_constraints: list[VariableConstraint] | None = None,
    x: npt.ArrayLike | None = None,           # <--- ДОБАВИЛИ x
    return_data: bool = False,                # <--- ДОБАВИЛИ ФЛАГ
):
    problem = create_problem(A, b, c, sense)
    A_np = np.asarray(problem.A, dtype=float)
    c_np = np.asarray(problem.c, dtype=float).reshape(-1)
    b_np = np.asarray(problem.b, dtype=float).reshape(-1)
    x_np = np.asarray(x, dtype=float).reshape(-1) if x is not None else None

    m, n = A_np.shape
    
    signs = ["<="] * m if constraint_signs is None else list(constraint_signs)
    if len(signs) != m or any(sign not in ("<=", ">=", "=") for sign in signs):
        raise ValueError("constraint_signs должны задавать знак '<=', '>=' или '=' для каждого ограничения.")

    if variable_constraints is None:
        variable_constraints = ["nonnegative"] * n
    if len(variable_constraints) != n or any(c not in ("nonnegative", "free") for c in variable_constraints):
        raise ValueError("variable_constraints должны задавать 'nonnegative' или 'free' для каждой переменной.")

    free_vars = [i for i, v in enumerate(variable_constraints) if v == "free"]
    var_names = [f"x{i+1}" for i in range(n)]

    result = ["Приведение к стандартной форме:"]
    
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
            

            if x_np is not None:
                x_pos = max(x_np[j], 0.0)
                x_neg = max(-x_np[j], 0.0)
                x_np[j] = x_pos
                x_np = np.insert(x_np, j + 1, x_neg)
                
            result.append(f"   x{j+1} -> x{j+1}' - x{j+1}''")
        n = A_np.shape[1]
    else:
        result.append("2) Свободных переменных нет.")

    result.append("3) Преобразование ограничений:")
    A_final, b_final = [], []

    for i in range(m):
        row, val, sign = A_np[i], b_np[i], signs[i]
        row_str = format_equation(row, var_names)

        if sign == "<=":
            A_final.append(row)
            b_final.append(val)
            result.append(f"   {row_str} <= {val:g} ")
        elif sign == ">=":
            new_row, new_val = -row, -val
            A_final.append(new_row)
            b_final.append(new_val)
            new_row_str = format_equation(new_row, var_names)
            result.append(f"   {row_str} >= {val:g}  =>  {new_row_str} <= {new_val:g}")
        elif sign == "=":
            A_final.append(row)
            b_final.append(val)
            A_final.append(-row)
            b_final.append(-val)
            result.append(f"   {row_str} = {val:g} -> разбили на два '<='")
            
    result.append("4) Условия неотрицательности: все переменные >= 0")
    result.append("\nИТОГОВАЯ СТАНДАРТНАЯ ФОРМА:")
    result.append(f"f(x) = {format_equation(c_np, var_names)} -> max")
    
    for i in range(len(A_final)):
        row_str = format_equation(A_final[i], var_names)
        result.append(f"  {row_str} <= {b_final[i]:g}")
    

    if return_data:
        return (
            "\n".join(result), 
            np.array(A_final), 
            np.array(b_final), 
            c_np, 
            x_np, 
            var_names
        )
        
    return "\n".join(result)