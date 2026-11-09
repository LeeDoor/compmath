"""Приведение задачи к канонической форме."""

import numpy.typing as npt
import numpy as np
from ..problem import Sense, create_problem
from ..display_methods.display_function import display_function

def to_canonical_form(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
    signs: list[str] | None = None,
    free_vars: list[int] | None = None,
) -> str:
    problem = create_problem(A, b, c, sense)
    A_np = np.asarray(problem.A, dtype=float)
    b_np = np.asarray(problem.b, dtype=float).reshape(-1)
    c_np = np.asarray(problem.c, dtype=float).reshape(-1)

    m, n = A_np.shape
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
        
        row_str = " + ".join(f"{row[j]:g}*x{j+1}" for j in range(n))
        
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

    if new_columns:
        A_final = np.hstack([A_np, np.column_stack(new_columns)])
        c_final = c_np
    else:
        A_final = A_np
        c_final = c_np
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
