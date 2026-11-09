"""Приведение задачи к стандартной форме."""

import numpy.typing as npt
import numpy as np
from ..problem import Sense, create_problem
from ..display_methods.display_function import display_function

def to_standard_form(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
    signs: list[str] | None = None,
    free_vars: list[int] | None = None
) -> str:
    problem = create_problem(A, b, c, sense)
    A_np = np.asarray(problem.A, dtype=float)
    c_np = np.asarray(problem.c, dtype=float).reshape(-1)
    b_np = np.asarray(problem.b, dtype=float).reshape(-1)

    m, n = A_np.shape

    result = ["Приведение к стандартной форме:"]
    if problem.sense == "min":
        result.append("1) Умножаем целевую функцию на -1, чтобы перейти к максимизации:")
        c_np = -c_np
        result.append(f"   f(x) = {display_function(c_np)} -> max")
    else:
        result.append(f"1) Задача уже на максимум: f(x) = {display_function(c_np)} -> max")

    if free_vars:
        result.append("2) Замена свободных переменных (x = x' - x''):")
        for j in reversed(sorted(free_vars)):
            A_np = np.insert(A_np, j + 1, -A_np[:, j], axis=1)
            c_np = np.insert(c_np, j + 1, -c_np[j])
            result.append(f"   x{j+1} -> x'{j+1} - x''{j+1}")
        n = A_np.shape[1]


    result.append("3) Преобразование ограничений:")
    A_final = []
    b_final = []

    for i in range(m):
        row = A_np[i]
        val = b_np[i]
        sign = signs[i]
        row_str = " + ".join(f"{row[j]:g}*x{j+1}" for j in range(n))

        if sign == "<=":
            A_final.append(row)
            b_final.append(val)
            result.append(f"   {row_str} <= {val:g}  (без изменений)")

        elif sign == ">=":
            new_row = -row
            new_val = -val
            A_final.append(new_row)
            b_final.append(new_val)
            new_row_str = " + ".join(f"{new_row[j]:g}*x{j+1}" for j in range(n))
            result.append(f"{new_row_str} >= {new_val:g}")

        elif sign == "=":
            A_final.append(row)
            b_final.append(val)
            A_final.append(-row)
            b_final.append(-val)
            result.append(f"   {row_str} = {val:g} -> разбили на два '<='")
    result.append("4) Условия неотрицательности: все x >= 0")

    result.append("\nИТОГОВАЯ СТАНДАРТНАЯ ФОРМА:")
    result.append(f"f(x) = {display_function(c_np)} -> max")
    for i in range(len(A_final)):
        row_str = " + ".join(f"{A_final[i][j]:g}*x{j+1}" for j in range(n))
        result.append(f"  {row_str} <= {b_final[i]:g}")
    
    return "\n".join(result)


