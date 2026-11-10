"""Одновременное решение прямой и двойственной задач."""

import numpy.typing as npt
import numpy as np
from ..problem import Sense, create_problem
from ..display_methods.display_function import display_function
from .simplex import ConstraintSign, VariableConstraint, solve_simplex

def format_vector(values, prefix):
    parts = [f"{prefix}{i} = {v:g}" for i, v in enumerate(values, start=1)]
    return ", ".join(parts)

def format_expression(row, var_names):
    """Красиво форматирует строку, пропуская нули и правильно ставя знаки."""
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
                     

def build_dual(A, b, c, sense):
    A_dual=A.T
    b_dual=c.reshape(-1, 1) 
    c_dual=b.reshape(1,-1)
    sense_dual= "min" if sense=="max" else "max"
    return A_dual, b_dual, c_dual, sense_dual

def solution_inverse_matrix(A, c, basis):
    B=A[:, basis]
    B_inv=np.linalg.inv(B)
    c_B=c[basis]

    y=c_B @ B_inv
    return y

def solution_from_delta(A, c, basis, delta):
    B=A[:, basis]
    c_B=c[basis]
    delta_B=delta[basis]

    right_side=delta_B+c_B
    return np.linalg.solve(B.T, right_side)

def solve_primal_dual(A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
    constraint_signs: list[ConstraintSign] | None = None,
    variable_constraints: list[VariableConstraint] | None = None,
)->str:
    problem = create_problem(A, b, c, sense)
    A_np = np.asarray(problem.A, dtype=float)
    b_np = np.asarray(problem.b, dtype=float).reshape(-1)
    c_np = np.asarray(problem.c, dtype=float).reshape(-1)

    m, n = A_np.shape
    if constraint_signs is None: constraint_signs = ["<="] * m
    if variable_constraints is None: variable_constraints = ["nonnegative"] * n

    y_names = [f"y{i+1}" for i in range(m)]
    A_dual, b_dual, c_dual, sense_dual = build_dual(A_np, b_np, c_np, sense)
    print(f"\nДВОЙСТВЕННАЯ ЗАДАЧА:")
    print(f"g(y) = {format_expression(b_np, y_names)} -> {sense_dual}")
    

    for j in range(n):
        row_dual = A_dual[j]
        col_str = format_expression(row_dual, y_names)
        
        rel = ("=" if variable_constraints[j] == "free" 
               else (">=" if sense == "max" else "<="))
        print(f"  {col_str} {rel} {c_np[j]:g}")
        

    print("\nОграничения на y:")
    for i in range(m):
        if constraint_signs[i] == "<=":
            print(f"  y{i+1} >= 0")
        elif constraint_signs[i] == ">=":
            print(f"  y{i+1} <= 0")
        else:
            print(f"  y{i+1} - любое (свободное)")



    try:

        x, basis, delta, A_simplex = solve_simplex(
            A_np, b_np, c_np, sense, 
            constraint_signs=constraint_signs, 
            variable_constraints=variable_constraints,
            return_details=True
        )
    except ValueError as e:
        return f"\nОШИБКА РЕШЕНИЯ: {e}"
    
    if A_simplex.shape[1] > len(c_np):
        n_slack = A_simplex.shape[1] - len(c_np)
        c_simplex = np.concatenate([c_np, np.zeros(n_slack)])
    else:
        c_simplex = c_np

    y1=solution_inverse_matrix(A_simplex, c_simplex, basis)
    y2=solution_from_delta(A_simplex, c_simplex, basis, delta)

    x = np.asarray(x).reshape(-1)
    f_val = c_np @ x
    g_val = b_np @ y1
    
    if not np.isclose(f_val, g_val, atol=1e-9):
        raise ValueError(
            f"Нарушена теорема двойственности: f(x*)={f_val:g} != g(y*)={g_val:g}"
        )
    x_str = format_vector(x, "x")
    y1_str = format_vector(y1, "y")
    y2_str = format_vector(y2, "y")
    return (
        f"x* = ({x_str})\n"
        f"Способ 1 (система):   y* = ({y1_str})\n"
        f"Способ 2 (обратная):  y* = ({y2_str})\n"
        f"f(x*) = g(y*) = {f_val:g}"
    )
    
