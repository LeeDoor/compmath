"""Одновременное решение прямой и двойственной задач."""

import numpy.typing as npt
import numpy as np
from ..problem import Sense, create_problem
from ._stub import describe_linear_problem, run_stub

def _format_vector(values, prefix):
    parts = [f"{prefix}{i} = {v:g}" for i, v in enumerate(values, start=1)]
    return ", ".join(parts)

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
)->str:
    problem = create_problem(A, b, c, sense)
    A_np = np.asarray(problem.A, dtype=float)
    b_np = np.asarray(problem.b, dtype=float).reshape(-1)
    c_np = np.asarray(problem.c, dtype=float).reshape(-1)

    A_dual, b_dual, c_dual, sense_dual = build_dual(A_np, b_np, c_np, sense)

    x, basis, delta, A_simplex = solve_simplex(A_np, b_np, c_np, sense)

    if A_simplex.shape[1] > len(c_np):
        n_slack = A_simplex.shape[1] - len(c_np)
        c_simplex = np.concatenate([c_np, np.zeros(n_slack)])
    else:
        c_simplex = c_np

    y1=solution_inverse_matrix(A_simplex, c_simplex, basis)
    y2=solution_from_delta(A_simplex, c_simplex, basis)

    f_val = c_np @ x
    g_val = b_np @ y1
    
    if not np.isclose(f_val, g_val, atol=1e-9):
        raise ValueError(
            f"Нарушена теорема двойственности: f(x*)={f_val:g} != g(y*)={g_val:g}"
        )
    x_str = _format_vector(x, "x")
    y1_str = _format_vector(y1, "y")
    y2_str = _format_vector(y2, "y")
    return (
        f"x* = ({x_str})\n"
        f"Способ 1 (система):   y* = ({y1_str})\n"
        f"Способ 2 (обратная):  y* = ({y2_str})\n"
        f"f(x*) = g(y*) = {f_val:g}"
    )
    
