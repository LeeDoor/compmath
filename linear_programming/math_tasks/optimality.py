"""Проверка решения на оптимальность."""

import numpy as np
import numpy.typing as npt
from ..problem import LinearProblem
from .standard_form import format_equation
from .primal_dual import format_vector, build_dual


def check_optimality(
    problem: LinearProblem, 
    primal_solution: object, 
    constraint_signs: list[str] | None = None,
    variable_constraints: list[str] | None = None,
) -> str:
    A = np.asarray(problem.A, dtype=float)
    b = np.asarray(problem.b, dtype=float).reshape(-1)
    c = np.asarray(problem.c, dtype=float).reshape(-1)
    x = np.asarray(primal_solution, dtype=float).reshape(-1)

    m, n = A.shape
    if len(x) != n:
        raise ValueError("Длина вектора решения должна совпадать с числом неизвестных.")
    if constraint_signs is None: constraint_signs = ["<="] * m
    if variable_constraints is None: variable_constraints = ["nonnegative"] * n

    result = ["Проверка на оптимальность: "]

    
    m_s, n_s = A.shape
    y_names = [f"y{i+1}" for i in range(m_s)]
    result.append("\n Построение двойственной задачи:")
    A_dual, b_dual, c_dual, sense_dual = build_dual(A, b, c, "max")
    
    result.append(f"  g(y) = {format_equation(c_dual.flatten(), y_names)} -> {sense_dual}")
    for j in range(n_s):
        row_str = format_equation(A_dual[j], y_names)
        result.append(f"     {row_str} >= {b_dual[j, 0]:g}")
    
    result.append("  Ограничения на знаки y:")
    for i in range(m):
        result.append(f"     y{i+1} >= 0")

    result.append("\nСтроим множества Jx и Ix:")
    Jx = [j for j in range(n_s) if x[j] > 1e-9]
    Ix = [i for i in range(m_s) if A[i] @ x < b[i]- 1e-9]
    
    result.append(f"   Jx = {{ {', '.join(f'x{j+1}' for j in Jx) or 'пусто'} }} (переменные > 0)")
    result.append(f"   Ix = {{ {', '.join(f'огр.{i+1}' for i in Ix) or 'пусто'} }} (ограничения строго < b)")
    result.append("\nСоставляем систему уравнений:")
    rows, rhs = [], []
    
    for j in Jx:
        rows.append(A[:, j])
        rhs.append(c[j])
        result.append(f"   (A^{j+1}, y) = {c[j]:g}  =>  {format_equation(A[:, j], y_names)} = {c[j]:g}")
    for i in Ix:
        vec = np.zeros(m_s)
        vec[i] = 1.0
        rows.append(vec)
        rhs.append(0.0)
        result.append(f"   y{i+1} = 0")

    if not rows:
        result.append("   Система равенств пуста.")

    dual_rows = list(rows)
    dual_rhs = list(rhs)
    dual_signs = ["="] * len(rows)
    for j in range(n_s):
        dual_rows.append(A_s[:, j])
        dual_rhs.append(c_s[j])
        dual_signs.append(">=")

    from .artificial_basis import _solve_artificial_basis_impl
    from .steps import StepReport

    try:
        y_solution = _solve_artificial_basis_impl(
            np.asarray(dual_rows),
            np.asarray(dual_rhs),
            np.zeros(m_s),
            "max",
            dual_signs,
            ["nonnegative"] * m_s,
            report=StepReport(),
        )
    except ValueError as error:
        if not str(error).startswith("Задача несовместна"):
            raise
        result.append(
            "\n   Нет двойственных переменных, удовлетворяющих "
            "условиям дополнительной нежёсткости."
        )
        result.append("\nВЫВОД: план НЕ является оптимальным.")
        return "\n".join(result)

    y_approx = np.asarray(y_solution, dtype=float).reshape(-1)
    y_approx[np.abs(y_approx) < 1e-7] = 0.0
   
    y_str = ", ".join(f"y{i+1} = {round(v, 6):g}" for i, v in enumerate(y_approx))
    result.append(f"\n   Найдено решение системы: ({y_str})")

    result.append("\n Проверяем решения на двойственной задаче:")
    
    if np.any(y_approx < -tolerance):
        result.append("   Ошибка: некоторые y_i < 0.")
        result.append("\nВЫВОД: план НЕ является оптимальным.")
        return "\n".join(result)
        
    dual_lhs = A.T @ y_approx
    if np.any(dual_lhs < c-1e-9):
        result.append("   Ошибка: не выполняется A^T y >= c.")
        result.append("\nВЫВОД: план НЕ является оптимальным.")
        return "\n".join(result)

    result.append("   Все ограничения двойственной выполняются.")
    result.append("\nВЫВОД: план x ЯВЛЯЕТСЯ оптимальным.")
    
    return "\n".join(result)