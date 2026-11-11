"""Проверка решения на оптимальность."""

import numpy as np
import numpy.typing as npt
from ..problem import LinearProblem
from .standard_form import to_standard_form, format_equation
from .primal_dual import build_dual


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

    result.append("1) Приводим задачу к стандартной форме: \n")
    
    std_text, A_s, b_s, c_s, x_s, var_names = to_standard_form(
        A, b, c, 
        problem.sense, 
        constraint_signs, 
        variable_constraints, 
        x=x, 
        return_data=True
    )
    result.append(std_text)
    
    m_s, n_s = A_s.shape
    tolerance = 1e-9
    violated_variables = np.flatnonzero(x_s < -tolerance)
    if len(violated_variables):
        result.append("\nОшибка: нарушены условия неотрицательности для "
                      + ", ".join(f"x{i+1}" for i in violated_variables) + ".")
        result.append("\nВЫВОД: план НЕ является оптимальным.")
        return "\n".join(result)

    constraint_values = A_s @ x_s
    violated_constraints = np.flatnonzero(
        constraint_values > b_s + tolerance
    )
    if len(violated_constraints):
        result.append(
            "\nОшибка: нарушены ограничения стандартной формы: "
            + ", ".join(f"огр.{i+1}" for i in violated_constraints)
            + "."
        )
        result.append("\nВЫВОД: план НЕ является оптимальным.")
        return "\n".join(result)

    y_names = [f"y{i+1}" for i in range(m_s)]
    result.append("\n Построение двойственной задачи:")
    A_dual, b_dual, c_dual, sense_dual = build_dual(A_s, b_s, c_s, "max")
    
    result.append(f"  g(y) = {format_equation(c_dual.flatten(), y_names)} -> {sense_dual}")
    for j in range(n_s):
        row_str = format_equation(A_dual[j], y_names)
        result.append(f"     {row_str} >= {b_dual[j, 0]:g}")
    
    result.append("  Ограничения на знаки y:")
    for i in range(m):
        result.append(f"     y{i+1} >= 0")

    result.append("\nСтроим множества Jx и Ix:")
    Jx = [j for j in range(n_s) if x_s[j] > 1e-9]
    Ix = [i for i in range(m_s) if A_s[i] @ x_s < b_s[i]- 1e-9]
    
    result.append(f"   Jx = {{ {', '.join(f'x{j+1}' for j in Jx) or 'пусто'} }} (переменные > 0)")
    result.append(f"   Ix = {{ {', '.join(f'огр.{i+1}' for i in Ix) or 'пусто'} }} (ограничения строго < b)")
    result.append("\nСоставляем систему уравнений:")
    rows, rhs = [], []
    
    for j in Jx:
        rows.append(A_s[:, j])
        rhs.append(c_s[j])
        result.append(f"   (A^{j+1}, y) = {c_s[j]:g}  =>  {format_equation(A_s[:, j], y_names)} = {c_s[j]:g}")
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
        
    dual_lhs = A_s.T @ y_approx
    if np.any(dual_lhs < c_s - tolerance):
        result.append("   Ошибка: не выполняется A^T y >= c.")
        result.append("\nВЫВОД: план НЕ является оптимальным.")
        return "\n".join(result)

    result.append("   Все ограничения двойственной выполняются.")
    result.append("\nВЫВОД: план x ЯВЛЯЕТСЯ оптимальным.")
    
    return "\n".join(result)