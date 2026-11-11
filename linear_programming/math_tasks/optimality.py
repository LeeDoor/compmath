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

    result = ["Проверка на оптимальность:"]

    result.append("\n1) Построение двойственной задачи:")
    A_dual, b_dual, c_dual, sense_dual = build_dual(A, b, c, problem.sense)
    y_names = [f"y{i+1}" for i in range(m)]

    for j in range(n):
        row_str = format_equation(A_dual[j], y_names)
        rel = "=" if variable_constraints[j] == "free" else (">=" if problem.sense == "max" else "<=")
        result.append(f"   {row_str} {rel} {c[j]:g}")
        
    result.append("   Ограничения на y:")
    for i in range(m):
        if constraint_signs[i] == "<=": result.append(f"     y{i+1} >= 0")
        elif constraint_signs[i] == ">=": result.append(f"     y{i+1} <= 0")
        else: result.append(f"     y{i+1} - любое")

    result.append("\n2) Строим множества Jx и Ix:")
    
    Jx = [j for j in range(n) if x[j] > 1e-9 and variable_constraints[j] == "nonnegative"]
    Ix = [i for i in range(m) if A[i] @ x < b[i] - 1e-9]
    
    result.append(f"   Jx = {{ {', '.join(f'x{j+1}' for j in Jx) or 'пусто'} }}")
    result.append(f"   Ix = {{ {', '.join(f'огр.{i+1}' for i in Ix) or 'пусто'} }}")


    result.append("\n3) Составляем систему уравнений:")
    
    rows, rhs = [], []
    

    for j in Jx:
        rows.append(A[:, j])
        rhs.append(c[j])
        result.append(f"   (A^{j+1}, y) = {c[j]:g}  =>  {format_equation(A[:, j], y_names)} = {c[j]:g}")
        

    for j in range(n):
        if variable_constraints[j] == "free":
            rows.append(A[:, j])
            rhs.append(c[j])
            result.append(f"   (A^{j+1}, y) = {c[j]:g} (свободная x{j+1})")
            
  
    for i in Ix:
        vec = np.zeros(m)
        vec[i] = 1.0
        rows.append(vec)
        rhs.append(0.0)
        result.append(f"   y{i+1} = 0")

    if not rows:
        result.append("   Система пуста.")
        return "\n".join(result)

    M = np.asarray(rows)
    v = np.asarray(rhs)
    
    y_approx, residuals, _, _ = np.linalg.lstsq(M, v, rcond=None)
    
    if len(residuals) > 0 and residuals[0] > 1e-6:
        result.append("\n   Система ПРОТИВОРЕЧИВА.")
        result.append("    наилучшее приближение:")
        
        y_str = ", ".join(f"y{i+1} ≈ {val:g}" for i, val in enumerate(y_approx))
        result.append(f"   ({y_str})")
        
        result.append("\n   Но при подстановке этого y в исходные уравнения:")
        for idx, (row, target) in enumerate(zip(M, v)):
            lhs = row @ y_approx
            error = abs(lhs - target)
            if error > 1e-6:
                if idx < len(Jx):
                    j = Jx[idx]
                    result.append(f"   (A^{j+1}, y) = {lhs:.4g}, а должно быть {target:g} (ошибка {error:.4g})")
                elif idx < len(Jx) + sum(1 for vc in variable_constraints if vc == "free"):
                    result.append(f"   (A^j, y) = {lhs:.4g}, а должно быть {target:g} (свободная) (ошибка {error:.4g})")
                else:
                    i = Ix[idx - len(Jx) - sum(1 for vc in variable_constraints if vc == "free")]
                    result.append(f"   y{i+1} = {lhs:.4g}, а должно быть 0 (ошибка {error:.4g})")
                    
        result.append("\nВЫВОД: план x НЕ является оптимальным.")
        return "\n".join(result)
        
    y_approx[np.abs(y_approx) < 1e-7] = 0.0
    y_str = ", ".join(f"y{i+1} = {val:g}" for i, val in enumerate(y_approx))
    result.append(f"\n   Найдено решение системы: ({y_str})")


    result.append("\n4) Проверяем решения на двойственной задаче:")
    
    valid_y = True
    for i in range(m):
        if constraint_signs[i] == "<=" and y_approx[i] < -1e-9:
            result.append(f"   Ошибка: y{i+1} = {y_approx[i]:g} < 0")
            valid_y = False
        elif constraint_signs[i] == ">=" and y_approx[i] > 1e-9:
            result.append(f"   Ошибка: y{i+1} = {y_approx[i]:g} > 0")
            valid_y = False
            
    dual_lhs = A.T @ y_approx
    for j in range(n):
        if variable_constraints[j] == "nonnegative":
            if problem.sense == "max" and dual_lhs[j] < c[j] - 1e-9:
                result.append(f"   Ошибка: (A^T y)_{j+1} = {dual_lhs[j]:g} < c_{j+1} = {c[j]:g}")
                valid_y = False
            elif problem.sense == "min" and dual_lhs[j] > c[j] + 1e-9:
                result.append(f"   Ошибка: (A^T y)_{j+1} = {dual_lhs[j]:g} > c_{j+1} = {c[j]:g}")
                valid_y = False
        else: # free
            if abs(dual_lhs[j] - c[j]) > 1e-9:
                result.append(f"   Ошибка: (A^T y)_{j+1} = {dual_lhs[j]:g} != c_{j+1} = {c[j]:g} (свободная x)")
                valid_y = False

    if not valid_y:
        result.append("\nВЫВОД: найденное y недопустимо. План x НЕ является оптимальным.")
        return "\n".join(result)

    result.append("   Все условия выполняются.")
    result.append("\nВЫВОД: план x ЯВЛЯЕТСЯ оптимальным.")
    
    return "\n".join(result)