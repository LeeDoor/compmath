"""Формирование строкового представления целевой функции."""

import sympy as sp


def _as_coefficients(values: object) -> list[sp.Expr]:
    if isinstance(values, sp.MatrixBase):
        return list(values)
    data = values.tolist() if hasattr(values, "tolist") else values
    matrix = sp.Matrix(data)
    return list(matrix)


def display_function(
    coefficients: object, variable_names: list[str] | None = None
) -> str:
    """Сформировать выражение линейной функции, например '3*x1 - 2*x2'."""
    values = _as_coefficients(coefficients)
    if variable_names is None:
        variable_names = [f"x{i}" for i in range(1, len(values) + 1)]
    if len(variable_names) != len(values):
        raise ValueError("Количество имен переменных должно совпадать с числом коэффициентов.")

    terms: list[str] = []
    for coefficient, variable in zip(values, variable_names):
        coefficient = sp.sympify(coefficient)
        if coefficient == 0:
            continue
        negative = coefficient.is_negative
        magnitude = -coefficient if negative else coefficient
        if magnitude == 1:
            term = variable
        else:
            term = f"{sp.sstr(magnitude)}*{variable}"
        if not terms:
            terms.append(f"-{term}" if negative else term)
        else:
            terms.append(f" - {term}" if negative else f" + {term}")
    return "".join(terms) or "0"
