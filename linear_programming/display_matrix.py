"""Вывод обычных матриц."""

import sympy as sp


def _as_sympy_matrix(values: object) -> sp.Matrix:
    data = values.tolist() if hasattr(values, "tolist") else values
    return sp.Matrix(data)


def display_matrix(matrix: object, title: str | None = None) -> None:
    """Напечатать матрицу в консоль."""
    if title:
        print(f"{title}:")
    print(sp.pretty(_as_sympy_matrix(matrix)))
