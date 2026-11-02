"""Вывод расширенных матриц."""

import sympy as sp


def _as_sympy_matrix(values: object) -> sp.Matrix:
    data = values.tolist() if hasattr(values, "tolist") else values
    return sp.Matrix(data)


def display_augmented_matrix(
    A: object, b: object, title: str | None = None
) -> None:
    """Напечатать расширенную матрицу [A | b]."""
    matrix_A = _as_sympy_matrix(A)
    vector_b = _as_sympy_matrix(b)
    if vector_b.cols != 1:
        if vector_b.rows == 1:
            vector_b = vector_b.T
        else:
            raise ValueError("b должен быть вектором.")
    if matrix_A.rows != vector_b.rows:
        raise ValueError("Число строк A и b должно совпадать.")
    if title:
        print(f"{title}:")
    print(sp.pretty(matrix_A.row_join(vector_b)))
