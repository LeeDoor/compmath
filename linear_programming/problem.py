"""Общая модель задачи линейного программирования."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import sympy as sp

Sense = Literal["max", "min"]


@dataclass(frozen=True)
class LinearProblem:
    """Задача вида: оптимизировать c*x при ограничениях A*x = b."""

    A: np.matrix
    b: np.matrix
    c: np.matrix
    sense: Sense


def _to_sympy_matrix(values: object, name: str) -> sp.Matrix:
    try:
        if isinstance(values, sp.MatrixBase):
            matrix = sp.Matrix(values)
        else:
            data = values.tolist() if hasattr(values, "tolist") else values
            matrix = sp.Matrix(data)
    except (TypeError, ValueError, sp.SympifyError) as error:
        raise ValueError(f"{name} должна быть непустой числовой матрицей.") from error

    if matrix.rows == 0 or matrix.cols == 0:
        raise ValueError(f"{name} не должна быть пустой.")
    if any(not value.is_number for value in matrix):
        raise ValueError(f"{name} должна содержать только числа.")
    return matrix


def _as_numpy_matrix(matrix: sp.Matrix) -> np.matrix:
    return np.matrix(matrix.tolist(), dtype=object)


def create_problem(A: object, b: object, c: object, sense: Sense = "max") -> LinearProblem:
    """Проверить и собрать задачу; векторы b и c можно передавать списками."""
    if sense not in ("max", "min"):
        raise ValueError("sense должен быть равен 'max' или 'min'.")

    sympy_A = _to_sympy_matrix(A, "A")
    sympy_b = _to_sympy_matrix(b, "b")
    sympy_c = _to_sympy_matrix(c, "c")

    if sympy_b.cols != 1:
        if sympy_b.rows == 1:
            sympy_b = sympy_b.T
        else:
            raise ValueError("b должен быть вектором-столбцом.")
    if sympy_c.rows != 1:
        if sympy_c.cols == 1:
            sympy_c = sympy_c.T
        else:
            raise ValueError("c должен быть вектором-строкой или вектором-столбцом.")
    if sympy_A.rows != sympy_b.rows:
        raise ValueError("Число строк A должно совпадать с длиной b.")
    if sympy_A.cols != sympy_c.cols:
        raise ValueError("Число столбцов A должно совпадать с длиной c.")

    return LinearProblem(
        A=_as_numpy_matrix(sympy_A),
        b=_as_numpy_matrix(sympy_b),
        c=_as_numpy_matrix(sympy_c),
        sense=sense,
    )
