"""Двойственный симплекс-метод."""

import numpy as np
import numpy.typing as npt

from ..problem import Sense, create_problem


def solve_dual_simplex(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
) -> np.matrix:
    """Решить СЛАУ с целевой функцией двойственным симплекс-методом."""
    create_problem(A, b, c, sense)
    raise NotImplementedError(
        "Двойственный симплекс-метод пока не реализован: "
        "добавьте алгоритм в solve_dual_simplex."
    )
