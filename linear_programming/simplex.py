"""Симплекс-метод."""

import numpy as np
import numpy.typing as npt

from .problem import Sense, create_problem


def solve_simplex(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
) -> np.matrix:
    """Решить СЛАУ с целевой функцией методом симплекс-метода."""
    create_problem(A, b, c, sense)
    raise NotImplementedError(
        "Симплекс-метод пока не реализован: добавьте алгоритм в solve_simplex."
    )
