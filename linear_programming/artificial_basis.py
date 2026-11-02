"""Метод искусственного базиса."""

import numpy as np
import numpy.typing as npt

from .problem import Sense, create_problem


def solve_artificial_basis(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
) -> np.matrix:
    """Решить СЛАУ с целевой функцией методом искусственного базиса."""
    create_problem(A, b, c, sense)
    raise NotImplementedError(
        "Метод искусственного базиса пока не реализован: "
        "добавьте алгоритм в solve_artificial_basis."
    )
