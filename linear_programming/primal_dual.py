"""Одновременное решение прямой и двойственной задач."""

import numpy as np
import numpy.typing as npt

from .problem import Sense, create_problem


def solve_primal_dual(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
) -> np.matrix:
    """Решить прямую и двойственную задачи по СЛАУ и целевой функции."""
    create_problem(A, b, c, sense)
    raise NotImplementedError(
        "Совместное решение прямой и двойственной задач пока не реализовано: "
        "добавьте алгоритм в solve_primal_dual."
    )
