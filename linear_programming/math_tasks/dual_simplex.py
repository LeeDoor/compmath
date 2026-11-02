"""Двойственный симплекс-метод."""

import numpy.typing as npt

from ..problem import Sense, create_problem
from ._stub import describe_linear_problem, run_stub


def solve_dual_simplex(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
) -> str:
    """Решить СЛАУ с целевой функцией двойственным симплекс-методом."""
    problem = create_problem(A, b, c, sense)
    return run_stub("двойственный симплекс-метод", describe_linear_problem(problem))
