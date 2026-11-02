"""Метод искусственного базиса."""

import numpy.typing as npt

from ..problem import Sense, create_problem
from ._stub import describe_linear_problem, run_stub


def solve_artificial_basis(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
) -> str:
    """Решить СЛАУ с целевой функцией методом искусственного базиса."""
    problem = create_problem(A, b, c, sense)
    return run_stub("метод искусственного базиса", describe_linear_problem(problem))
