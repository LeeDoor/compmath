"""Приведение задачи к стандартной форме."""

import numpy.typing as npt

from ..problem import Sense, create_problem
from ._stub import describe_linear_problem, run_stub


def to_standard_form(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
) -> str:
    """Преобразовать задачу в стандартную форму."""
    problem = create_problem(A, b, c, sense)
    return run_stub("приведение к стандартной форме", describe_linear_problem(problem))
