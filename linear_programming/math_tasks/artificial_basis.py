"""Метод искусственного базиса."""

import numpy.typing as npt

from ..problem import Sense, create_problem
from ._stub import describe_linear_problem, run_stub
from .simplex import ConstraintSign, VariableConstraint


def solve_artificial_basis(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
    constraint_signs: list[ConstraintSign] | None = None,
    variable_constraints: list[VariableConstraint] | None = None,
) -> str:
    """Решить СЛАУ с целевой функцией методом искусственного базиса."""
    problem = create_problem(A, b, c, sense)
    return run_stub("метод искусственного базиса", describe_linear_problem(problem))
