"""Приведение задачи к канонической форме."""

import numpy.typing as npt

from ..problem import Sense, create_problem
from ._stub import describe_linear_problem, run_stub
from .simplex import ConstraintSign, VariableConstraint


def to_canonical_form(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
    constraint_signs: list[ConstraintSign] | None = None,
    variable_constraints: list[VariableConstraint] | None = None,
) -> str:
    """Преобразовать задачу в каноническую форму."""
    problem = create_problem(A, b, c, sense)
    return run_stub("приведение к канонической форме", describe_linear_problem(problem))
