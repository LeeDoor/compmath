"""Одновременное решение прямой и двойственной задач."""

import numpy.typing as npt

from ..problem import Sense, create_problem
from ._stub import describe_linear_problem, run_stub


def solve_primal_dual(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
) -> str:
    """Решить прямую и двойственную задачи по СЛАУ и целевой функции."""
    problem = create_problem(A, b, c, sense)
    return run_stub(
        "одновременное решение прямой и двойственной задач",
        describe_linear_problem(problem),
    )
