"""Одновременное решение прямой и двойственной задач."""

from .problem import LinearProblem


def solve_primal_dual(problem: LinearProblem) -> None:
    """Одновременно решить прямую и двойственную задачи."""
    raise NotImplementedError(
        "Совместное решение прямой и двойственной задач пока не реализовано: "
        "добавьте алгоритм в solve_primal_dual."
    )
