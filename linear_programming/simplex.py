"""Симплекс-метод."""

from .problem import LinearProblem


def solve_simplex(problem: LinearProblem) -> None:
    """Решить задачу симплекс-методом."""
    raise NotImplementedError(
        "Симплекс-метод пока не реализован: добавьте алгоритм в solve_simplex."
    )
