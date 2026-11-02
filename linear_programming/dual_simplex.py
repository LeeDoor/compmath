"""Двойственный симплекс-метод."""

from .problem import LinearProblem


def solve_dual_simplex(problem: LinearProblem) -> None:
    """Решить задачу двойственным симплекс-методом."""
    raise NotImplementedError(
        "Двойственный симплекс-метод пока не реализован: "
        "добавьте алгоритм в solve_dual_simplex."
    )
