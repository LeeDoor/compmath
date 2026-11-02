"""Проверка решения на оптимальность."""

from .problem import LinearProblem


def check_optimality(
    problem: LinearProblem, primal_solution: object, dual_solution: object | None = None
) -> bool:
    """Проверить оптимальность прямого решения и, при наличии, двойственного."""
    raise NotImplementedError(
        "Проверка оптимальности пока не реализована: "
        "добавьте критерий в check_optimality."
    )
