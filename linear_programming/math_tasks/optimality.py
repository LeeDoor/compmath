"""Проверка решения на оптимальность."""

from ..problem import LinearProblem
from ._stub import describe_linear_problem, run_stub


def check_optimality(
    problem: LinearProblem, primal_solution: object, dual_solution: object | None = None
) -> str:
    """Проверить оптимальность прямого решения и, при наличии, двойственного."""
    condition = (
        f"{describe_linear_problem(problem)}, "
        f"прямой вектор решения={primal_solution}"
    )
    if dual_solution is not None:
        condition += f", двойственный вектор решения={dual_solution}"
    return run_stub("проверка на оптимальность", condition)
