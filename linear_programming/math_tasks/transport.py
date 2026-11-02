"""Решение транспортной задачи."""

import numpy.typing as npt

from ._stub import describe_transport_problem, run_stub


def solve_transport(
    costs: npt.ArrayLike, supply: npt.ArrayLike, demand: npt.ArrayLike
) -> str:
    """Решить транспортную задачу по матрице тарифов, запасам и потребностям."""
    return run_stub(
        "транспортная задача",
        describe_transport_problem(costs, supply, demand),
    )
