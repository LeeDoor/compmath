"""Общие помощники для явно обозначенных заглушек алгоритмов."""

import numpy as np

from ..problem import LinearProblem


def describe_linear_problem(problem: LinearProblem) -> str:
    """Вернуть краткое описание условия линейной задачи."""
    return (
        f"A={problem.A.tolist()}, b={problem.b.T.tolist()}, "
        f"c={problem.c.tolist()}, направление={problem.sense}"
    )


def describe_transport_problem(
    costs: object, supply: object, demand: object
) -> str:
    """Вернуть краткое описание условия транспортной задачи."""
    return (
        f"тарифы={np.asarray(costs).tolist()}, "
        f"запасы={np.asarray(supply).reshape(-1).tolist()}, "
        f"потребности={np.asarray(demand).reshape(-1).tolist()}"
    )


def run_stub(task_name: str, condition: str) -> str:
    """Напечатать условие заглушки и вернуть текстовый ответ-заглушку."""
    print(f"Выполняется заглушка для задачи «{task_name}» с условием: {condition}")
    return f"Заглушка для задачи «{task_name}»: решение пока не реализовано."
