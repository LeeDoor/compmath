"""Решение транспортной задачи методом северо-западного угла."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_GRAY = "\033[90m"
_RESET = "\033[0m"


@dataclass
class _ClosedProblem:
    costs: np.ndarray
    supply: np.ndarray
    demand: np.ndarray
    supplier_names: list[str]
    consumer_names: list[str]


def _validate_arguments(
    costs: npt.ArrayLike, supply: npt.ArrayLike, demand: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Преобразовать введённые данные к массивам и проверить неотрицательность."""
    cost_matrix = np.asarray(costs, dtype=float)
    supplies = np.asarray(supply, dtype=float).reshape(-1)
    demands = np.asarray(demand, dtype=float).reshape(-1)

    if (supplies < 0).any() or (demands < 0).any():
        raise ValueError("Запасы и потребности не могут быть отрицательными.")
    return cost_matrix, supplies, demands


def _format_number(value: float) -> str:
    return f"{value:g}"


def _close_problem(
    costs: np.ndarray, supply: np.ndarray, demand: np.ndarray
) -> _ClosedProblem:
    """Добавить фиктивного потребителя или склад, если задача не закрыта."""
    supplier_names = [f"a[{index + 1}]" for index in range(len(supply))]
    consumer_names = [f"b[{index + 1}]" for index in range(len(demand))]
    total_supply = float(supply.sum())
    total_demand = float(demand.sum())

    if total_supply > total_demand:
        difference = total_supply - total_demand
        costs = np.column_stack((costs, np.zeros(costs.shape[0])))
        demand = np.append(demand, difference)
        consumer_names.append("Фиктивный потребитель")
        print(
            "Задача закрыта: добавлен фиктивный потребитель "
            f"с потребностью {_format_number(difference)} и нулевыми тарифами."
        )
    elif total_demand > total_supply:
        difference = total_demand - total_supply
        costs = np.vstack((costs, np.zeros(costs.shape[1])))
        supply = np.append(supply, difference)
        supplier_names.append("Фиктивный склад")
        print(
            "Задача закрыта: добавлен фиктивный склад "
            f"с запасом {_format_number(difference)} и нулевыми тарифами."
        )
    else:
        print("Задача закрытая: общий запас равен общей потребности.")

    print("Условие закрытой задачи:")
    print(f"Тарифы: {costs.tolist()}")
    print(f"Запасы: {supply.tolist()}")
    print(f"Потребности: {demand.tolist()}")
    return _ClosedProblem(costs, supply, demand, supplier_names, consumer_names)


def _northwest_corner(supply: np.ndarray, demand: np.ndarray) -> np.ndarray:
    """Построить план перевозок методом северо-западного угла."""
    allocations = np.zeros((len(supply), len(demand)))
    remaining_supply = supply.copy()
    remaining_demand = demand.copy()
    row = column = 0

    while row < len(remaining_supply) and column < len(remaining_demand):
        amount = min(remaining_supply[row], remaining_demand[column])
        allocations[row, column] += amount
        remaining_supply[row] -= amount
        remaining_demand[column] -= amount

        if np.isclose(remaining_supply[row], 0):
            remaining_supply[row] = 0
        if np.isclose(remaining_demand[column], 0):
            remaining_demand[column] = 0

        if remaining_supply[row] == 0:
            row += 1
        if remaining_demand[column] == 0:
            column += 1

    return allocations


def _display_table(problem: _ClosedProblem, allocations: np.ndarray) -> None:
    """Показать единый результат распределения и стоимость по каждой клетке."""
    headers = [
        f"{name}={_format_number(problem.demand[index])}"
        for index, name in enumerate(problem.consumer_names)
    ]
    row_labels = [
        f"{name}={_format_number(problem.supply[index])}"
        for index, name in enumerate(problem.supplier_names)
    ]
    plain_cells = [
        [
            f"{_format_number(allocations[row, column])} | "
            f"{_format_number(problem.costs[row, column])}"
            for column in range(problem.costs.shape[1])
        ]
        for row in range(problem.costs.shape[0])
    ]
    widths = [
        max(
            len(headers[column]),
            *(len(plain_cells[row][column]) for row in range(problem.costs.shape[0])),
        )
        for column in range(problem.costs.shape[1])
    ]
    label_width = max(len("Склад"), *(len(label) for label in row_labels))

    def format_row(label: str, values: list[str]) -> str:
        padded_values = []
        for index, value in enumerate(values):
            visible_value = value.replace(_GRAY, "").replace(_RESET, "")
            padded_values.append(value + " " * (widths[index] - len(visible_value)))
        return (
            f"| {label:<{label_width}} | "
            + " | ".join(padded_values)
            + " |"
        )

    separator = (
        "+"
        + "-" * (label_width + 2)
        + "+"
        + "+".join("-" * (width + 2) for width in widths)
        + "+"
    )
    print("\nШаг 1. Метод северо-западного угла:")
    print(separator)
    print(format_row("Запасы", headers))
    print(separator)
    for row, label in enumerate(row_labels):
        cells = []
        for column in range(problem.costs.shape[1]):
            cells.append(
                f"{_format_number(allocations[row, column])} | "
                f"{_GRAY}{_format_number(problem.costs[row, column])}{_RESET}"
            )
        print(format_row(label, cells))
    print(separator)


def _display_solution(problem: _ClosedProblem, allocations: np.ndarray) -> None:
    _display_table(problem, allocations)
    total_cost = float(np.sum(allocations * problem.costs))
    print("\nИтоговый план перевозок:")
    print(np.matrix(allocations))
    print(f"Общая стоимость перевозок: {_format_number(total_cost)}")


def solve_transport(
    costs: npt.ArrayLike, supply: npt.ArrayLike, demand: npt.ArrayLike
) -> np.matrix:
    """Закрыть транспортную задачу и построить её начальный план."""
    costs, supply, demand = _validate_arguments(costs, supply, demand)
    problem = _close_problem(costs, supply, demand)
    allocations = _northwest_corner(problem.supply, problem.demand)
    _display_solution(problem, allocations)
    return np.matrix(allocations)
