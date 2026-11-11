"""Решение транспортной задачи методом северо-западного угла и потенциалов."""

import re
from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from .steps import StepReport

_GRAY = "\033[90m"
_MAROON = "\033[38;2;128;0;0m"
_SELECTED_BACKGROUND = "\033[48;2;255;242;179m"
_SELECTED_FOREGROUND = "\033[38;2;32;32;32m"
_SELECTED_TARIFF = "\033[38;2;90;90;90m"
_RESET = "\033[0m"
_TOLERANCE = 1e-10
_ANSI_ESCAPE = re.compile(r"\033\[[0-9;]*m")


@dataclass
class _ClosedProblem:
    costs: np.ndarray
    supply: np.ndarray
    demand: np.ndarray
    supplier_names: list[str]
    consumer_names: list[str]


@dataclass
class _PotentialStep:
    row_potentials: np.ndarray
    column_potentials: np.ndarray
    reduced_costs: np.ndarray


def _validate_arguments(
    costs: npt.ArrayLike, supply: npt.ArrayLike, demand: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert entered data to arrays and reject negative quantities."""
    # Приводим входы к числовым массивам единой формы; отрицательные запасы и
    # потребности не допускают допустимого плана перевозок.
    cost_matrix = np.asarray(costs, dtype=float)
    supplies = np.asarray(supply, dtype=float).reshape(-1)
    demands = np.asarray(demand, dtype=float).reshape(-1)
    if (supplies < 0).any() or (demands < 0).any():
        raise ValueError("Запасы и потребности не могут быть отрицательными.")
    return cost_matrix, supplies, demands


def _format_number(value: float) -> str:
    return f"{value:g}"


def _close_problem(
    costs: np.ndarray, supply: np.ndarray, demand: np.ndarray, report: StepReport
) -> _ClosedProblem:
    """Balance the transportation problem by adding a dummy row or column."""
    supplier_names = [f"a[{index + 1}]" for index in range(len(supply))]
    consumer_names = [f"b[{index + 1}]" for index in range(len(demand))]
    total_supply = float(supply.sum())
    total_demand = float(demand.sum())

    # Для допустимого плана общий объём отправок должен совпадать с объёмом
    # потребностей. Несбалансированную задачу превращаем в закрытую добавлением
    # фиктивного участника с нулевыми тарифами.
    if not np.isclose(total_supply, total_demand, rtol=_TOLERANCE, atol=_TOLERANCE):
        if total_supply > total_demand:
            # Излишек запасов отправляется фиктивному потребителю; это не меняет
            # стоимость реальных перевозок и позволяет распределить весь запас.
            difference = total_supply - total_demand
            costs = np.column_stack((costs, np.zeros(costs.shape[0])))
            demand = np.append(demand, difference)
            consumer_names.append("Фиктивный потребитель")
            report.add(
                "Задача закрыта: добавлен фиктивный потребитель "
                f"с потребностью {_format_number(difference)} и нулевыми тарифами."
            )
        else:
            # Недостающий объём покрывает фиктивный склад с нулевыми тарифами,
            # чтобы все реальные потребности вошли в сбалансированный план.
            difference = total_demand - total_supply
            costs = np.vstack((costs, np.zeros(costs.shape[1])))
            supply = np.append(supply, difference)
            supplier_names.append("Фиктивный склад")
            report.add(
                "Задача закрыта: добавлен фиктивный склад "
                f"с запасом {_format_number(difference)} и нулевыми тарифами."
            )
    else:
        report.add("Задача закрытая: общий запас равен общей потребности.")

    report.add("Условие закрытой задачи:")
    report.add(f"Тарифы: {costs.tolist()}")
    report.add(f"Запасы: {supply.tolist()}")
    report.add(f"Потребности: {demand.tolist()}")
    return _ClosedProblem(costs, supply, demand, supplier_names, consumer_names)


def _add_zero_basic_cells(basic: np.ndarray) -> None:
    """Complete a degenerate basis without introducing a cycle."""
    rows, columns = basic.shape
    # Базис транспортной задачи должен содержать rows + columns - 1 клеток.
    # Представляем клетки как рёбра двудольного графа: строки — склады,
    # столбцы — потребители. DSU помогает добавлять клетки без циклов.
    parent = list(range(rows + columns))

    def find(vertex: int) -> int:
        # Сжатие путей ускоряет поиск компоненты связности.
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    def union(first: int, second: int) -> bool:
        # Если вершины уже связаны, добавление ребра создало бы цикл.
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return False
        parent[first_root] = second_root
        return True

    basis_count = 0
    # Положительные перевозки уже образуют часть базиса. В базис включаем только
    # клетки, рёбра которых не замыкают цикл; остальные клетки не входят в дерево.
    for row, column in zip(*np.where(basic)):
        if union(int(row), rows + int(column)):
            basis_count += 1
        else:
            basic[row, column] = False

    # Достраиваем лес до остовного дерева фиктивными нулевыми базисными клетками.
    # Такое дополнение не меняет план перевозок, но делает потенциалы определимыми.
    for row in range(rows):
        for column in range(columns):
            if basis_count == rows + columns - 1:
                return
            if not basic[row, column] and union(row, rows + column):
                basic[row, column] = True
                basis_count += 1


def _northwest_corner(
    supply: np.ndarray, demand: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build the initial feasible plan using the northwest-corner rule."""
    allocations = np.zeros((len(supply), len(demand)))
    remaining_supply = supply.copy()
    remaining_demand = demand.copy()
    # Начинаем с северо-западной клетки (верхняя левая) и последовательно
    # перемещаемся вправо или вниз по мере исчерпания потребности/запаса.
    row = column = 0

    while row < len(remaining_supply) and column < len(remaining_demand):
        # Максимальная допустимая перевозка в текущую клетку — меньшая из
        # оставшегося запаса склада и оставшейся потребности потребителя.
        amount = min(remaining_supply[row], remaining_demand[column])
        allocations[row, column] += amount
        remaining_supply[row] -= amount
        remaining_demand[column] -= amount

        # Округляем очень малые остатки до нуля, чтобы ошибки float не мешали
        # определять, какую строку или колонку таблицы закрыть.
        if np.isclose(remaining_supply[row], 0, atol=_TOLERANCE):
            remaining_supply[row] = 0
        if np.isclose(remaining_demand[column], 0, atol=_TOLERANCE):
            remaining_demand[column] = 0

        # Если исчерпан запас, переходим к следующему складу; если закрыта
        # потребность — к следующему потребителю. При равенстве сдвигаемся по
        # обоим измерениям, а недостающие нулевые базисные клетки добавим ниже.
        if remaining_supply[row] == 0:
            row += 1
        if remaining_demand[column] == 0:
            column += 1

    # Положительные перевозки задают базисные клетки. При вырожденном плане
    # их может быть меньше rows + columns - 1, поэтому добавляем нулевые клетки.
    basic = allocations > 0
    _add_zero_basic_cells(basic)
    return allocations, basic


def _calculate_potentials(
    costs: np.ndarray, basic: np.ndarray
) -> _PotentialStep:
    """Find row and column potentials from the basic-cell equations."""
    rows, columns = costs.shape
    row_potentials = np.full(rows, np.nan)
    column_potentials = np.full(columns, np.nan)
    # Потенциалы не меняются при прибавлении константы ко всем u и вычитании её
    # из всех v, поэтому фиксируем один из них для однозначного распространения.
    row_potentials[0] = 0

    # В каждой базисной клетке выполняется u[i] + v[j] = c[i,j].
    # Многократно применяем эти равенства, пока не найдём потенциалы всех строк
    # и столбцов, соединённых базисными клетками.
    changed = True
    while changed:
        changed = False
        for row, column in zip(*np.where(basic)):
            if not np.isnan(row_potentials[row]) and np.isnan(
                column_potentials[column]
            ):
                column_potentials[column] = (
                    costs[row, column] - row_potentials[row]
                )
                changed = True
            elif np.isnan(row_potentials[row]) and not np.isnan(
                column_potentials[column]
            ):
                row_potentials[row] = costs[row, column] - column_potentials[column]
                changed = True

    # Для свободной клетки Δc = c[i,j] - u[i] - v[j]. В задаче минимизации
    # неотрицательные оценки означают, что ввод этой клетки в план не уменьшит
    # стоимость. Для базисных клеток оценка равна нулю по построению.
    reduced_costs = costs - (
        row_potentials[:, np.newaxis] + column_potentials[np.newaxis, :]
    )
    reduced_costs[basic] = 0
    return _PotentialStep(row_potentials, column_potentials, reduced_costs)


def _find_cycle(basic: np.ndarray, entering: tuple[int, int]) -> list[tuple[int, int]]:
    """Find the unique cycle formed by adding an entering cell to the basis tree."""
    rows, columns = basic.shape
    # Базисные клетки задают остовное дерево между вершинами-строками и
    # вершинами-столбцами. Добавление свободной входящей клетки замыкает ровно
    # один цикл; ищем в дереве путь между её концами.
    start = entering[0]
    target = rows + entering[1]
    adjacency: list[list[tuple[int, tuple[int, int]]]] = [
        [] for _ in range(rows + columns)
    ]
    for row, column in zip(*np.where(basic)):
        row = int(row)
        column = int(column)
        adjacency[row].append((rows + column, (row, column)))
        adjacency[rows + column].append((row, (row, column)))

    parents: dict[int, tuple[int, tuple[int, int]] | None] = {start: None}
    pending = deque([start])
    while pending and target not in parents:
        vertex = pending.popleft()
        for neighbor, cell in adjacency[vertex]:
            if neighbor not in parents:
                parents[neighbor] = (vertex, cell)
                pending.append(neighbor)

    path: list[tuple[int, int]] = []
    vertex = target
    while vertex != start:
        previous, cell = parents[vertex]  # type: ignore[misc]
        path.append(cell)
        vertex = previous
    path.reverse()
    # Входящая клетка получает знак «+», далее знаки на цикле чередуются.
    return [entering, *reversed(path)]


def _select_entering_cell(
    reduced_costs: np.ndarray, basic: np.ndarray
) -> tuple[int, int] | None:
    """Choose the free cell with the most negative reduced cost."""
    # В минимизационной задаче отрицательная оценка показывает, что включение
    # клетки позволяет уменьшить стоимость. Выбираем наиболее отрицательную;
    # если таких клеток нет, текущий допустимый план оптимален.
    candidates = np.where(~basic, reduced_costs, np.inf)
    row, column = np.unravel_index(np.argmin(candidates), candidates.shape)
    if candidates[row, column] >= -_TOLERANCE:
        return None
    return int(row), int(column)


def _redistribute(
    allocations: np.ndarray,
    basic: np.ndarray,
    costs: np.ndarray,
    cycle: list[tuple[int, int]],
) -> float:
    """Adjust shipments around the cycle and update the degenerate basis."""
    minus_cells = cycle[1::2]
    # Перевозка в клетках «−» уменьшается; чтобы ни одна не стала отрицательной,
    # величину шага θ выбираем равной минимальной перевозке среди этих клеток.
    amount = min(allocations[row, column] for row, column in minus_cells)

    # Прибавляем θ в клетках «+» и вычитаем θ в клетках «−». Суммы в каждой
    # строке и колонке сохраняются, а стоимость плана меняется по оценке цикла.
    for index, (row, column) in enumerate(cycle):
        allocations[row, column] += amount if index % 2 == 0 else -amount
        if np.isclose(allocations[row, column], 0, atol=_TOLERANCE):
            allocations[row, column] = 0

    leaving_candidates = [
        cell
        for cell in minus_cells
        if allocations[cell] == 0
    ]
    # Хотя несколько клеток «−» могут обнулиться одновременно, из цикла
    # достаточно удалить одну. Выбор клетки с наибольшим тарифом задаёт
    # детерминированное правило разрешения вырожденного шага.
    leaving = max(leaving_candidates, key=lambda cell: costs[cell])
    basic[cycle[0]] = True
    basic[leaving] = False
    return float(amount)


def _display_table(
    problem: _ClosedProblem,
    allocations: np.ndarray,
    title: str,
    report: StepReport,
    potentials: _PotentialStep | None = None,
    selected_cycle: list[tuple[int, int]] | None = None,
) -> None:
    """Display supplies, demands, shipments, tariffs and optional reduced costs."""
    headers = [
        f"{name}={_format_number(problem.demand[index])}"
        for index, name in enumerate(problem.consumer_names)
    ]
    if potentials is not None:
        headers.append("u[i]")
    row_labels = [
        f"{name}={_format_number(problem.supply[index])}"
        for index, name in enumerate(problem.supplier_names)
    ]
    plain_cells = []
    for row in range(problem.costs.shape[0]):
        cells = []
        for column in range(problem.costs.shape[1]):
            shipment_value = allocations[row, column]
            shipment = (
                "-"
                if np.isclose(shipment_value, 0, atol=_TOLERANCE)
                else _format_number(shipment_value)
            )
            cell = (
                f"{shipment} "
                f"({_format_number(problem.costs[row, column])})"
            )
            if potentials is not None:
                cell += f" {_format_number(potentials.reduced_costs[row, column])}"
            cells.append(cell)
        plain_cells.append(cells)

    if potentials is not None:
        for row, cells in enumerate(plain_cells):
            cells.append(_format_number(potentials.row_potentials[row]))

    widths = [
        max(
            len(headers[column]),
            *(
                len(plain_cells[row][column])
                for row in range(problem.costs.shape[0])
            ),
        )
        for column in range(len(headers))
    ]
    if potentials is not None:
        widths[-1] = max(widths[-1], *(len(value) for value in headers[-1:]))
    label_width = max(len("Запасы"), *(len(label) for label in row_labels))

    def format_row(
        label: str, values: list[str], row_index: int | None = None
    ) -> str:
        padded_values = []
        for index, value in enumerate(values):
            visible_value = _ANSI_ESCAPE.sub("", value)
            padding = " " * (widths[index] - len(visible_value))
            is_selected_cell = (
                selected_cycle is not None
                and row_index is not None
                and (row_index, index) in selected_cycle
            )
            if is_selected_cell:
                padded_values.append(
                    value
                    + f"{_SELECTED_BACKGROUND}{_SELECTED_FOREGROUND}{padding}{_RESET}"
                )
            else:
                padded_values.append(value + padding)
        return f"| {label:<{label_width}} | " + " | ".join(padded_values) + " |"

    separator = (
        "+"
        + "-" * (label_width + 2)
        + "+"
        + "+".join("-" * (width + 2) for width in widths)
        + "+"
    )
    report.add(f"\n{title}")
    report.add(separator)
    report.add(format_row("", headers))
    report.add(separator)
    for row, label in enumerate(row_labels):
        cells = []
        for column in range(problem.costs.shape[1]):
            shipment_value = allocations[row, column]
            shipment = (
                "-"
                if np.isclose(shipment_value, 0, atol=_TOLERANCE)
                else _format_number(shipment_value)
            )
            tariff = _format_number(problem.costs[row, column])
            is_selected = selected_cycle is not None and (row, column) in selected_cycle
            if is_selected:
                formatted = (
                    f"{_SELECTED_BACKGROUND}{_SELECTED_FOREGROUND}{shipment} "
                    f"{_SELECTED_BACKGROUND}{_SELECTED_TARIFF}({tariff})"
                )
            else:
                formatted = f"{shipment} {_GRAY}({tariff}){_RESET}"
            if potentials is not None:
                reduced_cost = _format_number(potentials.reduced_costs[row, column])
                if is_selected:
                    formatted += f" {_SELECTED_BACKGROUND}{_MAROON}{reduced_cost}"
                else:
                    formatted += f" {_MAROON}{reduced_cost}{_RESET}"
            if is_selected:
                formatted += _RESET
            cells.append(formatted)
        if potentials is not None:
            cells.append(plain_cells[row][-1])
        report.add(format_row(label, cells, row))
    if potentials is not None:
        potential_row = [
            *(_format_number(value) for value in potentials.column_potentials),
            "",
        ]
        report.add(separator)
        report.add(format_row("v[j]", potential_row))
    report.add(separator)


def _display_potential_step(
    iteration: int,
    problem: _ClosedProblem,
    allocations: np.ndarray,
    basic: np.ndarray,
    potentials: _PotentialStep,
    entering: tuple[int, int] | None,
    report: StepReport,
    cycle: list[tuple[int, int]] | None = None,
) -> None:
    _display_table(
        problem,
        allocations,
        f"Шаг {iteration}. Метод потенциалов",
        report,
        potentials,
        cycle,
    )
    if entering is None:
        report.add("Все оценки Δc[i,j] неотрицательны: план оптимален.")
    else:
        report.add(
            f"Входящая клетка: ({entering[0] + 1}, {entering[1] + 1}), "
            f"оценка Δc={_format_number(potentials.reduced_costs[entering])}."
        )
        if cycle is not None:
            signed_cycle = [
                f"{'+' if index % 2 == 0 else '-'}({row + 1},{column + 1})"
                for index, (row, column) in enumerate(cycle)
            ]
            report.add(
                f"Цикл перераспределения: {' '.join(signed_cycle)}"
            )


def _display_solution(
    problem: _ClosedProblem, allocations: np.ndarray, report: StepReport
) -> None:
    total_cost = float(np.sum(allocations * problem.costs))
    report.add("\nОптимальный план перевозок:")
    report.add(allocations)
    report.add(f"Минимальная стоимость перевозок: {_format_number(total_cost)}")


def _solve_transport_impl(
    costs: npt.ArrayLike,
    supply: npt.ArrayLike,
    demand: npt.ArrayLike,
    *,
    report: StepReport,
) -> np.matrix:
    """Find a minimum-cost transportation plan using the lecture's potentials method."""
    # Подготовка: преобразуем входы к числам, балансируем задачу и строим
    # допустимый начальный план правилом северо-западного угла.
    costs, supply, demand = _validate_arguments(costs, supply, demand)
    problem = _close_problem(costs, supply, demand, report)
    allocations, basic = _northwest_corner(problem.supply, problem.demand)
    _display_table(
        problem, allocations, "Шаг 1. Метод северо-западного угла", report
    )

    # Начальный план уже выведен как шаг 1, поэтому первая проверка потенциалов
    # и последующие улучшения нумеруются начиная со шага 2.
    iteration = 2
    while True:
        # На текущем базисе восстанавливаем потенциалы и считаем оценки всех
        # свободных клеток. Они указывают, можно ли удешевить план.
        potentials = _calculate_potentials(problem.costs, basic)
        entering = _select_entering_cell(potentials.reduced_costs, basic)
        if entering is None:
            # Все оценки неотрицательны: условие оптимальности выполнено.
            _display_potential_step(
                iteration,
                problem,
                allocations,
                basic,
                potentials,
                entering,
                report,
            )
            break

        # Выбранная свободная клетка входит в базис. Добавление её к базисному
        # дереву задаёт цикл, по которому можно изменить перевозки, не нарушив
        # балансы складов и потребителей.
        cycle = _find_cycle(basic, entering)
        _display_potential_step(
            iteration,
            problem,
            allocations,
            basic,
            potentials,
            entering,
            report,
            cycle,
        )
        # Перераспределяем поток по циклу и удаляем из базиса клетку, ставшую
        # нулевой; затем на следующей итерации пересчитываем потенциалы.
        amount = _redistribute(allocations, basic, problem.costs, cycle)
        report.add(f"Величина перераспределения Δx={_format_number(amount)}.")
        iteration += 1

    # После достижения оптимальности суммируем тариф × объём каждой перевозки.
    _display_solution(problem, allocations, report)
    return np.matrix(allocations)


def solve_transport(
    costs: npt.ArrayLike, supply: npt.ArrayLike, demand: npt.ArrayLike
) -> tuple[np.matrix, str]:
    report = StepReport()
    result = _solve_transport_impl(costs, supply, demand, report=report)
    return result, report.get_text()
