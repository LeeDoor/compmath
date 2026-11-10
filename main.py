"""Интерактивное консольное меню лабораторной работы."""

import math
import sys
from collections.abc import Callable

import numpy as np

from linear_programming.display_methods.display_max import display_max
from linear_programming.display_methods.display_min import display_min
from linear_programming.math_tasks.artificial_basis import solve_artificial_basis
from linear_programming.math_tasks.canonical_form import run_canonical_algorithm
from linear_programming.math_tasks.dual_simplex import solve_dual_simplex
from linear_programming.math_tasks._stub import (
    describe_linear_problem,
    describe_transport_problem,
)
from linear_programming.math_tasks.optimality import check_optimality
from linear_programming.math_tasks.primal_dual import solve_primal_dual
from linear_programming.math_tasks.simplex import solve_simplex
from linear_programming.math_tasks.standard_form import to_standard_form
from linear_programming.math_tasks.transport import solve_transport
from linear_programming.problem import LinearProblem, create_problem


TaskResult = str | np.matrix


def _read_number_list(prompt: str) -> list[int | float]:
    while True:
        raw_values = input(prompt).strip()
        try:
            values = []
            for raw_value in raw_values.split(","):
                value = float(raw_value.strip())
                if not math.isfinite(value):
                    raise ValueError
                values.append(int(value) if value.is_integer() else value)

            if not values:
                raise ValueError

            return values

        except ValueError:
            print("Ошибка: введите конечные числа через запятую.")


def _read_matrix(prompt: str) -> list[list[int | float]]:
    while True:
        raw_rows = input(prompt).strip()
        try:
            rows = []

            for raw_row in raw_rows.split(";"):
                row = []

                for raw_value in raw_row.split(","):
                    value = float(raw_value.strip())
                    if not math.isfinite(value):
                        raise ValueError
                    row.append(int(value) if value.is_integer() else value)

                if not row:
                    raise ValueError

                rows.append(row)

            if not rows or any(len(row) != len(rows[0]) for row in rows):
                raise ValueError

            return rows

        except ValueError:
            print(
                "Ошибка: введите прямоугольную матрицу, "
                "например 1, 2; 3, 4."
            )


def _read_linear_problem() -> LinearProblem:
    while True:
        A = _read_matrix("Введите матрицу A (например 1, 2; 3, 4): ")
        b = _read_number_list("Введите правые части b (например, 4, 5): ")
        c = _read_number_list(
            "Введите коэффициенты функции c (например, 3, 2): "
        )
        sense = input(
            "Направление целевой функции (max/min): "
        ).strip().lower()

        try:
            return create_problem(A, b, c, sense)

        except ValueError as error:
            print(
                f"Некорректное условие: {error} "
                "Попробуйте ввести задачу заново."
            )


def _read_transport_problem() -> tuple[
    list[list[int | float]],
    list[int | float],
    list[int | float],
]:
    while True:
        costs = _read_matrix(
            "Введите матрицу тарифов (например 2, 3; 4, 1): "
        )
        supply = _read_number_list("Введите запасы (например, 20, 30): ")
        demand = _read_number_list(
            "Введите потребности (например, 25, 25): "
        )

        if len(costs) == len(supply) and len(costs[0]) == len(demand):
            return costs, supply, demand

        print(
            "Размеры не совпадают: число запасов должно быть равно "
            "числу строк тарифов, а число потребностей — числу столбцов. "
            "Введите условие заново."
        )


def _get_cached_input(
    cache: dict[str, object],
    key: str,
    description: str,
    read_input: Callable[[], object],
) -> object:
    if key in cache:
        while True:
            choice = input(
                f"Есть сохранённое условие ({description}). "
                "Enter/u — использовать его, n — ввести заново: "
            ).strip().lower()

            if choice in ("", "u", "use", "д", "да"):
                print("Используется сохранённое условие.")
                return cache[key]

            if choice in ("n", "new", "н", "нет"):
                break

            print(
                "Введите Enter/u для использования или n для нового ввода."
            )

    value = read_input()
    cache[key] = value
    return value


def _show_result(result: TaskResult) -> None:
    print(f"\033[92mРезультат выполнения: {result}\033[0m")


def _show_condition(
    task_name: str,
    condition: str
) -> None:
    message = f"Условие задачи «{task_name}»: {condition}"

    print(f"\033[93m{message}\033[0m")


def _run_optimality_check(problem: LinearProblem) -> TaskResult:
    if problem.sense == "max":
        display_max(problem.c)
    else:
        display_min(problem.c)

    solution = _read_number_list(
        "Введите компоненты найденного вектора решения через запятую: "
    )

    return check_optimality(problem, solution)


def main() -> None:
    input_cache: dict[str, object] = {}

    tasks: dict[int, tuple[str, str, Callable[..., TaskResult]]] = {
        1: (
            "Одновременное решение прямой и двойственной задач",
            "linear",
            lambda problem: solve_primal_dual(
                problem.A, problem.b, problem.c, problem.sense
            ),
        ),
        2: (
            "Симплекс-метод",
            "linear",
            lambda problem: solve_simplex(
                problem.A, problem.b, problem.c, problem.sense
            ),
        ),
        3: (
            "Транспортная задача",
            "transport",
            solve_transport,
        ),
        4: (
            "Двойственный симплекс-метод",
            "linear",
            lambda problem: solve_dual_simplex(
                problem.A, problem.b, problem.c, problem.sense
            ),
        ),
        5: (
            "Метод искусственного базиса",
            "linear",
            lambda problem: solve_artificial_basis(
                problem.A, problem.b, problem.c, problem.sense
            ),
        ),
        6: (
            "Проверка на оптимальность",
            "linear",
            _run_optimality_check,
        ),
        7: (
            "Приведение к канонической форме",
            "linear",
            lambda problem: run_canonical_algorithm(
                problem.A, problem.b, problem.c, problem.sense
            ),
        ),
        8: (
            "Приведение к стандартной форме",
            "linear",
            lambda problem: to_standard_form(
                problem.A, problem.b, problem.c, problem.sense
            ),
        ),
    }

    while True:
        print(
            "\nВыберите задачу:"
            "\n1. Одновременное решение прямой и двойственной задач"
            "\n2. Симплекс-метод"
            "\n3. Транспортная задача"
            "\n4. Двойственный симплекс-метод"
            "\n5. Метод искусственного базиса"
            "\n6. Проверка на оптимальность"
            "\n7. Каноническая форма"
            "\n8. Стандартная форма"
            "\n0. Выход"
        )

        try:
            choice = int(input("Введите номер задачи: "))

            if choice == 0:
                print("Выход.")
                return

            if choice not in tasks:
                print("Ошибка: такого пункта меню нет.")
                continue

            task_name, input_type, task_function = tasks[choice]

            print(f"Выбрана задача: {task_name}")

            if input_type == "linear":
                problem = _get_cached_input(
                    input_cache,
                    "linear_problem",
                    "СЛАУ и целевая функция",
                    _read_linear_problem,
                )

                assert isinstance(problem, LinearProblem)

                condition = describe_linear_problem(problem)

                _show_condition(
                    task_name,
                    condition
                )

                result = task_function(problem)

            else:
                transport_input = _get_cached_input(
                    input_cache,
                    "transport_problem",
                    "матрица тарифов, запасы и потребности",
                    _read_transport_problem,
                )

                costs, supply, demand = transport_input

                condition = describe_transport_problem(
                    costs, supply, demand
                )

                _show_condition(
                    task_name,
                    condition
                )

                result = task_function(costs, supply, demand)

        except ValueError as error:
            print(f"Ошибка ввода: {error}")
            continue

        except EOFError:
            print("\nЗавершение работы.")
            return

        _show_result(result)


if __name__ == "__main__":
    if sys.argv[1:] == ["--cli"]:
        main()
    else:
        from server import main as run_web_server

        run_web_server()
