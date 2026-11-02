"""Пример запуска и отображения задачи."""

from linear_programming.artificial_basis import solve_artificial_basis
from linear_programming.canonical_form import to_canonical_form
from linear_programming.display_augmented_matrix import display_augmented_matrix
from linear_programming.display_max import display_max
from linear_programming.display_min import display_min
from linear_programming.dual_simplex import solve_dual_simplex
from linear_programming.optimality import check_optimality
from linear_programming.primal_dual import solve_primal_dual
from linear_programming.problem import create_problem
from linear_programming.simplex import solve_simplex
from linear_programming.standard_form import to_standard_form
from linear_programming.transport import solve_transport


def main() -> None:
    problem = create_problem(
        A=[[1, 1], [2, 1]],
        b=[4, 5],
        c=[3, 2],
        sense="max",
    )
    display_augmented_matrix(problem.A, problem.b, "Расширенная матрица")
    if problem.sense == "max":
        display_max(problem.c)
    else:
        display_min(problem.c)

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
        except ValueError:
            print("Ошибка: введите целый номер пункта меню.")
            continue
        except EOFError:
            print("\nЗавершение работы.")
            break

        if choice == 0:
            print("Выход.")
            break

        if choice == 6:
            print("Выбрана задача: Проверка на оптимальность")
            try:
                primal_solution = [
                    float(value.strip())
                    for value in input(
                        "Введите компоненты прямого решения через запятую: "
                    ).split(",")
                ]
            except ValueError:
                print("Ошибка: введите числовые компоненты через запятую.")
                continue
            selected_task = lambda: check_optimality(problem, primal_solution)
        else:
            tasks = {
                1: (
                    "Одновременное решение прямой и двойственной задач",
                    lambda: solve_primal_dual(
                        problem.A, problem.b, problem.c, problem.sense
                    ),
                ),
                2: (
                    "Симплекс-метод",
                    lambda: solve_simplex(
                        problem.A, problem.b, problem.c, problem.sense
                    ),
                ),
                3: (
                    "Транспортная задача",
                    lambda: solve_transport([[2, 3], [4, 1]], [20, 30], [25, 25]),
                ),
                4: (
                    "Двойственный симплекс-метод",
                    lambda: solve_dual_simplex(
                        problem.A, problem.b, problem.c, problem.sense
                    ),
                ),
                5: (
                    "Метод искусственного базиса",
                    lambda: solve_artificial_basis(
                        problem.A, problem.b, problem.c, problem.sense
                    ),
                ),
                7: (
                    "Приведение к канонической форме",
                    lambda: to_canonical_form(
                        problem.A, problem.b, problem.c, problem.sense
                    ),
                ),
                8: (
                    "Приведение к стандартной форме",
                    lambda: to_standard_form(
                        problem.A, problem.b, problem.c, problem.sense
                    ),
                ),
            }
            task = tasks.get(choice)
            if task is None:
                print("Ошибка: такого пункта меню нет.")
                continue
            print(f"Выбрана задача: {task[0]}")
            selected_task = task[1]

        try:
            result = selected_task()
        except NotImplementedError as error:
            print(error)
        else:
            print(f"Результат: {result}")


if __name__ == "__main__":
    main()
