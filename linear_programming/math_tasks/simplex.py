
"""Решение задачи линейного программирования симплекс-методом."""

import numpy as np
import numpy.typing as npt

from ..problem import Sense, create_problem


def solve_simplex(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
    *,
    return_details: bool = False,
) -> np.matrix | tuple[np.matrix, list[int], np.ndarray, np.ndarray]:
    """Решить задачу симплекс-методом с выводом всех таблиц."""

    eps = 1e-10
    max_iterations = 1000

    problem = create_problem(A, b, c, sense)

    A = np.asarray(problem.A, dtype=float)
    b = np.asarray(problem.b, dtype=float).reshape(-1)
    c = np.asarray(problem.c, dtype=float).reshape(-1)

    original_c = c.copy()

    m, n = A.shape

    if np.any(b < -eps):
        raise ValueError(
            "Для данного варианта симплекс-метода все b должны быть >= 0."
        )

    b[np.abs(b) < eps] = 0

    # Преобразование задачи минимизации в максимизацию
    work_c = c.copy()

    if sense == "min":
        work_c = -work_c

    # Добавление дополнительных (базисных) переменных
    tableau = np.zeros((m + 1, n + m + 1))

    tableau[:m, :n] = A
    tableau[:m, n:n + m] = np.eye(m)
    tableau[:m, -1] = b

    # Начальный базис: x(n+1), ..., x(n+m)
    basis = list(range(n, n + m))

    # Строка оценок Δ
    tableau[-1, :n] = -work_c

    def variable_name(index):
        return f"x{index + 1}"

    def print_table(number):
        print(f"\n{'=' * 65}")
        print(f"СИМПЛЕКСНАЯ ТАБЛИЦА №{number}")
        print(f"{'=' * 65}")

        names = (
            [variable_name(i) for i in range(n)]
            + [f"x{i + 1}" for i in range(n, n + m)]
            + ["b"]
        )

        header = ["Базис"] + names

        widths = [9] + [11] * (n + m + 1)

        print("".join(
            str(value).center(width)
            for value, width in zip(header, widths)
        ))

        for i in range(m):
            row = [variable_name(basis[i])]
            row += [f"{value:.4f}" for value in tableau[i]]

            print("".join(
                str(value).center(width)
                for value, width in zip(row, widths)
            ))

        row = ["Δ"] + [f"{value:.4f}" for value in tableau[-1]]

        print("-" * sum(widths))
        print("".join(
            str(value).center(width)
            for value, width in zip(row, widths)
        ))

    def print_equations():
        print("\nСистема уравнений текущего базиса:")

        for i in range(m):
            terms = []

            for j in range(n + m):
                if j == basis[i]:
                    continue

                coefficient = tableau[i, j]

                if abs(coefficient) < eps:
                    continue

                terms.append(
                    f"({coefficient:.4f})*{variable_name(j)}"
                )

            equation = f"{variable_name(basis[i])} = {tableau[i, -1]:.4f}"

            if terms:
                equation += " - " + " - ".join(terms)

            print(equation)

    print("\nИСХОДНАЯ ЗАДАЧА")

    function_sign = "max" if sense == "max" else "min"

    print("Целевая функция:")
    print(
        "F = "
        + " + ".join(
            f"({original_c[j]:g})*x{j + 1}"
            for j in range(n)
        )
        + f" -> {function_sign}"
    )

    print("\nОграничения:")

    for i in range(m):
        terms = " + ".join(
            f"({A[i, j]:g})*x{j + 1}"
            for j in range(n)
        )
        print(f"{terms} <= {b[i]:g}")

    print("\nДобавляем базисные переменные:")

    for i in range(m):
        terms = " + ".join(
            f"({A[i, j]:g})*x{j + 1}"
            for j in range(n)
        )
        print(f"{terms} + x{n + i + 1} = {b[i]:g}")

    iteration = 0

    while iteration < max_iterations:

        iteration += 1

        print(f"\n\nИТЕРАЦИЯ №{iteration}")

        print_table(iteration)
        print_equations()

        # Ищем отрицательные оценки Δ
        candidates = [
            j for j in range(n + m)
            if tableau[-1, j] < -eps
        ]

        # Все оценки неотрицательны — оптимум найден
        if not candidates:
            print("\nВсе оценки Δ >= 0.")
            print("Получено оптимальное решение.")
            break

        # Выбираем столбец с наиболее отрицательной оценкой
        entering = min(
            candidates,
            key=lambda j: tableau[-1, j]
        )

        print(
            f"\nРазрешающий столбец: {variable_name(entering)}"
        )
        print(
            f"Оценка Δ = {tableau[-1, entering]:.4f}"
        )

        # Вычисляем отношения b_i / a_ij
        ratios = []

        print("\nВычисление отношений b_i / a_ij:")

        for i in range(m):
            coefficient = tableau[i, entering]

            if coefficient > eps:
                ratio = tableau[i, -1] / coefficient
                ratios.append((ratio, i))

                print(
                    f"Строка {variable_name(basis[i])}: "
                    f"{tableau[i, -1]:.4f} / "
                    f"{coefficient:.4f} = {ratio:.4f}"
                )
            else:
                print(
                    f"Строка {variable_name(basis[i])}: "
                    "не подходит (коэффициент <= 0)"
                )

        if not ratios:
            raise ValueError(
                "Целевая функция не ограничена на допустимой области."
            )

        # Минимальное положительное отношение
        _, leaving = min(ratios)

        pivot = tableau[leaving, entering]

        print(
            f"\nРазрешающая строка: {variable_name(basis[leaving])}"
        )
        print(f"Разрешающий элемент: {pivot:.4f}")

        print(
            f"Из базиса выходит {variable_name(basis[leaving])}, "
            f"в базис входит {variable_name(entering)}."
        )

        # Делим разрешающую строку на разрешающий элемент
        tableau[leaving, :] /= pivot

        # Обнуляем остальные элементы разрешающего столбца
        for i in range(m + 1):

            if i == leaving:
                continue

            factor = tableau[i, entering]
            tableau[i, :] -= factor * tableau[leaving, :]

        # Обновляем базис
        basis[leaving] = entering

        # Убираем погрешности вычислений
        tableau[np.abs(tableau) < eps] = 0

        print("\nТаблица после пересчёта:")
        print_table(iteration + 1)
        print_equations()

    else:
        raise ValueError(
            "Превышено максимальное количество итераций."
        )

    # Получаем полный оптимальный вектор
    def get_solution(current_tableau, current_basis):
        result = np.zeros(n + m)

        for i, variable in enumerate(current_basis):
            result[variable] = current_tableau[i, -1]

        return result

    solution = get_solution(tableau, basis)

    # Целевая функция зависит только от исходных переменных
    objective_value = float(np.dot(original_c, solution[:n]))

    # Ищем альтернативное оптимальное решение
    alternate_solution = None
    zero_delta_found = False

    for j in range(n + m):

        # Базисные переменные не проверяем
        if j in basis:
            continue

        # Ищем небазисные переменные с нулевой оценкой
        if abs(tableau[-1, j]) >= eps:
            continue

        zero_delta_found = True

        # Вычисляем отношения для дополнительного перехода
        ratios = []

        for i in range(m):
            coefficient = tableau[i, j]

            if coefficient > eps:
                ratio = tableau[i, -1] / coefficient
                ratios.append((ratio, i))

        if ratios:
            # Выполняем дополнительный симплекс-переход
            _, leaving = min(ratios)

            alt_tableau = tableau.copy()
            alt_basis = basis.copy()

            pivot = alt_tableau[leaving, j]

            alt_tableau[leaving, :] /= pivot

            for i in range(m + 1):
                if i == leaving:
                    continue

                factor = alt_tableau[i, j]
                alt_tableau[i, :] -= (
                        factor * alt_tableau[leaving, :]
                )

            alt_basis[leaving] = j

            alt_tableau[
                np.abs(alt_tableau) < eps
                ] = 0

            candidate = get_solution(alt_tableau, alt_basis)

        else:
            # Если в столбце нет положительных элементов,
            # можно увеличить эту переменную без изменения F.
            candidate = solution.copy()
            candidate[j] = 1

            for i, variable in enumerate(basis):
                candidate[variable] -= tableau[i, j]

        # Проверяем, получилось ли другое решение
        if (
                np.all(candidate >= -eps)
                and not np.allclose(candidate, solution, atol=eps)
        ):
            alternate_solution = candidate
            break

    print("\n\nИТОГОВЫЙ РЕЗУЛЬТАТ")
    print("-" * 40)

    solution_text = ", ".join(
        f"{value:g}" for value in solution
    )

    print(f"x* = ({solution_text})")
    print(
        f"F{'max' if sense == 'max' else 'min'} = "
        f"{objective_value:g}"
    )

    if alternate_solution is not None:
        alternate_value = float(
            np.dot(original_c, alternate_solution[:n])
        )

        alternate_text = ", ".join(
            f"{value:g}" for value in alternate_solution
        )

        print("\nНайдено альтернативное оптимальное решение:")
        print(f"x** = ({alternate_text})")
        print(f"F = {alternate_value:g}")

    elif zero_delta_found:
        print(
            "\nОбнаружена нулевая оценка Δ, "
            "но дополнительный переход не дал другого вектора."
        )

    else:
        print(
            "\nАльтернативных оптимальных решений "
            "по нулевым оценкам Δ не обнаружено."
        )

    solution_matrix = np.matrix(solution[:n].reshape(1, -1))
    if return_details:
        augmented_A = np.column_stack([A, np.eye(m)])
        return solution_matrix, basis, tableau[-1, :n + m].copy(), augmented_A
    return np.matrix(solution.reshape(1, -1))