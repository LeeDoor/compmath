"""Решение задачи методом искусственного базиса."""

import numpy as np
import numpy.typing as npt

from ..problem import Sense
from .canonical_form import make_canonical
from .simplex import ConstraintSign, VariableConstraint, _run_simplex_tableau_impl
from .steps import StepReport


def _solve_artificial_basis_impl(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
    constraint_signs: list[ConstraintSign] | None = None,
    variable_constraints: list[VariableConstraint] | None = None,
    *,
    report: StepReport,
) -> np.matrix:
    """Решить задачу методом искусственного базиса."""
    eps = 1e-10

    # 1. Приведение задачи к канонической форме.
    report.add("\nПРИВЕДЕНИЕ К КАНОНИЧЕСКОЙ ФОРМЕ")
    A, b, c, decision_count = make_canonical(
        A,
        b,
        c,
        sense,
        constraint_signs,
        variable_constraints,
    )

    A = A.copy()
    b = b.copy()
    c = c.copy()

    m, n = A.shape

    report.add("\nМатрица A:")
    report.add(A)
    report.add("\nВектор b:")
    report.add(b)
    report.add("\nЦелевая функция:")
    report.add(c)

    # 2. Делаем правые части неотрицательными.
    report.add("\nПРИВЕДЕНИЕ ПРАВЫХ ЧАСТЕЙ К НЕОТРИЦАТЕЛЬНОМУ ВИДУ")
    for i in range(m):
        if b[i] < -eps:
            A[i, :] *= -1
            b[i] *= -1
            report.add(f"Уравнение {i + 1} умножено на -1.")

    b[np.abs(b) < eps] = 0

    # 3. Добавляем искусственные переменные и формируем начальный базис.
    report.add("\nДОБАВЛЕНИЕ ИСКУССТВЕННЫХ ПЕРЕМЕННЫХ")
    total_columns = n + m
    tableau = np.zeros((m + 1, total_columns + 1))
    tableau[:m, :n] = A
    tableau[:m, n:total_columns] = np.eye(m)
    tableau[:m, -1] = b

    basis = list(range(n, total_columns))
    names = [f"x{i + 1}" for i in range(total_columns)]

    report.add("Добавлены искусственные переменные:")
    report.add(", ".join(names[n:]))

    # 4. Первая фаза: минимизируем сумму искусственных переменных,
    # что эквивалентно максимизации её отрицательного значения.
    report.add("\nЭТАП 1. ВСПОМОГАТЕЛЬНАЯ ЗАДАЧА")
    phase_one_costs = np.zeros(total_columns)
    phase_one_costs[n:] = -1

    tableau, basis, _ = _run_simplex_tableau_impl(
        tableau,
        basis,
        phase_one_costs,
        report,
        names,
        title="Первый этап",
        eps=eps,
    )

    phase_one_value = sum(
        phase_one_costs[basis[i]] * tableau[i, -1]
        for i in range(len(basis))
    )

    if phase_one_value < -eps:
        raise ValueError(
            "Исходная задача несовместна: "
            "искусственные переменные не удалось обнулить."
        )

    report.add("\nВспомогательная задача решена.")
    report.add("Сумма искусственных переменных равна нулю.")

    # 5. Удаляем искусственные переменные из базиса.
    report.add("\nУДАЛЕНИЕ ИСКУССТВЕННЫХ ПЕРЕМЕННЫХ")
    rows_to_remove = []

    for i in range(len(basis)):
        if basis[i] < n:
            continue

        entering = None
        for j in range(n):
            if j not in basis and abs(tableau[i, j]) > eps:
                entering = j
                break

        if entering is None:
            if abs(tableau[i, -1]) > eps:
                raise ValueError(
                    "Не удалось удалить искусственную переменную."
                )
            rows_to_remove.append(i)
            continue

        pivot = tableau[i, entering]
        tableau[i, :] /= pivot

        # Пересчитываем остальные строки ограничений.
        for k in range(len(basis)):
            if k == i:
                continue
            factor = tableau[k, entering]
            tableau[k, :] -= factor * tableau[i, :]

        basis[i] = entering

    # Удаляем избыточные ограничения.
    for i in reversed(rows_to_remove):
        tableau = np.delete(tableau, i, axis=0)
        basis.pop(i)

    # Удаляем столбцы искусственных переменных, сохраняя столбец b.
    tableau = np.column_stack([tableau[:, :n], tableau[:, -1]])
    names = names[:n]
    tableau[np.abs(tableau) < eps] = 0

    report.add("\nИскусственные переменные удалены.")

    # 6. Вторая фаза: решаем исходную задачу тем же симплекс-алгоритмом.
    report.add("\nЭТАП 2. РЕШЕНИЕ ИСХОДНОЙ ЗАДАЧИ")
    tableau, basis, _ = _run_simplex_tableau_impl(
        tableau,
        basis,
        c,
        report,
        names,
        title="Второй этап",
        eps=eps,
    )

    # 7. Получаем итоговое решение.
    solution = np.zeros(n)
    for i, basic in enumerate(basis):
        solution[basic] = tableau[i, -1]

    solution[np.abs(solution) < eps] = 0
    objective_value = float(np.dot(c, solution))

    # make_canonical переводит min в max, меняя знак c.
    if sense == "min":
        objective_value = -objective_value

    report.add("\nИТОГОВОЕ РЕШЕНИЕ")
    report.add("-" * 40)
    report.add(
        "x* = ("
        + ", ".join(f"{value:g}" for value in solution[:decision_count])
        + ")"
    )
    report.add(f"F{'max' if sense == 'max' else 'min'} = {objective_value:g}")

    return np.matrix(solution[:decision_count].reshape(1, -1))


def solve_artificial_basis(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
    constraint_signs: list[ConstraintSign] | None = None,
    variable_constraints: list[VariableConstraint] | None = None,
) -> tuple[np.matrix, str]:
    report = StepReport()
    result = _solve_artificial_basis_impl(
        A,
        b,
        c,
        sense,
        constraint_signs,
        variable_constraints,
        report=report,
    )
    return result, report.get_text()
