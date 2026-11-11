"""Решение задачи двойственным симплекс-методом."""

import numpy as np
import numpy.typing as npt

from ..display_methods import format_math_function
from ..display_methods import format_system_of_equations
from ..problem import Sense
from .canonical_form import make_canonical
from .simplex import ConstraintSign, VariableConstraint, append_simplex_table
from .steps import StepReport


_TOLERANCE = 1e-10
_MAX_ITERATIONS = 1000


def _solve_dual_simplex_impl(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
    constraint_signs: list[ConstraintSign] | None = None,
    variable_constraints: list[VariableConstraint] | None = None,
    *,
    report: StepReport,
) -> str:
    """Solve an LP with dual simplex when its canonical basis allows it."""
    # Сначала приводим исходную задачу к равенствам с неотрицательными
    # переменными. Для двойственного симплекса дальше нужна таблица именно
    # в каноническом представлении, а не исходные неравенства.
    (
        canonical_matrix,
        canonical_rhs,
        objective,
        decision_count,
    ) = make_canonical(
        A,
        b,
        c,
        sense,
        constraint_signs,
        variable_constraints,
    )
    constraint_count, variable_count = canonical_matrix.shape
    original_objective = np.asarray(c, dtype=float).reshape(-1)

    variable_options = (
        ["nonnegative"] * len(original_objective)
        if variable_constraints is None
        else list(variable_constraints)
    )
    variable_mapping: list[tuple[int, int | None]] = []
    names = []
    transformed_index = 0
    # Запоминаем, как восстановить исходные переменные после решения:
    # свободная x[i] заменяется разностью x'[i] - x''[i], остальные переменные
    # представлены одним неотрицательным компонентом.
    for index, option in enumerate(variable_options):
        if option == "free":
            variable_mapping.append((transformed_index, transformed_index + 1))
            names.extend((f"x'{index + 1}", f"x''{index + 1}"))
            transformed_index += 2
        else:
            variable_mapping.append((transformed_index, None))
            names.append(f"x{index + 1}")
            transformed_index += 1
    names.extend(
        f"x{index}"
        for index in range(
            len(original_objective) + 1,
            len(original_objective) + variable_count - decision_count + 1,
        )
    )

    # Ищем в каждой строке базисный столбец: он должен содержать ровно один
    # ненулевой коэффициент. В первую очередь берём добавленные slack/surplus
    # переменные; если их нет, используем подходящий столбец исходной части.
    # При отсутствии такого столбца у задачи нет готового начального базиса,
    # поэтому для его построения нужен метод искусственного базиса.
    column_support = np.abs(canonical_matrix) > _TOLERANCE
    basis = []
    for row in range(constraint_count):
        candidates = np.flatnonzero(
            column_support[row] & (column_support.sum(axis=0) == 1)
        )
        if not len(candidates):
            raise ValueError("Нет базиса. Используйте МИБ.")
        supplemental = candidates[candidates >= decision_count]
        basis.append(int(supplemental[0] if len(supplemental) else candidates[0]))

    # Масштабируем каждую базисную строку так, чтобы базисный коэффициент стал
    # равен 1. После этого базисные столбцы образуют единичную матрицу, а RHS
    # показывает текущие значения базисных переменных.
    for row, basic_variable in enumerate(basis):
        coefficient = canonical_matrix[row, basic_variable]
        canonical_matrix[row, :] /= coefficient
        canonical_rhs[row] /= coefficient

    row_count = constraint_count

    # Эти выражения описывают уже преобразованную систему, на которой будет
    # построена симплекс-таблица; форматтеры выводят их как математический HTML.
    def format_expression(coefficients: np.ndarray, variable_names: list[str]) -> str:
        terms = []
        for coefficient, name in zip(coefficients, variable_names):
            value = float(coefficient)
            if abs(value) < _TOLERANCE:
                continue
            magnitude = abs(value)
            term = name if abs(magnitude - 1) < _TOLERANCE else f"{magnitude:g}*{name}"
            if not terms:
                terms.append(f"-{term}" if value < 0 else term)
            else:
                terms.append(f" - {term}" if value < 0 else f" + {term}")
        return "".join(terms) or "0"

    report.add("\nСистема после приведения к канонической форме:")
    canonical_function = f"F = {format_expression(objective, names)} -> max"
    report.add_formatted_function(
        canonical_function,
        format_math_function(canonical_function),
    )
    equations = [
        f"{format_expression(row, names)} = {value:g}"
        for row, value in zip(canonical_matrix, canonical_rhs)
    ]
    report.add_formatted_system(
        equations,
        format_system_of_equations(equations),
    )

    # Двойственный симплекс нужен, когда текущий базис недопустим по RHS:
    # некоторые базисные переменные отрицательны. Если RHS уже неотрицателен,
    # этот метод не требуется.
    dual_candidates = np.flatnonzero(canonical_rhs < -_TOLERANCE)
    if not len(dual_candidates):
        return "Нет нужды в двойственном симплексе. Используйте обычный симплекс-метод."

    report.add(
        "После нормализации базисных переменных найдены отрицательные b; "
        "применяем двойственный симплекс-метод."
    )
    tableau = np.zeros((row_count + 1, variable_count + 1))
    tableau[:row_count, :variable_count] = canonical_matrix
    tableau[:row_count, -1] = canonical_rhs
    # Нижняя строка хранит Δ = z - c для задачи на максимум. Сначала записываем
    # -c, затем прибавляем стоимость каждой базисной переменной, умноженную на
    # её строку; в результате оценки базисных переменных обнуляются.
    tableau[-1, :variable_count] = -objective
    for row, basic_variable in enumerate(basis):
        tableau[-1, :] += (
            objective[basic_variable] * tableau[row, :]
        )
    tableau[np.abs(tableau) < _TOLERANCE] = 0

    def pivot(row: int, column: int) -> None:
        # Разрешающий элемент нормируем до 1, затем зануляем остальные элементы
        # его столбца, включая строку оценок. Так обновляются таблица и базис.
        element = tableau[row, column]
        tableau[row, :] /= element
        for other_row in range(row_count + 1):
            if other_row == row:
                continue
            factor = tableau[other_row, column]
            if abs(factor) > _TOLERANCE:
                tableau[other_row, :] -= factor * tableau[row, :]
        basis[row] = column
        tableau[np.abs(tableau) < _TOLERANCE] = 0

    for iteration in range(1, _MAX_ITERATIONS + 1):
        # На каждой итерации проверяем допустимость базисного решения (b >= 0)
        # и оптимальность по оценкам (Δ >= 0 для задачи максимизации).
        report.add(
            "Базисные переменные: "
            + ", ".join(names[index] for index in basis)
        )
        negative_rhs = np.flatnonzero(tableau[:row_count, -1] < -_TOLERANCE)
        negative_delta = np.flatnonzero(tableau[-1, :-1] < -_TOLERANCE)
        if not len(negative_rhs) and not len(negative_delta):
            # Нет ни отрицательных RHS, ни улучшающих оценок: решение допустимо
            # и оптимально, поэтому вычислительный цикл завершён.
            append_simplex_table(
                report,
                tableau,
                basis,
                names,
                f"Симплекс-таблица, итерация {iteration}",
            )
            break

        if len(negative_rhs):
            # Пока есть отрицательные базисные значения, выбираем самое
            # отрицательное RHS. Строка должна покинуть базис, иначе получить
            # допустимое решение перестановкой базиса невозможно.
            leaving = int(negative_rhs[np.argmin(tableau[negative_rhs, -1])])
            # Разрешающим может быть только отрицательный коэффициент этой
            # строки: после поворота он позволяет поднять выбранный RHS к нулю.
            candidates = [
                column
                for column in range(variable_count)
                if tableau[leaving, column] < -_TOLERANCE
            ]
            if not candidates:
                # В выбранной строке нет переменной, чей ввод мог бы устранить
                # отрицательный RHS, поэтому допустимый базис восстановить нельзя.
                append_simplex_table(
                    report,
                    tableau,
                    basis,
                    names,
                    f"Симплекс-таблица, итерация {iteration}",
                    pivot_row=leaving,
                )
                raise ValueError(
                    "Задача несовместна: в ведущей строке нет подходящего "
                    "ведущего столбца."
                )
            # Отношения Δ/a задают допустимые повороты в ведущей строке.
            # Выбираем наибольшее, как требует правило двойственного симплекса;
            # если оценки до шага были двойственно допустимы, такой выбор
            # сохраняет их неотрицательность.
            ratios = {
                column: tableau[-1, column] / tableau[leaving, column]
                for column in candidates
            }
            entering = max(candidates, key=ratios.__getitem__)
            report.add(
                f"Ведущая строка: {leaving + 1}; ведущий столбец: "
                f"{names[entering]}; Δ/a = {ratios[entering]:g}."
            )
        else:
            # Все RHS уже неотрицательны, но есть отрицательные Δ. Допустимость
            # восстановлена, поэтому продолжаем обычным симплекс-шагом:
            # вводим переменную с наиболее отрицательной оценкой.
            entering = int(negative_delta[np.argmin(tableau[-1, negative_delta])])
            candidates = [
                row
                for row in range(row_count)
                if tableau[row, entering] > _TOLERANCE
            ]
            if not candidates:
                # В разрешающем столбце нет положительных коэффициентов, значит
                # увеличить целевую функцию можно без конечного ограничения.
                append_simplex_table(
                    report,
                    tableau,
                    basis,
                    names,
                    f"Симплекс-таблица, итерация {iteration}",
                    pivot_column=entering,
                )
                raise ValueError("Целевая функция не ограничена на допустимой области.")
            # Ограничиваем шаг так, чтобы RHS всех строк остались
            # неотрицательными: минимальное b[i]/a[i,j] задаёт выходящую
            # переменную, как в обычном симплекс-методе.
            leaving = min(
                candidates,
                key=lambda row: tableau[row, -1] / tableau[row, entering],
            )
            report.add(
                "Все b неотрицательны; завершаем оптимизацию обычным "
                f"симплекс-шагом. Входит {names[entering]}."
            )

        # Показываем таблицу до преобразования с отмеченными разрешающими
        # строкой и столбцом, затем выполняем поворот и обновляем базис.
        append_simplex_table(
            report,
            tableau,
            basis,
            names,
            f"Симплекс-таблица, итерация {iteration}",
            pivot_column=entering,
            pivot_row=leaving,
        )
        pivot(leaving, entering)
    else:
        # Ограничиваем число шагов, чтобы зацикливание или численная проблема
        # не приводили к бесконечному вычислению.
        raise ValueError("Превышено максимальное количество итераций.")

    # Считываем значения переменных из финального базиса; добавленные slack /
    # surplus-переменные не входят в возвращаемый вектор исходных переменных.
    transformed_solution = np.zeros(variable_count)
    for row, variable in enumerate(basis):
        if variable < variable_count:
            transformed_solution[variable] = tableau[row, -1]
    transformed_solution = transformed_solution[:decision_count]
    # Обратная замена возвращает свободные переменные как x' - x''. Значение
    # функции считаем по исходным коэффициентам и исходному вектору решения.
    original_solution = np.asarray(
        [
            transformed_solution[first]
            if second is None
            else transformed_solution[first] - transformed_solution[second]
            for first, second in variable_mapping
        ]
    )
    objective_value = float(np.dot(original_objective, original_solution))
    return (
        "Решение исходной задачи: x* = ("
        + ", ".join(f"{value:g}" for value in original_solution)
        + f")\nЗначение целевой функции: {objective_value:g}"
    )


def solve_dual_simplex(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
    constraint_signs: list[ConstraintSign] | None = None,
    variable_constraints: list[VariableConstraint] | None = None,
) -> tuple[str, str]:
    report = StepReport()
    result = _solve_dual_simplex_impl(
        A,
        b,
        c,
        sense,
        constraint_signs,
        variable_constraints,
        report=report,
    )
    return result, report.get_text()
