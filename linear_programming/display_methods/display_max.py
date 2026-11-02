"""Вывод линейной целевой функции на максимум."""

from .display_function import display_function


def display_max(
    coefficients: object, variable_names: list[str] | None = None
) -> None:
    """Напечатать целевую функцию с направлением max."""
    print(f"z = {display_function(coefficients, variable_names)} -> max")
