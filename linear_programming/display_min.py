"""Вывод линейной целевой функции на минимум."""

from .display_function import display_function


def display_min(
    coefficients: object, variable_names: list[str] | None = None
) -> None:
    """Напечатать целевую функцию с направлением min."""
    print(f"z = {display_function(coefficients, variable_names)} -> min")
