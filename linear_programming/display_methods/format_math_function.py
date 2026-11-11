"""HTML formatter for mathematical functions."""

from .format_system_of_equations import format_math_expression


def format_math_function(function: str) -> str:
    """Render a function expression as a styled mathematical HTML block."""
    return (
        '<div class="math-function-scroll"><div class="math-function" '
        'role="math">'
        f"{format_math_expression(function)}"
        "</div></div>"
    )
