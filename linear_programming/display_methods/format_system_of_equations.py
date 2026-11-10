"""HTML formatters for systems of equations and mathematical expressions."""

import html
import re


def format_math_expression(expression: str) -> str:
    parts = []
    position = 0
    for match in re.finditer(r"([A-Za-z])('*)(\d+)('*)", expression):
        parts.append(html.escape(expression[position:match.start()]))
        variable, prefix_primes, index, suffix_primes = match.groups()
        parts.append(f"<var>{html.escape(variable)}</var>")
        prime_count = len(prefix_primes) + len(suffix_primes)
        if prime_count:
            primes = {1: "′", 2: "″"}.get(prime_count, "′" * prime_count)
            parts.append(f"<sup>{primes}</sup>")
        parts.append(f"<sub>{index}</sub>")
        position = match.end()
    parts.append(html.escape(expression[position:]))
    return "".join(parts)


def format_system_of_equations(equations: list[str] | tuple[str, ...]) -> str:
    """Format a linear system as accessible, styled HTML."""
    if not equations:
        return ""
    rows = "".join(
        '<div class="equation-system-row">'
        f"{format_math_expression(equation)}"
        "</div>"
        for equation in equations
    )
    return (
        '<div class="equation-system-scroll"><div class="equation-system" '
        f'role="group" aria-label="Система линейных уравнений" '
        f'style="--equation-count: {len(equations)}">'
        '<span class="equation-system-brace" aria-hidden="true">{</span>'
        f'<div class="equation-system-rows">{rows}</div>'
        "</div></div>"
    )
