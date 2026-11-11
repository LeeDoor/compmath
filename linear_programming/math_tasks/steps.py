"""Step-report buffer shared by the mathematical algorithms."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FormattedSystem:
    equations: tuple[str, ...]
    html: str


@dataclass(frozen=True)
class FormattedFunction:
    expression: str
    html: str


@dataclass(frozen=True)
class SimplexTableStyle:
    pivot_column: str | None
    pivot_row: str | None


class ReportText(str):
    formatted_systems: tuple[FormattedSystem, ...]
    formatted_functions: tuple[FormattedFunction, ...]
    simplex_table_styles: tuple[SimplexTableStyle, ...]

    def __new__(
        cls,
        text: str,
        formatted_systems: tuple[FormattedSystem, ...] = (),
        formatted_functions: tuple[FormattedFunction, ...] = (),
        simplex_table_styles: tuple[SimplexTableStyle, ...] = (),
    ) -> "ReportText":
        report = super().__new__(cls, text)
        report.formatted_systems = formatted_systems
        report.formatted_functions = formatted_functions
        report.simplex_table_styles = simplex_table_styles
        return report


class StepReport:
    def __init__(self) -> None:
        self._parts: list[str] = []
        self._formatted_systems: list[FormattedSystem] = []
        self._formatted_functions: list[FormattedFunction] = []
        self._simplex_table_styles: list[SimplexTableStyle] = []

    def add(self, text: object = "", *, end: str = "\n") -> None:
        self._parts.append(f"{text}{end}")

    def add_formatted_system(
        self,
        equations: list[str],
        formatted_html: str,
        *,
        heading: str | None = None,
    ) -> None:
        if heading:
            self.add(f"\n{heading}")
        for equation in equations:
            self.add(equation)
        if equations:
            self._formatted_systems.append(
                FormattedSystem(tuple(equations), formatted_html)
            )

    def add_formatted_function(
        self,
        expression: str,
        formatted_html: str,
        *,
        prefix: str = "",
    ) -> None:
        self.add(f"{prefix}{expression}")
        self._formatted_functions.append(
            FormattedFunction(expression, formatted_html)
        )

    def add_simplex_table_style(
        self,
        pivot_column: str | None = None,
        pivot_row: str | None = None,
    ) -> None:
        self._simplex_table_styles.append(
            SimplexTableStyle(pivot_column, pivot_row)
        )

    def get_text(self) -> ReportText:
        return ReportText(
            "".join(self._parts).strip(),
            tuple(self._formatted_systems),
            tuple(self._formatted_functions),
            tuple(self._simplex_table_styles),
        )
