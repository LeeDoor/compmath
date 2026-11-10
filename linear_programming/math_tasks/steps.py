"""Step-report buffer shared by the mathematical algorithms."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FormattedSystem:
    equations: tuple[str, ...]
    html: str


class ReportText(str):
    formatted_systems: tuple[FormattedSystem, ...]

    def __new__(
        cls, text: str, formatted_systems: tuple[FormattedSystem, ...] = ()
    ) -> "ReportText":
        report = super().__new__(cls, text)
        report.formatted_systems = formatted_systems
        return report


class StepReport:
    def __init__(self) -> None:
        self._parts: list[str] = []
        self._formatted_systems: list[FormattedSystem] = []

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

    def get_text(self) -> ReportText:
        return ReportText(
            "".join(self._parts).strip(),
            tuple(self._formatted_systems),
        )
