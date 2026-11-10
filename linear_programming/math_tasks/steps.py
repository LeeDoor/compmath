"""Step-report buffer shared by the mathematical algorithms."""


class StepReport:
    def __init__(self) -> None:
        self._parts: list[str] = []

    def add(self, text: object = "", *, end: str = "\n") -> None:
        self._parts.append(f"{text}{end}")

    def get_text(self) -> str:
        return "".join(self._parts).strip()
