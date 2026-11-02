"""Приведение задачи к стандартной форме."""

import numpy as np
import numpy.typing as npt

from .problem import LinearProblem, Sense


def to_standard_form(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
) -> LinearProblem:
    """Преобразовать задачу в стандартную форму."""
    raise NotImplementedError(
        "Приведение к стандартной форме оставлено для самостоятельной реализации."
    )
