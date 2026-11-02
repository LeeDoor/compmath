"""Приведение задачи к канонической форме."""

import numpy as np
import numpy.typing as npt

from .problem import LinearProblem, Sense


def to_canonical_form(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    sense: Sense = "max",
) -> LinearProblem:
    """Преобразовать задачу в каноническую форму."""
    raise NotImplementedError(
        "Приведение к канонической форме оставлено для самостоятельной реализации."
    )
