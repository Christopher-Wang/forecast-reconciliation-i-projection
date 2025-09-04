from enum import Enum
from functools import total_ordering
from pathlib import Path
from typing import T

import numpy as np


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


@total_ordering
class ListableEnum(Enum):

    def _generate_next_value_(name, start: ..., count: int, last_values: ...) -> str:
        """ Generate the next value when not given. """
        return name

    @classmethod
    def list(cls: T) -> list[T]:
        return list(map(lambda c: c.value, cls))

    @classmethod
    def _rev_dict(cls) -> dict[str, ...]:
        if not hasattr(cls, "_lookup"):
            # Create the reverse lookup only once per class
            cls._lookup = {member.value: member for member in cls}
        return cls._lookup

    @classmethod
    def get_enum_element(cls: T, value: str) -> T:
        if value not in cls._rev_dict():
            msg = f'Element with value "{value}" not in enum class {cls}.\nOnly has:\n\t- '
            msg += "\n\t- ".join(cls._rev_dict())
            raise ValueError(msg)
        return cls._rev_dict()[value]

    def __eq__(self, other: ...) -> bool:
        if isinstance(other, ListableEnum):
            return self.value == other.value
        return NotImplemented

    def __lt__(self, other: ...) -> bool:
        if isinstance(other, ListableEnum):
            return self.value < other.value  # lexicographic ordering
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)



def np_date_to_str(date: np.datetime64 | None) -> str:
    if date is None:
        return ""
    main, ns = date.astype(str).split('.')
    main = main.replace('T', '_').replace(':', '-')
    if ns != ("0" * len(ns)):
        main += f"_{ns}"
    return main


def get_cross_val_dates(dates: np.ndarray[np.datetime64], n_cross_val: int, horizon: int) -> tuple[list[
    tuple[int, np.datetime64 | None]], list[tuple[int, np.datetime64 | None]], list[tuple[int, np.datetime64 | None]]]:
    """
    Get cross validation dates for a time series with time steps dates.

    Args:
        dates: array of dates
        n_cross_val: number of expanding cross validation segments
        horizon: prediction period (which also conditions the validation period):
                 - Validation context: [0, T-h[
                 - Validation test: [T - h, T[
                 - Training: [0, T[
                 - Test: [T, T + h[

    Returns:
        list of (validation_end, train_end, test_end), where each element is made of the index and corresponding date
    """
    dates = np.sort(np.unique(dates))
    validation_context_cutoff_dates: list[tuple[int, np.datetime64 | None]] = []
    training_cutoff_dates: list[tuple[int, np.datetime64 | None]] = []
    test_cutoff_dates: list[tuple[int, np.datetime64 | None]] = []
    last_test_ind = len(dates)  # Excluded
    last_test_date = None
    for _ in range(n_cross_val):
        test_cutoff_dates.append((last_test_ind, last_test_date))
        training_cutoff_ind = last_test_ind - horizon
        training_cutoff_dates.append((training_cutoff_ind, dates[training_cutoff_ind]))
        validation_cutoff_ind = training_cutoff_ind - horizon
        validation_context_cutoff_dates.append((validation_cutoff_ind, dates[validation_cutoff_ind]))
        last_test_ind -= horizon
        last_test_date = dates[last_test_ind]

    return validation_context_cutoff_dates, training_cutoff_dates, test_cutoff_dates
