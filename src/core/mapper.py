"""Mapping logic for assigning test points to selected ideal functions."""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from src.core.exceptions import MissingColumnError, UnknownXValueError


class DeviationThresholdMapper:
    """Assign test points to selected ideal functions using the sqrt(2) rule."""

    def __init__(
        self,
        ideal_df: pd.DataFrame,
        best_map: dict[str, str],
        max_devs: dict[str, float],
        *,
        fail_on_unknown_x: bool = False,
    ) -> None:
        """Create a mapper for one selected set of ideal functions."""
        if "x" not in ideal_df.columns:
            raise MissingColumnError("Missing ideal column: x")
        self.ideal_by_x = ideal_df.set_index("x")
        self.best_map = best_map
        self.max_devs = max_devs
        self.fail_on_unknown_x = fail_on_unknown_x

    def assign(self, x: float, y: float) -> Optional[tuple[str, float]]:
        """Assign a test point to the closest acceptable selected ideal function."""
        return assign_test_point(
            x,
            y,
            self.ideal_by_x,
            self.best_map,
            self.max_devs,
            fail_on_unknown_x=self.fail_on_unknown_x,
        )

    def assign_all(self, test_df: pd.DataFrame) -> list[dict[str, float | str | None]]:
        """Assign all rows from a test DataFrame and return database-ready records."""
        missing = [col for col in ("x", "y") if col not in test_df.columns]
        if missing:
            raise MissingColumnError(f"Missing test columns: {', '.join(missing)}")

        results: list[dict[str, float | str | None]] = []
        for _, row in test_df.iterrows():
            x = float(row["x"])
            y = float(row["y"])
            assignment = self.assign(x, y)

            if assignment is None:
                results.append({"x": x, "y": y, "delta_y": None, "ideal_func_no": None})
            else:
                ideal_col, delta = assignment
                results.append({"x": x, "y": y, "delta_y": float(delta), "ideal_func_no": ideal_col})

        return results


def assign_test_point(
    x: float,
    y: float,
    ideal_by_x: pd.DataFrame,
    best_map: dict[str, str],
    max_devs: dict[str, float],
    *,
    fail_on_unknown_x: bool = False,
) -> Optional[tuple[str, float]]:
    """Assign a test point (x, y) to the best matching ideal function.

    A test point is assignable to an ideal function if the absolute deviation
    from the ideal function at the same x does not exceed:

        |y_test - y_ideal(x)| <= max_dev(train, ideal) * sqrt(2)

    where max_dev(train, ideal) is the maximum absolute deviation observed
    between the training function and the selected ideal function.

    Among all assignable ideal functions, the one with the smallest deviation
    is chosen.

    Args:
        x: X-value of the test point.
        y: Y-value of the test point.
        ideal_by_x: Ideal DataFrame indexed by "x" (created via ideal_df.set_index("x")).
        best_map: Mapping of training columns to selected ideal columns,
            e.g. {"y1": "y13", "y2": "y24", ...}.
        max_devs: Mapping of training columns to max deviation values.
        fail_on_unknown_x: Raise a domain exception instead of returning None
            when x is not present in the ideal-function table.

    Returns:
        (ideal_col, delta_y) if the point can be assigned, otherwise None.
    """
    # If x is not available in the ideal functions table, no assignment is possible.
    if x not in ideal_by_x.index:
        if fail_on_unknown_x:
            raise UnknownXValueError(f"Test x-value {x} is not available in ideal functions")
        return None

    best_candidate: Optional[tuple[str, float]] = None

    for train_col, ideal_col in best_map.items():
        if ideal_col not in ideal_by_x.columns:
            raise MissingColumnError(f"Ideal column '{ideal_col}' not found in ideal data")
        if train_col not in max_devs:
            raise MissingColumnError(f"Missing max deviation for training column '{train_col}'")

        ideal_y = float(ideal_by_x.loc[x, ideal_col])
        delta = abs(y - ideal_y)

        limit = max_devs[train_col] * math.sqrt(2)

        if delta <= limit:
            if best_candidate is None or delta < best_candidate[1]:
                best_candidate = (ideal_col, float(delta))

    return best_candidate

