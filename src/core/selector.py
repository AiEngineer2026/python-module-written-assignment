"""Selection logic for mapping training functions to ideal functions."""

from __future__ import annotations

import pandas as pd

from src.core.exceptions import MissingColumnError
from src.core.metrics import max_abs_deviation, sse


class IdealFunctionSelector:
    """Select ideal functions for training functions using the SSE criterion."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        ideal_df: pd.DataFrame,
        train_columns: tuple[str, ...] = ("y1", "y2", "y3", "y4"),
        ideal_count: int = 50,
    ) -> None:
        """Create a selector for the provided training and ideal data."""
        self.train_df = train_df
        self.ideal_df = ideal_df
        self.train_columns = train_columns
        self.ideal_columns = tuple(f"y{i}" for i in range(1, ideal_count + 1))
        self._validate_columns()

    def _validate_columns(self) -> None:
        """Validate that all required columns exist before computing scores."""
        missing_train = [col for col in ("x", *self.train_columns) if col not in self.train_df.columns]
        missing_ideal = [col for col in ("x", *self.ideal_columns) if col not in self.ideal_df.columns]

        if missing_train:
            raise MissingColumnError(f"Missing training columns: {', '.join(missing_train)}")
        if missing_ideal:
            raise MissingColumnError(f"Missing ideal columns: {', '.join(missing_ideal)}")

    def best_for_train_col(self, train_col: str) -> tuple[str, float]:
        """Find the best matching ideal function for one training function using SSE.

        Args:
            train_col: Training column name (e.g. "y1").

        Returns:
            Tuple of (ideal_column_name, sse_score), e.g. ("y13", 34.08).

        Raises:
            MissingColumnError: If train_col does not exist in train_df.
        """
        if train_col not in self.train_df.columns:
            raise MissingColumnError(f"Training column '{train_col}' not found in train_df")

        best_col = ""
        best_score = float("inf")

        for ideal_col in self.ideal_columns:
            score = sse(self.train_df[train_col], self.ideal_df[ideal_col])
            if score < best_score:
                best_score = score
                best_col = ideal_col

        return best_col, best_score

    def select_best_four(self) -> dict[str, str]:
        """Select the best ideal function for each training function."""
        mapping: dict[str, str] = {}
        for train_col in self.train_columns:
            best_col, _ = self.best_for_train_col(train_col)
            mapping[train_col] = best_col
        return mapping

    def compute_max_devs(self, best_map: dict[str, str]) -> dict[str, float]:
        """Compute maximum absolute deviations for each training-to-ideal mapping."""
        max_devs: dict[str, float] = {}

        for train_col, ideal_col in best_map.items():
            if train_col not in self.train_df.columns:
                raise MissingColumnError(f"Training column '{train_col}' not found in train_df")
            if ideal_col not in self.ideal_df.columns:
                raise MissingColumnError(f"Ideal column '{ideal_col}' not found in ideal_df")

            max_devs[train_col] = max_abs_deviation(self.train_df[train_col], self.ideal_df[ideal_col])

        return max_devs


def best_ideal_for_train_col(
    train_df: pd.DataFrame,
    ideal_df: pd.DataFrame,
    train_col: str,
) -> tuple[str, float]:
    """Find the best matching ideal function for one training function using SSE."""
    return IdealFunctionSelector(train_df, ideal_df).best_for_train_col(train_col)


def select_best_four(train_df: pd.DataFrame, ideal_df: pd.DataFrame) -> dict[str, str]:
    """Select the best ideal function for each of the four training functions.

    Returns:
        Mapping from training column to selected ideal column,
        e.g. {"y1": "y13", "y2": "y24", "y3": "y36", "y4": "y40"}.
    """
    return IdealFunctionSelector(train_df, ideal_df).select_best_four()


def compute_max_devs(
    train_df: pd.DataFrame,
    ideal_df: pd.DataFrame,
    best_map: dict[str, str],
) -> dict[str, float]:
    """Compute maximum absolute deviations for each training->ideal mapping.

    For each training column (y1..y4) mapped to an ideal column (y1..y50),
    compute max(|train - ideal|) over all x.

    Args:
        train_df: Training data DataFrame.
        ideal_df: Ideal functions DataFrame.
        best_map: Mapping like {"y1": "y13", ...}.

    Returns:
        Dict mapping training column to its max absolute deviation.
    """
    return IdealFunctionSelector(train_df, ideal_df).compute_max_devs(best_map)
