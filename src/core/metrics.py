"""Metric functions used for model selection and test point assignment."""

from __future__ import annotations

import pandas as pd


def sse(a: pd.Series, b: pd.Series) -> float:
    """Compute the sum of squared errors (SSE) between two series.

    Args:
        a: First series.
        b: Second series.

    Returns:
        Sum of squared differences between a and b.

    Raises:
        ValueError: If the series lengths differ.
    """
    if len(a) != len(b):
        raise ValueError("Series must have the same length")

    diff = a - b
    return float((diff**2).sum())


def max_abs_deviation(a: pd.Series, b: pd.Series) -> float:
    """Compute the maximum absolute deviation between two series.

    This is defined as max(|a_i - b_i|) over all i.

    Args:
        a: First series.
        b: Second series.

    Returns:
        Maximum absolute difference between a and b.

    Raises:
        ValueError: If the series lengths differ.
    """
    if len(a) != len(b):
        raise ValueError("Series must have the same length")

    return float((a - b).abs().max())
