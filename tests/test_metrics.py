import pandas as pd
import pytest

from src.core.metrics import max_abs_deviation, sse


def test_sse_simple_example() -> None:
    a = pd.Series([1, 2, 3])
    b = pd.Series([4, 6, 8])
    # (1-4)^2 + (2-6)^2 + (3-8)^2 = 9 + 16 + 25 = 50
    assert sse(a, b) == 50.0


def test_sse_length_mismatch_raises() -> None:
    a = pd.Series([1, 2])
    b = pd.Series([1, 2, 3])
    with pytest.raises(ValueError):
        sse(a, b)


def test_max_abs_deviation_simple() -> None:
    a = pd.Series([1, 2, 3])
    b = pd.Series([2, 4, 1])
    # abs diffs: [1, 2, 2] => max = 2
    assert max_abs_deviation(a, b) == 2.0


def test_max_abs_deviation_length_mismatch_raises() -> None:
    a = pd.Series([1, 2])
    b = pd.Series([1, 2, 3])
    with pytest.raises(ValueError):
        max_abs_deviation(a, b)
