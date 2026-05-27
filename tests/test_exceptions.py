import pandas as pd
import pytest

from src.core.exceptions import MissingColumnError, UnknownXValueError
from src.core.mapper import DeviationThresholdMapper, assign_test_point
from src.core.selector import IdealFunctionSelector
from src.io.csv_loader import _validate_columns


def test_selector_raises_user_defined_exception_for_missing_column() -> None:
    train_df = pd.DataFrame({"x": [0.0], "y1": [1.0], "y2": [2.0], "y3": [3.0]})
    ideal_df = pd.DataFrame({"x": [0.0], **{f"y{i}": [0.0] for i in range(1, 51)}})

    with pytest.raises(MissingColumnError):
        IdealFunctionSelector(train_df, ideal_df)


def test_csv_validation_raises_user_defined_exception() -> None:
    df = pd.DataFrame({"x": [0.0], "y1": [1.0]})

    with pytest.raises(MissingColumnError):
        _validate_columns(df, ["x", "y1", "y2"], "test.csv")


def test_mapper_can_raise_user_defined_exception_for_unknown_x() -> None:
    ideal_df = pd.DataFrame({"x": [0.0], "y13": [1.0]})
    mapper = DeviationThresholdMapper(
        ideal_df,
        best_map={"y1": "y13"},
        max_devs={"y1": 0.5},
        fail_on_unknown_x=True,
    )

    with pytest.raises(UnknownXValueError):
        mapper.assign(1.0, 1.0)


def test_function_mapper_keeps_backward_compatible_none_for_unknown_x() -> None:
    ideal_df = pd.DataFrame({"x": [0.0], "y13": [1.0]})

    result = assign_test_point(
        1.0,
        1.0,
        ideal_df.set_index("x"),
        best_map={"y1": "y13"},
        max_devs={"y1": 0.5},
    )

    assert result is None
