import math
import pandas as pd

from src.core.mapper import assign_test_point


def test_assign_test_point_returns_none_if_x_missing() -> None:
    ideal_df = pd.DataFrame({"x": [0.0], "y13": [1.0]})
    ideal_by_x = ideal_df.set_index("x")
    res = assign_test_point(
        x=1.0, y=1.0, ideal_by_x=ideal_by_x,
        best_map={"y1": "y13"},
        max_devs={"y1": 0.5},
    )
    assert res is None


def test_assign_test_point_assigns_when_within_threshold() -> None:
    ideal_df = pd.DataFrame({"x": [0.0], "y13": [10.0]})
    ideal_by_x = ideal_df.set_index("x")

    # max_dev = 0.5 => limit = 0.5*sqrt(2) ≈ 0.707
    res = assign_test_point(
        x=0.0,
        y=10.3,  # delta=0.3 within limit
        ideal_by_x=ideal_by_x,
        best_map={"y1": "y13"},
        max_devs={"y1": 0.5},
    )
    assert res is not None
    ideal_col, delta = res
    assert ideal_col == "y13"
    assert abs(delta - 0.3) < 1e-9


def test_assign_test_point_none_when_outside_threshold() -> None:
    ideal_df = pd.DataFrame({"x": [0.0], "y13": [10.0]})
    ideal_by_x = ideal_df.set_index("x")

    res = assign_test_point(
        x=0.0,
        y=11.0,  # delta=1.0 outside 0.707
        ideal_by_x=ideal_by_x,
        best_map={"y1": "y13"},
        max_devs={"y1": 0.5},
    )
    assert res is None
