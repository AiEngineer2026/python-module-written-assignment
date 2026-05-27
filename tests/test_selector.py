import pandas as pd

from src.core.selector import best_ideal_for_train_col, compute_max_devs, select_best_four


def _make_ideal_df() -> pd.DataFrame:
    # x with 3 points, y1..y50 exist
    x = [0.0, 1.0, 2.0]
    data = {"x": x}
    for i in range(1, 51):
        data[f"y{i}"] = [0.0, 0.0, 0.0]
    # Define 4 ideal functions with distinct shapes
    data["y13"] = [1.0, 2.0, 3.0]
    data["y24"] = [10.0, 20.0, 30.0]
    data["y36"] = [-1.0, -2.0, -3.0]
    data["y40"] = [5.0, 5.0, 5.0]
    return pd.DataFrame(data)


def _make_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y1": [1.0, 2.0, 3.0],      # matches y13 perfectly
            "y2": [10.0, 20.0, 30.0],   # matches y24 perfectly
            "y3": [-1.0, -2.0, -3.0],   # matches y36 perfectly
            "y4": [5.0, 5.0, 5.0],      # matches y40 perfectly
        }
    )


def test_best_ideal_for_train_col_returns_expected() -> None:
    train_df = _make_train_df()
    ideal_df = _make_ideal_df()
    best_col, best_score = best_ideal_for_train_col(train_df, ideal_df, "y1")
    assert best_col == "y13"
    assert best_score == 0.0


def test_select_best_four_returns_expected_mapping() -> None:
    train_df = _make_train_df()
    ideal_df = _make_ideal_df()
    mapping = select_best_four(train_df, ideal_df)
    assert mapping == {"y1": "y13", "y2": "y24", "y3": "y36", "y4": "y40"}


def test_compute_max_devs_zero_when_perfect_match() -> None:
    train_df = _make_train_df()
    ideal_df = _make_ideal_df()
    mapping = {"y1": "y13", "y2": "y24", "y3": "y36", "y4": "y40"}
    max_devs = compute_max_devs(train_df, ideal_df, mapping)
    assert max_devs == {"y1": 0.0, "y2": 0.0, "y3": 0.0, "y4": 0.0}
