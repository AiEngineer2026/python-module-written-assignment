import pandas as pd

from src.config import IDEAL_CSV, TEST_CSV, TRAIN_CSV
from src.core.mapper import assign_test_point
from src.core.selector import compute_max_devs, select_best_four


def test_real_data_pipeline_mapping_and_assignment_count() -> None:
    train_df = pd.read_csv(TRAIN_CSV)
    ideal_df = pd.read_csv(IDEAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    best_map = select_best_four(train_df, ideal_df)
    assert best_map == {"y1": "y13", "y2": "y24", "y3": "y36", "y4": "y40"}

    max_devs = compute_max_devs(train_df, ideal_df, best_map)

    ideal_by_x = ideal_df.set_index("x")

    assigned = 0
    for _, r in test_df.iterrows():
        x = float(r["x"])
        y = float(r["y"])
        if assign_test_point(x, y, ideal_by_x, best_map, max_devs) is not None:
            assigned += 1

    assert assigned == 34
