from __future__ import annotations

from pathlib import Path

import pandas as pd
from bokeh.io import output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure

from src.core.selector import IdealFunctionSelector
from src.db.engine import get_session
from src.db.repository import (
    load_ideal_df,
    load_test_results_df,
    load_training_df,
)


def build_plot(train_df: pd.DataFrame, ideal_df: pd.DataFrame, results_df: pd.DataFrame):
    """Create a Bokeh plot for training/ideal functions and test point assignments.

    Deviation is visualized as a vertical segment from ideal_y to test_y for assigned points.
    """
    # Best mapping (y1..y4 -> selected ideal y*)
    best_map = IdealFunctionSelector(train_df, ideal_df).select_best_four()

    # Consistent colors for training + corresponding ideal function
    colors = {
        "y1": "#1f77b4",  # blue
        "y2": "#ff7f0e",  # orange
        "y3": "#2ca02c",  # green
        "y4": "#d62728",  # red
    }

    p = figure(
        title="Ideal Function Assignment (Training / Ideal / Test Results)",
        x_axis_label="x",
        y_axis_label="y",
        width=1100,
        height=650,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # Plot training functions (lines)
    for col in ["y1", "y2", "y3", "y4"]:
        p.line(
            train_df["x"],
            train_df[col],
            legend_label=f"training {col}",
            line_width=3,
            line_alpha=0.6,
            color=colors[col],
        )

    # Plot selected ideal functions (dashed lines)
    for train_col, ideal_col in best_map.items():
        p.line(
            ideal_df["x"],
            ideal_df[ideal_col],
            legend_label=f"ideal {ideal_col} (for {train_col})",
            line_dash="dashed",
            line_width=2,
            line_alpha=1.0,
            color=colors[train_col],
        )

    # Prepare test result data
    assigned_df = results_df[results_df["ideal_func_no"].notna()].copy()
    unassigned_df = results_df[results_df["ideal_func_no"].isna()].copy()

    # Add ideal_y for assigned points so we can draw deviation segments
    ideal_by_x = ideal_df.set_index("x")

    def _ideal_y(row) -> float | None:
        x = row["x"]
        col = row["ideal_func_no"]
        if pd.isna(col):
            return None
        if x not in ideal_by_x.index:
            return None
        return float(ideal_by_x.loc[x, col])

    if not assigned_df.empty:
        assigned_df["ideal_y"] = assigned_df.apply(_ideal_y, axis=1)

        # Some safety: drop rows where ideal_y could not be computed
        assigned_df = assigned_df[assigned_df["ideal_y"].notna()].copy()

        # segment endpoints for deviation visualization
        assigned_df["x0"] = assigned_df["x"]
        assigned_df["y0"] = assigned_df["ideal_y"]
        assigned_df["x1"] = assigned_df["x"]
        assigned_df["y1"] = assigned_df["y"]

    assigned_src = (
        ColumnDataSource(assigned_df)
        if not assigned_df.empty
        else ColumnDataSource(
            {
                "x": [],
                "y": [],
                "delta_y": [],
                "ideal_func_no": [],
                "ideal_y": [],
                "x0": [],
                "y0": [],
                "x1": [],
                "y1": [],
            }
        )
    )
    unassigned_src = ColumnDataSource(unassigned_df)

    # Deviation segments (assigned only) - not shown in legend to keep it clean
    p.segment(
        x0="x0",
        y0="y0",
        x1="x1",
        y1="y1",
        source=assigned_src,
        line_width=2,
        color="#888888",
    )

    # Test points
    assigned_renderer = p.scatter(
        x="x",
        y="y",
        source=assigned_src,
        size=7,
        color="black",
        legend_label="test assigned",
    )
    unassigned_renderer = p.scatter(
        x="x",
        y="y",
        source=unassigned_src,
        marker="x",
        size=8,
        color="#666666",
        legend_label="test unassigned",
    )

    # Hover (assigned)
    p.add_tools(
        HoverTool(
            renderers=[assigned_renderer],
            tooltips=[
                ("x", "@x"),
                ("y (test)", "@y"),
                ("ideal func", "@ideal_func_no"),
                ("ideal y", "@ideal_y"),
                ("delta_y", "@delta_y"),
            ],
        )
    )

    # Hover (unassigned)
    p.add_tools(
        HoverTool(
            renderers=[unassigned_renderer],
            tooltips=[
                ("x", "@x"),
                ("y (test)", "@y"),
                ("ideal func", "None"),
                ("delta_y", "None"),
            ],
        )
    )

    p.legend.click_policy = "hide"
    p.legend.location = "top_left"

    return p


def main() -> None:
    """Load data from the database and generate the Bokeh HTML visualization."""
    with get_session() as session:
        train_df = load_training_df(session)
        ideal_df = load_ideal_df(session)
        results_df = load_test_results_df(session)

    if results_df.empty:
        raise RuntimeError("test_results is empty. Run `python -m src.main` first.")

    plot = build_plot(train_df, ideal_df, results_df)

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "assignment_visualization.html"

    output_file(out_path.as_posix(), title="Ideal Function Assignment")
    save(plot)
    print(f"Saved Bokeh visualization to: {out_path.resolve()}")


if __name__ == "__main__":
    main()

