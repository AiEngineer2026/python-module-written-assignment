"""End-to-end assignment pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bokeh.io import output_file, save

from src.core.mapper import DeviationThresholdMapper
from src.core.selector import IdealFunctionSelector
from src.db.engine import get_session
from src.db.init_db import init_db
from src.db.models import IdealFunctions, TestData, TrainingData
from src.db.repository import (
    clear_test_results,
    count_rows,
    load_ideal_df,
    load_test_df,
    load_test_results_df,
    load_training_df,
    save_test_results,
)
from src.io.csv_loader import load_ideal_to_db, load_test_to_db, load_training_to_db
from src.viz.plot_bokeh import build_plot


class AssignmentPipeline:
    """Run data loading, selection, mapping, persistence, and visualization."""

    def __init__(self, output_dir: Path | str = "output") -> None:
        """Create a pipeline that writes generated artifacts to output_dir."""
        self.output_dir = Path(output_dir)

    def ensure_input_data(self) -> None:
        """Create tables and load CSV inputs into empty database tables."""
        init_db()
        with get_session() as session:
            if count_rows(session, TrainingData) == 0:
                load_training_to_db(session)
            if count_rows(session, IdealFunctions) == 0:
                load_ideal_to_db(session)
            if count_rows(session, TestData) == 0:
                load_test_to_db(session)

    def run(self, *, rebuild_results: bool = True, create_visualization: bool = True) -> dict[str, Any]:
        """Run the complete assignment workflow and return a compact summary."""
        self.ensure_input_data()

        with get_session() as session:
            train_df = load_training_df(session)
            ideal_df = load_ideal_df(session)
            test_df = load_test_df(session)

        selector = IdealFunctionSelector(train_df, ideal_df)
        best_map = selector.select_best_four()
        max_devs = selector.compute_max_devs(best_map)

        mapper = DeviationThresholdMapper(ideal_df, best_map, max_devs)
        results = mapper.assign_all(test_df)

        with get_session() as session:
            if rebuild_results:
                clear_test_results(session)
            inserted = save_test_results(session, results)
            results_df = load_test_results_df(session)

        assigned = sum(1 for row in results if row["ideal_func_no"] is not None)
        visualization_path = None
        if create_visualization:
            visualization_path = self.create_visualization(train_df, ideal_df, results_df)

        return {
            "train_shape": train_df.shape,
            "ideal_shape": ideal_df.shape,
            "test_shape": test_df.shape,
            "best_map": best_map,
            "max_devs": max_devs,
            "result_rows": inserted,
            "assigned": assigned,
            "unassigned": len(results) - assigned,
            "visualization_path": visualization_path,
        }

    def create_visualization(self, train_df, ideal_df, results_df) -> Path:
        """Create the standalone Bokeh visualization and return its path."""
        self.output_dir.mkdir(exist_ok=True)
        out_path = self.output_dir / "assignment_visualization.html"
        output_file(out_path.as_posix(), title="Ideal Function Assignment")
        save(build_plot(train_df, ideal_df, results_df))
        return out_path.resolve()
