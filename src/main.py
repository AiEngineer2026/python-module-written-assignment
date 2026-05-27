"""Command-line entry point for the ideal-function assignment pipeline."""

from __future__ import annotations

import argparse

from src.core.pipeline import AssignmentPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Run the ideal-function mapping assignment.")
    parser.add_argument(
        "--keep-results",
        action="store_true",
        help="Append results instead of rebuilding the test_results table.",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Run computations without generating the Bokeh HTML visualization.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full project pipeline and print a reproducible summary."""
    args = parse_args()
    pipeline = AssignmentPipeline()
    summary = pipeline.run(
        rebuild_results=not args.keep_results,
        create_visualization=not args.skip_visualization,
    )

    print("DataFrames:")
    print("train_df:", summary["train_shape"])
    print("ideal_df:", summary["ideal_shape"])
    print("test_df:", summary["test_shape"])
    print("\nBest mapping:", summary["best_map"])
    print("Max deviations:", summary["max_devs"])
    print("\nAssignment results:")
    print("Saved rows:", summary["result_rows"])
    print("Assigned:", summary["assigned"])
    print("Unassigned:", summary["unassigned"])
    if summary["visualization_path"] is not None:
        print("Visualization:", summary["visualization_path"])


if __name__ == "__main__":
    main()
