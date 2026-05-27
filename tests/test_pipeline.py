from pathlib import Path

from src.core.pipeline import AssignmentPipeline


def test_pipeline_rebuilds_results_and_creates_visualization() -> None:
    pipeline = AssignmentPipeline(output_dir=Path("output"))

    summary = pipeline.run(rebuild_results=True, create_visualization=True)

    assert summary["best_map"] == {"y1": "y13", "y2": "y24", "y3": "y36", "y4": "y40"}
    assert summary["result_rows"] == 100
    assert summary["assigned"] == 34
    assert summary["unassigned"] == 66
    assert summary["visualization_path"] is not None
    assert Path(summary["visualization_path"]).exists()
