from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from src.config import IDEAL_CSV, TEST_CSV, TRAIN_CSV
from src.core.exceptions import MissingColumnError
from src.db.models import IdealFunctions, TestData, TrainingData


def _validate_columns(df: pd.DataFrame, required_columns: list[str], source_name: str) -> None:
    """Raise a domain-specific exception when a CSV schema is incomplete."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise MissingColumnError(f"{source_name} is missing columns: {', '.join(missing)}")


def _load_csv_to_db(session: Session, csv_path, model_cls, required_columns: list[str]) -> int:
    """Load a CSV file into the database using a given ORM model.

    Args:
        session: SQLAlchemy session.
        csv_path: Path to the CSV file.
        model_cls: ORM model class that matches the CSV columns.
        required_columns: Columns that must exist in the CSV file.

    Returns:
        Number of inserted rows.
    """
    df = pd.read_csv(csv_path)
    _validate_columns(df, required_columns, str(csv_path))
    records = df.to_dict(orient="records")
    session.add_all(model_cls(**r) for r in records)
    session.commit()
    return len(records)


def load_training_to_db(session: Session) -> int:
    """Load training data from train.csv into the training_data table."""
    return _load_csv_to_db(session, TRAIN_CSV, TrainingData, ["x", "y1", "y2", "y3", "y4"])


def load_ideal_to_db(session: Session) -> int:
    """Load ideal functions from ideal.csv into the ideal_functions table."""
    return _load_csv_to_db(
        session,
        IDEAL_CSV,
        IdealFunctions,
        ["x", *(f"y{i}" for i in range(1, 51))],
    )


def load_test_to_db(session: Session) -> int:
    """Load test points from test.csv into the test_data table."""
    return _load_csv_to_db(session, TEST_CSV, TestData, ["x", "y"])
