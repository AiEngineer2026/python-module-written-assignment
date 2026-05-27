"""Database repository helpers.

This module contains small helper functions to read/write data using SQLAlchemy ORM
and convert database rows into pandas DataFrames for further processing.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from src.db.models import IdealFunctions, TestData, TestResult, TrainingData


def count_rows(session: Session, table_model: Any) -> int:
    """Count rows in a database table.

    Args:
        session: SQLAlchemy session.
        table_model: ORM model class (e.g., TrainingData).

    Returns:
        Number of rows in the table.
    """
    stmt = select(func.count()).select_from(table_model)
    return int(session.execute(stmt).scalar_one())


def load_training_df(session: Session) -> pd.DataFrame:
    """Load the training_data table as a pandas DataFrame.

    Returns a DataFrame with columns: x, y1, y2, y3, y4.
    """
    rows = session.execute(select(TrainingData)).scalars().all()
    data = [
        {"x": r.x, "y1": r.y1, "y2": r.y2, "y3": r.y3, "y4": r.y4}
        for r in rows
    ]
    return pd.DataFrame(data)


def load_ideal_df(session: Session) -> pd.DataFrame:
    """Load the ideal_functions table as a pandas DataFrame.

    Returns a DataFrame with columns: x and y1..y50.
    """
    rows = session.execute(select(IdealFunctions)).scalars().all()

    data: list[dict[str, float]] = []
    for r in rows:
        row_dict: dict[str, float] = {"x": float(r.x)}
        for i in range(1, 51):
            key = f"y{i}"
            row_dict[key] = float(getattr(r, key))
        data.append(row_dict)

    return pd.DataFrame(data)


def load_test_df(session: Session) -> pd.DataFrame:
    """Load the test_data table as a pandas DataFrame.

    Returns a DataFrame with columns: x, y.
    """
    rows = session.execute(select(TestData)).scalars().all()
    data = [{"x": r.x, "y": r.y} for r in rows]
    return pd.DataFrame(data)


def save_test_results(session: Session, results: list[dict[str, Any]]) -> int:
    """Insert computed test results into the test_results table.

    Each dict in `results` must contain:
        - x (float)
        - y (float)
        - delta_y (float | None)
        - ideal_func_no (str | None)

    Args:
        session: SQLAlchemy session.
        results: List of result dicts.

    Returns:
        Number of inserted rows.
    """
    objs = [
        TestResult(
            x=r["x"],
            y=r["y"],
            delta_y=r["delta_y"],
            ideal_func_no=r["ideal_func_no"],
        )
        for r in results
    ]
    session.add_all(objs)
    session.commit()
    return len(objs)


def clear_test_results(session: Session) -> None:
    """Delete all rows from the test_results table."""
    session.execute(delete(TestResult))
    session.commit()


def load_test_results_df(session: Session) -> pd.DataFrame:
    """Load the test_results table as a pandas DataFrame."""
    rows = session.execute(select(TestResult)).scalars().all()
    data = [
        {"x": r.x, "y": r.y, "delta_y": r.delta_y, "ideal_func_no": r.ideal_func_no}
        for r in rows
    ]
    return pd.DataFrame(data)


