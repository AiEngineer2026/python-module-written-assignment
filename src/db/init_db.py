"""Database initialization helpers."""

from src.db.engine import ENGINE
from src.db.models import (
    Base,
    TrainingData,
    IdealFunctions,
    TestData,
    TestResult,
)


def init_db() -> None:
    """Create all database tables if they do not exist yet."""
    # The imports above ensure the models are registered in Base.metadata.
    Base.metadata.create_all(ENGINE)


