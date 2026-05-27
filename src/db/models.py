from __future__ import annotations

from sqlalchemy import Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base class for all SQLAlchemy ORM models."""
    pass


class TrainingData(Base):
    """Training dataset (x, y1..y4) loaded from train.csv."""
    __tablename__ = "training_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    x: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)
    y3: Mapped[float] = mapped_column(Float, nullable=False)
    y4: Mapped[float] = mapped_column(Float, nullable=False)


class IdealFunctions(Base):
    """Ideal functions (x, y1..y50) loaded from ideal.csv."""
    __tablename__ = "ideal_functions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    x: Mapped[float] = mapped_column(Float, nullable=False, index=True)

    # Ideal function values y1..y50
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)
    y3: Mapped[float] = mapped_column(Float, nullable=False)
    y4: Mapped[float] = mapped_column(Float, nullable=False)
    y5: Mapped[float] = mapped_column(Float, nullable=False)
    y6: Mapped[float] = mapped_column(Float, nullable=False)
    y7: Mapped[float] = mapped_column(Float, nullable=False)
    y8: Mapped[float] = mapped_column(Float, nullable=False)
    y9: Mapped[float] = mapped_column(Float, nullable=False)
    y10: Mapped[float] = mapped_column(Float, nullable=False)
    y11: Mapped[float] = mapped_column(Float, nullable=False)
    y12: Mapped[float] = mapped_column(Float, nullable=False)
    y13: Mapped[float] = mapped_column(Float, nullable=False)
    y14: Mapped[float] = mapped_column(Float, nullable=False)
    y15: Mapped[float] = mapped_column(Float, nullable=False)
    y16: Mapped[float] = mapped_column(Float, nullable=False)
    y17: Mapped[float] = mapped_column(Float, nullable=False)
    y18: Mapped[float] = mapped_column(Float, nullable=False)
    y19: Mapped[float] = mapped_column(Float, nullable=False)
    y20: Mapped[float] = mapped_column(Float, nullable=False)
    y21: Mapped[float] = mapped_column(Float, nullable=False)
    y22: Mapped[float] = mapped_column(Float, nullable=False)
    y23: Mapped[float] = mapped_column(Float, nullable=False)
    y24: Mapped[float] = mapped_column(Float, nullable=False)
    y25: Mapped[float] = mapped_column(Float, nullable=False)
    y26: Mapped[float] = mapped_column(Float, nullable=False)
    y27: Mapped[float] = mapped_column(Float, nullable=False)
    y28: Mapped[float] = mapped_column(Float, nullable=False)
    y29: Mapped[float] = mapped_column(Float, nullable=False)
    y30: Mapped[float] = mapped_column(Float, nullable=False)
    y31: Mapped[float] = mapped_column(Float, nullable=False)
    y32: Mapped[float] = mapped_column(Float, nullable=False)
    y33: Mapped[float] = mapped_column(Float, nullable=False)
    y34: Mapped[float] = mapped_column(Float, nullable=False)
    y35: Mapped[float] = mapped_column(Float, nullable=False)
    y36: Mapped[float] = mapped_column(Float, nullable=False)
    y37: Mapped[float] = mapped_column(Float, nullable=False)
    y38: Mapped[float] = mapped_column(Float, nullable=False)
    y39: Mapped[float] = mapped_column(Float, nullable=False)
    y40: Mapped[float] = mapped_column(Float, nullable=False)
    y41: Mapped[float] = mapped_column(Float, nullable=False)
    y42: Mapped[float] = mapped_column(Float, nullable=False)
    y43: Mapped[float] = mapped_column(Float, nullable=False)
    y44: Mapped[float] = mapped_column(Float, nullable=False)
    y45: Mapped[float] = mapped_column(Float, nullable=False)
    y46: Mapped[float] = mapped_column(Float, nullable=False)
    y47: Mapped[float] = mapped_column(Float, nullable=False)
    y48: Mapped[float] = mapped_column(Float, nullable=False)
    y49: Mapped[float] = mapped_column(Float, nullable=False)
    y50: Mapped[float] = mapped_column(Float, nullable=False)


class TestData(Base):
    """Raw test points (x, y) loaded from test.csv."""
    __tablename__ = "test_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    x: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    y: Mapped[float] = mapped_column(Float, nullable=False)


class TestResult(Base):
    """Assignment results for test points (x, y) to an ideal function."""
    __tablename__ = "test_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    x: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    delta_y: Mapped[float] = mapped_column(Float, nullable=True)
    ideal_func_no: Mapped[str] = mapped_column(String(3), nullable=True)  # e.g. "y13"

