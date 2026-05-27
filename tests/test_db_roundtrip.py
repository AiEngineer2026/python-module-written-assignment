from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, TrainingData
from src.db.repository import count_rows, load_training_df


def test_db_roundtrip_trainingdata_in_memory() -> None:
    # 1) Create in-memory SQLite DB
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    SessionLocal = sessionmaker(bind=engine, future=True)

    # 2) Insert a few rows
    with SessionLocal() as session:
        session.add_all(
            [
                TrainingData(x=0.0, y1=1.0, y2=2.0, y3=3.0, y4=4.0),
                TrainingData(x=1.0, y1=10.0, y2=20.0, y3=30.0, y4=40.0),
            ]
        )
        session.commit()

        # 3) Count rows via repository
        assert count_rows(session, TrainingData) == 2

        # 4) Load DataFrame via repository
        df = load_training_df(session)

    # 5) Verify DataFrame content
    assert df.shape == (2, 5)
    assert list(df.columns) == ["x", "y1", "y2", "y3", "y4"]
    assert df.iloc[0].to_dict() == {"x": 0.0, "y1": 1.0, "y2": 2.0, "y3": 3.0, "y4": 4.0}
