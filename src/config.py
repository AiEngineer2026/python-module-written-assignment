"""Project-wide configuration constants (paths)."""

from pathlib import Path

# Project root directory (…/Written Assignment/Python)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data and database locations
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = PROJECT_ROOT / "assignment.sqlite"

# Input CSV files
TRAIN_CSV = DATA_DIR / "train.csv"
IDEAL_CSV = DATA_DIR / "ideal.csv"
TEST_CSV = DATA_DIR / "test.csv"

