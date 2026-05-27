# Python Written Assignment - Ideal Function Mapping

This project implements the DLMDSPWP01 written-assignment task: it loads training, ideal, and test data into SQLite, selects the four best ideal functions by least squares, maps test points with the sqrt(2) deviation rule, stores the mapping results, and generates a Bokeh visualization.

## Project Structure

- `src/main.py` - command-line entry point
- `src/core/pipeline.py` - end-to-end workflow orchestration
- `src/core/selector.py` - object-oriented ideal-function selection
- `src/core/mapper.py` - object-oriented test-point mapping
- `src/core/exceptions.py` - user-defined domain exceptions
- `src/db/` - SQLAlchemy models, database setup, and repository helpers
- `src/io/csv_loader.py` - CSV loading with schema validation
- `src/viz/plot_bokeh.py` - standalone Bokeh visualization builder
- `tests/` - unit and integration tests

## Requirements

- Python 3.11 or newer
- Dependencies from `requirements.txt`

Install dependencies in a local virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Input Data

Place the assignment CSV files in `data/`:

- `data/train.csv`
- `data/ideal.csv`
- `data/test.csv`

The paths are configured in `src/config.py`.

## Run the Project

Run the complete reproducible pipeline:

```powershell
python -m src.main
```

The command:

1. creates the SQLite schema if needed,
2. loads the input tables from CSV,
3. rebuilds `test_results`,
4. prints the selected ideal functions and assignment counts,
5. writes `output/assignment_visualization.html`.

Use this optional command if you only want the computation without generating HTML:

```powershell
python -m src.main --skip-visualization
```

## Run Tests

```powershell
pytest -q
```

The test suite covers the numerical metrics, ideal-function selection, threshold mapping, CSV validation, database roundtrip behavior, and the real-data integration result.

## Expected Result for the Provided Data

- Training `y1` maps to ideal `y13`
- Training `y2` maps to ideal `y24`
- Training `y3` maps to ideal `y36`
- Training `y4` maps to ideal `y40`
- 34 test points are assigned
- 66 test points remain unassigned

## Clean Submission Package

Do not submit local environment or cache folders. A clean package should contain only:

- `README.md`
- `requirements.txt`
- `src/`
- `tests/`

Do not include `.venv/`, `.git/`, `.idea/`, `.pytest_cache/`, `__pycache__/`, `*.pyc`, or generated SQLite databases.
