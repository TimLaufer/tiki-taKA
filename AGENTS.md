# Repository Guidelines

## Project Structure & Module Organization

- `development/src/` contains tracking loaders, player/ball cleaning, synchronization, and possession-analysis modules. `development/main.py` orchestrates analysis; several modeling stages remain placeholders.
- `development/tests/` holds pytest tests and synthetic fixtures; `development/config/config.yaml` defines paths and analysis parameters.
- Root `evaluation/` contains the official scorer and submission validator. `development/evaluation/` contains separate development helpers.
- `starter/` provides baseline scripts and notebooks; `games.py` defines match metadata. Read `DATA.md`, `EVALUATION.md`, `SUBMISSION.md`, and `RULES.md` before changing data or scoring behavior.
- `competition_data/data/` and `competition_data/cache/` hold ignored local assets and caches. PNGs illustrate analysis results.

## Build, Test, and Development Commands

Run from the repository root unless indicated. Use a virtual environment; there is no build step.

- `python -m pip install -r requirements.txt`: install the scorer's NumPy/pandas dependencies.
- `python -m pip install -r requirements-starter.txt pytest PyYAML`: install notebook, tracking, plotting, and development-test dependencies.
- From `development/`, run `python main.py --save-plot ball.png` to analyze configured match data and save per-half plots.
- From `development/`, run `python -m pytest` for the suite, or `python -m pytest tests/test_data_loader.py` for focused checks.
- `python evaluation/validate_submission.py submission.csv`: validate the frame/label contract.
- `python evaluation/run_evaluation.py`: score CSVs in `evaluation/results/` against the included public five-minute ground truth.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and `UPPER_CASE` constants. Follow nearby type hints and docstrings. Keep algorithms in focused `src/` modules and configurable thresholds in YAML. No formatter or linter is configured.

## Testing Guidelines

Name files `test_<module>.py` and functions `test_<behavior>`. Prefer synthetic arrays and temporary EPTS fixtures from `conftest.py`; tests should not require licensed match data. Cover missing detections, frame boundaries, and invalid inputs when relevant. No coverage threshold is configured.

## Commit & Pull Request Guidelines

History uses short descriptive subjects without a fixed prefix convention. Write focused commits describing the change. PRs should explain behavior, relevant issues, validation commands/results, and configuration changes; include plots when visualization changes.

## Data & Reproducibility

Set `MTAH_DATA_DIR` to locate licensed assets. Never commit private ground truth, tracking exports, or videos. Use public labels only for correctness checks, never training or tuning. Generate corrections and submission labels algorithmically, following `RULES.md`.
