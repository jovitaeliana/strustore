# Repository Guidelines

## Project Structure & Modules
- `strustore-vector-classification/`: Core ML code (training, evaluation, vector DB) under `src/` with data in `src/data/` and artifacts in `models/`.
- `GroundingDINO/` and `Open-GroundingDino/`: Detection models and training utilities.
- `lens/`: Notebook workspace and experiments (`requirements.txt`, reports, comparisons).
- `gdinoOutput/`, `gdino_reports/`, `models/`, `logs/`: Generated outputs and artifacts (do not commit large binaries).
- `yjpa_scraper/`, `zm_scraper/`: Auction scraping notebooks and CSV assets.

## Build, Test, and Dev Commands
- Create env: `python -m venv venv && source venv/bin/activate`.
- Vector pipeline deps: `pip install -r strustore-vector-classification/requirements.txt`.
- Lens notebooks deps: `pip install -r lens/requirements.txt`.
- GroundingDINO (editable): `cd GroundingDINO && pip install -e . && cd -`.
- Train model: `python strustore-vector-classification/src/train_model.py`.
- Evaluate: `python strustore-vector-classification/src/evaluate_model.py`.
- Build vector DB: `python strustore-vector-classification/src/create_vector_database.py`.
- Notebooks: `jupyter notebook` (use the project venv kernel).

## Coding Style & Naming
- Python 3.8+; 4‑space indentation; prefer type hints and docstrings for public functions.
- Naming: snake_case for files/functions, PascalCase for classes, UPPER_SNAKE_CASE for constants.
- Keep modules small and cohesive: data loaders in `src/data/`, training/eval logic in `src/` root, configs in JSON.
- Avoid committing notebook outputs and large artifacts; keep paths configurable.

## Testing Guidelines
- Framework: `pytest` (present in vector-classification). Run with `pytest` from repo root or the subproject.
- Test names: `tests/test_*.py`, function names `test_*`.
- Cover: data loading, training loops (smoke), similarity thresholds, and vector search ranking.
- Include small, synthetic fixtures; avoid relying on large external files.

## Commit & PR Guidelines
- Commits: imperative, concise, and scoped (e.g., `fix vector DB`, `add evaluation thresholds`).
- PRs: clear description, linked issues, steps to reproduce, and sample output snippets/metrics (screenshots for notebooks when relevant).
- Checklists: pass `pytest`, run core scripts end‑to‑end, and ensure no large artifacts in diff (e.g., under `gdinoOutput/` or `models/`).

## Security & Configuration
- Secrets: store in `.env` and never commit. Reference via `python-dotenv` when needed.
- Reproducibility: pin dependencies via the provided `requirements.txt` files; document any CUDA/PyTorch variants used.
