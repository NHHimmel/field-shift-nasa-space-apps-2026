# Field Shift: Adapting Farms with NASA Data

NASA Space Apps Challenge 2026 project for climate-aware crop-rotation decision support in Northwest Bangladesh.

## Project structure

```text
.
├── artifacts/models/     Trained model checkpoints
├── database/schema/       Relational database schema
├── database/seeders/      Database seeding workflow
├── data/raw/             Source datasets and downloads
├── data/processed/       Prepared datasets for experiments
├── docs/                 Challenge brief, research notes, and data guide
├── outputs/              Generated JSON payloads and reports
├── scripts/              Runnable workflows and demos
├── src/field_shift/      Reusable NASA-data and RL modules
├── tests/                Automated tests
├── pyproject.toml        Python project metadata and dependencies
└── uv.lock               Reproducible dependency lockfile
```

## Setup

1. Create a local environment with `uv sync`.
2. Copy `.env.example` to `.env` and add NASA Earthdata credentials when authenticated data sources are needed.
3. Run the feature-pipeline demo:

   ```powershell
   uv run python scripts/run_feature_pipeline.py
   ```

4. Run the full NASA + reinforcement-learning workflow:

   ```powershell
   uv run python scripts/integrate_nasa_with_rl.py
   ```

5. Run the database seeder from the project root:

   ```powershell
   uv run python database/seeders/field_shift_db_seeder.py
   ```

Generated caches are stored under `data/cache/`, model checkpoints under `artifacts/models/`, and frontend-ready exports under `outputs/`.

## Important note

Field Shift is a decision-support prototype. Its recommendations should be reviewed with local agronomic expertise, field observations, and plot-level soil testing before use.
