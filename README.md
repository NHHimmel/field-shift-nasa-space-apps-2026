---
title: Field Shift - NASA Space Apps 2026
emoji: 🛰️🌱
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

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

## Website (FastAPI + React, বাংলা / English)

```text
FastApi/app/   main.py (routes) · engine.py (live model runs) · units.py (USD/ha -> Taka/bigha)
               explainer.py (Gemini, placeholder-based, numbers never touched by the LLM) · cache.py (SQLite)
React/src/     pages/ (Home, Results, About) · components/ · i18n/strings.ts (all UI text, bn + en)
```

1. Backend (from the project root):

   ```powershell
   uv run uvicorn app.main:app --app-dir FastApi --reload --reload-dir FastApi
   ```

   API docs: http://127.0.0.1:8000/docs. Set `GEMINI_API_KEY` in `.env` for AI explanations; without it a built-in bilingual template is used.

2. Frontend (second terminal):

   ```powershell
   cd React
   npm install
   npm run dev
   ```

   Open http://localhost:5173. For a single-server deploy run `npm run build`; FastAPI then serves `React/dist`.

3. Tests: `uv run pytest`.

No login and no user database: the farmer's answers live in the URL, so results can be bookmarked or shared.

## Important note

Field Shift is a decision-support prototype. Its recommendations should be reviewed with local agronomic expertise, field observations, and plot-level soil testing before use.
