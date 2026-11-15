"""Field Shift web API.

Run from the project root:
    uv run uvicorn app.main:app --app-dir FastApi --reload --reload-dir FastApi
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import engine, explainer
from .config import settings
from .names import SEASON_KEYS, SEASON_NAMES, STRATEGY_KEYS

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s - %(message)s")

app = FastAPI(title="Field Shift API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

Priority = Literal["balanced", "water_conservation", "soil_health", "high_value_vegetable"]


class SeasonCrops(BaseModel):
    rabi: Optional[str] = "auto"
    kharif1: Optional[str] = "auto"
    kharif2: Optional[str] = "auto"


class RecommendRequest(BaseModel):
    district: str
    irrigation: Optional[bool] = Field(None, description="null = use the district default")
    priority: Priority = "balanced"
    soil_fertility: Literal["low", "medium", "high", "unknown"] = "unknown"
    current_crops: Optional[SeasonCrops] = None
    excluded_crops: List[str] = Field(default_factory=list, max_length=60)


class ExplainRequest(RecommendRequest):
    strategy: Literal["baseline", "balanced", "water_conservation", "soil_health", "high_value_vegetable"]
    lang: Literal["bn", "en"] = "bn"


def _run(req: RecommendRequest) -> dict:
    if not engine.is_known_district(req.district):
        raise HTTPException(404, f"Unknown district '{req.district}'")
    current = None
    if req.current_crops is not None:
        current = (
            req.current_crops.rabi or "auto",
            req.current_crops.kharif1 or "auto",
            req.current_crops.kharif2 or "auto",
        )
    bad = [
        c for c in (current or ()) + tuple(req.excluded_crops)
        if c and c not in ("auto", "none", "") and not engine.is_known_crop(c)
    ]
    if bad:
        raise HTTPException(422, f"Unknown crop(s): {', '.join(sorted(set(bad)))}")
    return engine.recommend(
        district=req.district,
        irrigation=req.irrigation,
        soil_fertility=req.soil_fertility,
        current_crops=current,
        excluded_crops=req.excluded_crops,
    )


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "gemini_configured": explainer.gemini_configured(),
        "gemini_model": settings.gemini_model,
        "policy_model": engine.policy_status(),
    }


@app.get("/api/meta")
def meta() -> dict:
    districts = [engine.district_info(d) for d in engine.district_keys()]
    return {
        "districts": districts,
        "crops": engine.crop_catalog(),
        "season_crops": {s: engine.crops_for_season(s) for s in SEASON_KEYS},
        "seasons": [{"key": k, "name_en": SEASON_NAMES[k][0], "name_bn": SEASON_NAMES[k][1]} for k in SEASON_KEYS],
        "strategies": list(STRATEGY_KEYS),
        "default_current_crops": dict(zip(SEASON_KEYS, engine.DEFAULT_BASELINE)),
        "units": {"usd_to_bdt": settings.usd_to_bdt, "ha_per_bigha": settings.ha_per_bigha},
    }


@app.post("/api/recommend")
def recommend(req: RecommendRequest) -> dict:
    result = _run(req)
    return {**result, "recommended_strategy": req.priority}


@app.post("/api/explain")
def explain(req: ExplainRequest) -> dict:
    result = _run(req)
    return {"strategy": req.strategy, "lang": req.lang, **explainer.explain(result, req.strategy, req.lang)}


# ---------------------------------------------------------------------------
# Serve the built React app (React/dist) when it exists — single-server deploys.
# ---------------------------------------------------------------------------
_dist = settings.frontend_dist
if (_dist / "index.html").exists():
    if (_dist / "assets").exists():
        app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404, "Not found")
        candidate = (_dist / full_path).resolve()
        if full_path and candidate.is_file() and _dist.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(_dist / "index.html")
