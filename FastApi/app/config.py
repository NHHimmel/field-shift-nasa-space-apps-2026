"""Runtime settings for the Field Shift web backend, read from the project `.env`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class Settings:
    # --- LLM (Google Gemini, called over REST with httpx) ---
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    gemini_timeout_s: float = _float("GEMINI_TIMEOUT_S", 45.0)

    # --- Unit conversion (model outputs USD/ha; the website shows Taka per bigha) ---
    usd_to_bdt: float = _float("USD_TO_BDT", 122.0)
    # Standard Bangladesh bigha = 33 decimals = 1,335.46 m^2 = 0.13355 ha
    ha_per_bigha: float = _float("HA_PER_BIGHA", 0.13355)

    # --- Data locations ---
    nasa_cache_dir: Path = _path("FIELD_SHIFT_CACHE_DIR", PROJECT_ROOT / "data" / "cache")
    policy_checkpoint: Path = _path(
        "FIELD_SHIFT_POLICY", PROJECT_ROOT / "artifacts" / "models" / "field_shift_sota_policy.pt"
    )
    web_cache_db: Path = _path("FIELD_SHIFT_WEB_CACHE_DB", BACKEND_ROOT / "data" / "web_cache.db")
    frontend_dist: Path = PROJECT_ROOT / "React" / "dist"

    climate_start_year: int = 2005
    climate_end_year: int = 2024


settings = Settings()
