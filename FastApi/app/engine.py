"""Field Shift recommendation engine for the website.

Thin wrapper around the team's existing pipeline (`src/field_shift/rl_pipeline.py` and
`scripts/integrate_nasa_with_rl.py`). It reuses the same building blocks the JSON
exporter uses (evaluate_rotation_trajectory, compute_20yr_historical_uncertainty,
simulate_climate_scenario, the trained temporal-attention encoder) but:

  * runs live per request, so farmer inputs (irrigation, soil fertility, current crops,
    crops to avoid) are respected, using the district's REAL latest NASA climate
    vectors instead of the placeholder climates in `query_field_shift_api()`;
  * converts every money / per-area value to Taka per bigha (see units.py);
  * drops the pre-written English sentences (they contain $ figures) — the website
    builds its own bilingual text from the structured numbers.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
from datetime import date, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import PROJECT_ROOT, settings
from .names import DISTRICT_BN, DIVISION_BN, SEASON_KEYS, crop_name, district_name
from .units import kg_per_bigha, money_bdt_per_bigha, round1

# rl_pipeline looks for `field_shift_bangladesh.db` relative to the working directory.
os.chdir(PROJECT_ROOT)
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from field_shift import rl_pipeline as rl  # noqa: E402
from field_shift.district_catalog import DISTRICT_METADATA  # noqa: E402
from field_shift.nasa_data import FarmLocation, load_bangladesh_districts  # noqa: E402
from integrate_nasa_with_rl import build_multi_decade_regional_dataset  # noqa: E402

logger = logging.getLogger("field_shift_web.engine")

# ---------------------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------------------

STRATEGY_MODES: List[Tuple[str, str]] = [
    ("balanced", "core"),
    ("water_conservation", "core"),
    ("soil_health", "core"),
    ("high_value_vegetable", "veg"),
]

NASA_SOURCES: List[Dict[str, str]] = [
    {
        "name": "NASA POWER",
        "role": "daily rainfall and weather history",
        "status": "active",
    },
    {
        "name": "NASA SMAP",
        "role": "root-zone soil wetness feature",
        "status": "derived",
    },
    {
        "name": "NASA GRACE / GRACE-FO",
        "role": "groundwater percentile and drought context",
        "status": "derived",
    },
    {
        "name": "NASA MODIS / ECOSTRESS",
        "role": "evapotranspiration demand feature",
        "status": "derived",
    },
]

SCENARIOS: List[Dict[str, Any]] = [
    {"id": "normal", "rain_pct": 0.0, "temp_c": 0.0, "gw_pct": 0.0},
    {"id": "drought", "rain_pct": -25.0, "temp_c": 1.5, "gw_pct": -10.0},
    {"id": "heatwave", "rain_pct": -10.0, "temp_c": 2.5, "gw_pct": -5.0},
    {"id": "wet_monsoon", "rain_pct": 30.0, "temp_c": 0.5, "gw_pct": 15.0},
]

# Same soil-fertility presets as `query_field_shift_api()` (soil N kg/ha, organic matter %)
FERTILITY_PRESETS: Dict[str, Tuple[float, float]] = {
    "low": (50.0, 0.8),
    "medium": (75.0, 1.2),
    "high": (105.0, 1.8),
}

# Season windows used by `extract_year_trajectories()` -> (start month, start day, length)
SEASON_WINDOWS = [(11, 1, 135), (3, 16, 107), (7, 1, 123)]

DEFAULT_BASELINE = ("Boro", "Fallow", "Aman")

# ---------------------------------------------------------------------------
# District + climate data
# ---------------------------------------------------------------------------

_DISTRICTS: Dict[str, FarmLocation] = {
    loc.district_name: loc
    for loc in load_bangladesh_districts(PROJECT_ROOT / "field_shift_bangladesh.db")
}
_climate_lock = threading.Lock()
_climate_cache: Dict[str, Tuple[Dict[str, Any], List[Dict[str, Any]]]] = {}


def district_keys() -> List[str]:
    return sorted(_DISTRICTS)


def district_info(name: str) -> Dict[str, Any]:
    loc = _DISTRICTS[name]
    division = DISTRICT_METADATA.get(name, ("",))[0]
    return {
        "key": name,
        "name_en": district_name(name, "en"),
        "name_bn": DISTRICT_BN.get(name, name),
        "division_en": division,
        "division_bn": DIVISION_BN.get(division, division),
        "agro_ecological_zone": loc.aez_zone,
        "latitude": float(loc.latitude),
        "longitude": float(loc.longitude),
        "groundwater_pctl": float(loc.groundwater_baseline_pctl),
        "default_irrigation": bool(loc.irrigation_access),
    }


def _district_climate(name: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Latest-year record + 20-year history for one district (NASA POWER, disk-cached)."""
    with _climate_lock:
        if name not in _climate_cache:
            loc = _DISTRICTS[name]
            _, history_map, latest_map = build_multi_decade_regional_dataset(
                districts=[loc],
                start_year=settings.climate_start_year,
                end_year=settings.climate_end_year,
                cache_dir=str(settings.nasa_cache_dir),
            )
            _climate_cache[name] = (latest_map[name], history_map[name])
        return _climate_cache[name]


# ---------------------------------------------------------------------------
# Trained policy (only the temporal-attention encoder is used, for stress windows)
# ---------------------------------------------------------------------------

_policy_lock = threading.Lock()
_policy_state: Dict[str, Any] = {"loaded": False, "model": None, "note": ""}


def _policy_model() -> Optional[Any]:
    with _policy_lock:
        if _policy_state["loaded"]:
            return _policy_state["model"]
        _policy_state["loaded"] = True
        if not rl.HAS_TORCH:
            _policy_state["note"] = "PyTorch not available"
            return None
        path = settings.policy_checkpoint
        if not path.exists():
            _policy_state["note"] = f"checkpoint not found: {path.name}"
            return None
        try:
            # Handles the {"state_dict": ..., "model_config": ...} format the training script saves.
            model = rl.load_trained_policy(path, device="cpu")
            _policy_state["model"] = model
            _policy_state["note"] = "trained checkpoint loaded"
        except Exception as exc:  # e.g. checkpoint trained with a different crop catalog size
            logger.warning("Could not load policy checkpoint (%s); using physics stress proxy.", exc)
            _policy_state["note"] = f"checkpoint incompatible ({type(exc).__name__})"
        return _policy_state["model"]


def policy_status() -> str:
    _policy_model()
    return _policy_state["note"]


def inference_info(stress_source: str) -> Dict[str, Any]:
    """Return the method actually used, so the UI cannot overstate model usage."""
    _policy_model()
    checkpoint = settings.policy_checkpoint
    return {
        # The live website currently evaluates candidate rotations with the
        # structured simulator. The neural checkpoint is used only for stress
        # windows until a compatible policy is retrained and wired into this path.
        "recommendation_method": "deterministic_rotation_search",
        "neural_policy_used": False,
        "policy_status": _policy_state["note"],
        "stress_window_source": stress_source,
        "checkpoint_name": checkpoint.name if checkpoint.exists() else None,
    }


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------

def crop_catalog() -> List[Dict[str, Any]]:
    out = []
    for c in rl.CROP_CATALOG:
        out.append({
            "key": c.name,
            "name_en": crop_name(c.name, "en"),
            "name_bn": crop_name(c.name, "bn"),
            "season": SEASON_KEYS[rl.SEASON_NAMES.index(c.season)] if c.season in rl.SEASON_NAMES else None,
            "category": c.category,
            "family": c.family,
        })
    return out


def crops_for_season(season_key: str) -> List[str]:
    model_season = rl.SEASON_NAMES[SEASON_KEYS.index(season_key)]
    names = [c.name for c in rl.CROP_CATALOG if c.season == model_season]
    if "Fallow" in rl.CROP_NAME_TO_ID and "Fallow" not in names:
        names.append("Fallow")
    return names


def is_known_crop(name: str) -> bool:
    return name in rl.CROP_NAME_TO_ID


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def _pest(status: str) -> Dict[str, Any]:
    if status.startswith("Pest cycle broken"):
        m = re.search(r"\((.+?) -> (.+?)\)", status)
        return {"kind": "broken", "from": m.group(1) if m else None, "to": m.group(2) if m else None}
    if status.startswith("Monoculture"):
        m = re.search(r"\((\S+) repeated", status)
        return {"kind": "repeat", "family": m.group(1) if m else None}
    if status.startswith("Fallow"):
        return {"kind": "fallow"}
    return {"kind": "first"}


def _season(step: Dict[str, Any]) -> Dict[str, Any]:
    idx = int(step["season_index"])
    return {
        "season": SEASON_KEYS[idx],
        "crop": str(step["crop"]),
        "category": str(step["crop_category"]),
        "family": str(step["botanical_family"]),
        "sowing_window": str(step["sowing_window"]),
        "harvest_window": str(step["harvest_window"]),
        "season_days": int(rl.SEASON_TOTAL_DAYS[idx]),
        "active_days": int(step["active_growing_days"]),
        "bare_days": int(step["bare_field_days"]),
        "water_mm": round1(step["water_mm"]),
        "irrigation_mm": round1(step["blue_water_irrigation_mm"]),
        "rain_mm": round1(step["effective_rain_mm"]),
        "mean_temp_c": round1(step["mean_temp_c"]),
        "heat_stress_days": int(step["heat_stress_days"]),
        "dry_spell_days": int(step["consecutive_dry_days"]),
        "rootzone_wetness_pct": int(round(float(step["rootzone_wetness_frac"]) * 100)),
        "groundwater_pctl": round1(step["grace_groundwater_pctl"]),
        "soil_n_change_kg_bigha": kg_per_bigha(step["soil_n_delta_kg_ha"]),
        "pest": _pest(str(step["pest_break_status"])),
    }


def _metrics(ev: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    profit = money_bdt_per_bigha(ev["total_profit_usd"])
    base_profit = money_bdt_per_bigha(base["total_profit_usd"])
    n_change = kg_per_bigha(ev["n_delta"])
    base_n_change = kg_per_bigha(base["n_delta"])
    return {
        "profit_bdt_bigha": profit,
        "profit_change_bdt_bigha": profit - base_profit,
        "water_mm": round1(ev["total_water_mm"]),
        "irrigation_mm": round1(ev["total_blue_water_mm"]),
        "water_saved_mm": round1(float(base["total_water_mm"]) - float(ev["total_water_mm"])),
        "soil_n_kg_bigha": kg_per_bigha(ev["final_soil_n"]),
        "soil_n_change_kg_bigha": n_change,
        "soil_n_change_vs_baseline_kg_bigha": round1(n_change - base_n_change),
        "cover_days": int(ev["active_cover_days_yr"]),
        "bare_days": int(ev["bare_field_days_yr"]),
    }


def _range(unc: Dict[str, float]) -> Dict[str, Any]:
    return {
        "profit_typical_bdt_bigha": money_bdt_per_bigha(unc["profit_mean_usd"]),
        "profit_spread_bdt_bigha": money_bdt_per_bigha(unc["profit_std_usd"]),
        "profit_bad_year_bdt_bigha": money_bdt_per_bigha(unc["profit_p10_downside_usd"]),
        "profit_good_year_bdt_bigha": money_bdt_per_bigha(unc["profit_p90_upside_usd"]),
        "water_typical_mm": round1(unc["water_mean_mm"]),
        "water_dry_year_mm": round1(unc["water_p90_drought_demand_mm"]),
    }


def _crop_keys(ev: Dict[str, Any]) -> List[str]:
    return [rl.CROP_CATALOG[i].name for i in ev["sequence_ids"]]


def _stress_windows(daily_seqs: Sequence[np.ndarray]) -> Dict[str, Any]:
    model = _policy_model()
    source = "trained_attention" if model is not None else "physics_proxy"
    windows = []
    for s_idx, seq in enumerate(daily_seqs):
        if model is not None:
            import torch

            with torch.no_grad():
                _, attn = model.temporal_encoder(
                    torch.tensor(seq, dtype=torch.float32).unsqueeze(0), return_weights=True
                )
            weights = attn.numpy().flatten()
        else:
            # Same physics proxy as extract_temporal_attention_windows(): high VPD + Tmax - root-zone wetness
            stress_raw = seq[:, 1] + seq[:, 10] - seq[:, 9]
            exp_s = np.exp(stress_raw - np.max(stress_raw))
            weights = exp_s / np.sum(exp_s)

        profile = [round(float(np.sum(weights[i * 10:(i + 1) * 10])), 4) for i in range(12)]
        peak = int(np.argmax(profile))
        month, day, length = SEASON_WINDOWS[s_idx]
        start = date(2021 if s_idx == 0 else 2022, month, day)  # non-leap reference years
        peak_start = start + timedelta(days=int(peak * length / 12))
        peak_end = start + timedelta(days=int((peak + 1) * length / 12) - 1)
        windows.append({
            "season": SEASON_KEYS[s_idx],
            "profile": profile,
            "peak_index": peak,
            "peak_start": peak_start.strftime("%m-%d"),
            "peak_end": peak_end.strftime("%m-%d"),
        })
    return {"source": source, "seasons": windows}


def _filter_pool(pool: List[Tuple[int, int, int]], excluded_ids: set) -> List[Tuple[int, int, int]]:
    filtered = [p for p in pool if not (set(p) & excluded_ids)]
    return filtered or pool


def _best(farm, climates, pool, mode, daily_seqs=None) -> Dict[str, Any]:
    evals = [rl.evaluate_rotation_trajectory(farm, climates, path, mode, daily_seqs) for path in pool]
    return max(evals, key=lambda e: e["cumulative_reward"])


@lru_cache(maxsize=512)
def _compute(
    district: str,
    irrigation: bool,
    fertility: str,
    current: Tuple[str, str, str],
    excluded: Tuple[str, ...],
) -> Dict[str, Any]:
    latest, history = _district_climate(district)
    climates = latest["seasonal_vectors"]
    daily_seqs = latest["seasonal_daily_seqs"]
    loc = _DISTRICTS[district]

    base_n, base_som = FERTILITY_PRESETS[fertility]
    farm = rl.FarmProfile(
        farm_id=loc.farm_id,
        irrigation_access=irrigation,
        baseline_soil_n_kg_ha=base_n,
        soil_organic_matter_pct=base_som,
        priority_mode="balanced",
        district_name=district,
    )

    baseline_ids = tuple(rl.CROP_NAME_TO_ID[c] for c in current)
    excluded_ids = {rl.CROP_NAME_TO_ID[c] for c in excluded}
    core_pool, veg_pool = rl._get_candidate_pools()
    pools = {
        "core": _filter_pool(core_pool, excluded_ids),
        "veg": _filter_pool(veg_pool, excluded_ids),
    }

    # --- Current-climate strategies (with 20-year NASA uncertainty band) ---
    base_ev = rl.evaluate_rotation_trajectory(farm, climates, baseline_ids, "balanced", daily_seqs)
    base_unc = rl.compute_20yr_historical_uncertainty(farm, history, baseline_ids, "balanced")
    evaluated = [("baseline", base_ev, base_unc)]
    for key, pool_name in STRATEGY_MODES:
        ev = _best(farm, climates, pools[pool_name], key, daily_seqs)
        unc = rl.compute_20yr_historical_uncertainty(farm, history, tuple(ev["sequence_ids"]), key)
        evaluated.append((key, ev, unc))

    strategies = [
        {
            "key": key,
            "crops": _crop_keys(ev),
            "metrics": _metrics(ev, base_ev),
            "range_20yr": _range(unc),
            "seasons": [_season(step) for step in ev["seasonal_timeline"]],
        }
        for key, ev, unc in evaluated
    ]

    # --- What-if weather scenarios ---
    scenarios = []
    for sc in SCENARIOS:
        if sc["id"] == "normal":
            rows = [{"key": s["key"], "crops": s["crops"], "metrics": s["metrics"]} for s in strategies]
        else:
            pert = rl.simulate_climate_scenario(
                climates,
                rainfall_delta_pct=sc["rain_pct"],
                temp_delta_c=sc["temp_c"],
                grace_gw_delta_pct=sc["gw_pct"],
            )
            sc_base = rl.evaluate_rotation_trajectory(farm, pert, baseline_ids, "balanced")
            rows = [{"key": "baseline", "crops": _crop_keys(sc_base), "metrics": _metrics(sc_base, sc_base)}]
            for key, pool_name in STRATEGY_MODES:
                ev = _best(farm, pert, pools[pool_name], key)
                rows.append({"key": key, "crops": _crop_keys(ev), "metrics": _metrics(ev, sc_base)})
        scenarios.append({**sc, "strategies": rows})

    stress_windows = _stress_windows(daily_seqs)
    return {
        "district": district_info(district),
        "inputs": {
            "irrigation": irrigation,
            "soil_fertility": fertility,
            "current_crops": dict(zip(SEASON_KEYS, current)),
            "excluded_crops": list(excluded),
        },
        "strategies": strategies,
        "scenarios": scenarios,
        "stress_windows": stress_windows,
        "units": {
            "currency": "BDT",
            "area": "bigha",
            "usd_to_bdt": settings.usd_to_bdt,
            "ha_per_bigha": settings.ha_per_bigha,
        },
        "data_info": {
            "climate_years": [settings.climate_start_year, settings.climate_end_year],
            "latest_year": settings.climate_end_year,
            "crop_catalog_size": rl.NUM_CROPS,
            "nasa_sources": NASA_SOURCES,
            "inference": inference_info(stress_windows["source"]),
        },
    }


def recommend(
    district: str,
    irrigation: Optional[bool],
    soil_fertility: str,
    current_crops: Optional[Tuple[str, str, str]],
    excluded_crops: Sequence[str],
) -> Dict[str, Any]:
    loc = _DISTRICTS[district]
    irr = bool(loc.irrigation_access) if irrigation is None else bool(irrigation)
    fert = soil_fertility if soil_fertility in FERTILITY_PRESETS else "medium"
    current = tuple(current_crops) if current_crops else DEFAULT_BASELINE
    excluded = tuple(sorted(set(excluded_crops)))
    result = _compute(district, irr, fert, current, excluded)
    return {**result, "inputs": {**result["inputs"], "irrigation_was_default": irrigation is None}}


def is_known_district(name: str) -> bool:
    return name in _DISTRICTS
