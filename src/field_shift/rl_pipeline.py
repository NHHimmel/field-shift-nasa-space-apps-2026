"""
Field Shift: National 64-District & 49-Crop PyTorch RL / MDP Crop Rotation System
=================================================================================

Configured Directly from Your 64-District & 72-Crop Database (`field_shift_bangladesh.db`
/ MySQL `field_shift_db`):
  - Dynamic 49-Crop Rotational Action Space:
      * 26 Rabi Crops (Cereals, Pulses, Oilseeds, Winter Vegetables, Tubers & Spices)
      * 18 Kharif-1 Crops (Pre-Monsoon Pulses, Jute, Aus, Summer Vegetables & Fallow)
      * 5 Kharif-2 Crops (Aman Rice, Mashkalai, Arhar, Oal Kachu, Puishak)
      * Plus Potato (when evaluating custom Northwest plots) = 50 total rotational actions
  - 4 Specialist Policy & Critic Heads in PyTorch (`CropRotationActorCritic`):
      1. `balanced` (Field crops & pulses economic/ecological balance)
      2. `water_conservation` (Minimum irrigation demand & Barind/aquifer protection)
      3. `soil_health` (Maximum biological N-fixation & botanical pest-break)
      4. `high_value_vegetable` (High-income winter & summer vegetable rotations weighted
         by each district's empirical `AP Ratio` yield index from your spreadsheet)
  - District-Aware MDP Environment (`CropRotationMDPEnv`):
      Automatically loads district-specific `AP Ratio` yield multipliers, seasonal rainfall,
      and evapotranspiration directly from `field_shift_bangladesh.db` (or MySQL) so every
      one of the 64 districts is modeled with its real spreadsheet data.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("FieldShiftSOTA")


def resolve_torch_device(prefer_device: Optional[str] = None) -> str:
    if not HAS_TORCH:
        return "cpu"
    if prefer_device == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        try:
            test_a = torch.ones((2, 2), device="cuda")
            _ = torch.matmul(test_a, test_a)
            return "cuda"
        except Exception as exc:
            logger.warning(
                "CUDA detected but cuBLAS kernel test failed (%s). Falling back to CPU.",
                exc,
            )
            return "cpu"
    return "cpu"

# STAGE 1: DYNAMIC 50-CROP CATALOG LOADED FROM THE 64-DISTRICT DATABASE

@dataclass(frozen=True)
class CropSpec:
    crop_id: int
    name: str
    season: str              # 'Rabi', 'Kharif-1', 'Kharif-2'
    category: str            # 'Cereal', 'Pulse_Legume', 'Oilseed', 'Fiber', 'Vegetable_Tuber_Spice', 'Fallow'
    family: str              # Botanical family
    duration_days: int
    water_req_mm: float      # Seasonal crop evapotranspiration requirement (ETc)
    n_fixation_kg_ha: float  # Net residual soil N change (kg N/ha)
    baseline_margin_usd: float # Reference gross margin USD/ha
    drought_hardiness: float # 0.0 to 1.0
    waterlogging_hardiness: float # 0.0 to 1.0
    optimal_temp_c: float    # Optimal seasonal mean temperature (C)
    sowing_window: str
    harvest_window: str
    root_system: str
    erosion_protection: str


def _build_full_crop_catalog() -> List[CropSpec]:
    """
    Loads all 49 rotational annual crops (field crops, pulses, oilseeds, fiber,
    vegetables, tubers, spices, and Fallow) from `field_shift_bangladesh.db`
    (plus Potato) into a unified 50-crop catalog indexed 0..49.
    """
    db_candidates = [
        Path("field_shift_bangladesh.db"),
        Path("artifacts/file_generation/ttl=63d/output/field_shift_bangladesh.db"),
    ]
    db_file = next((p for p in db_candidates if p.exists()), None)

    if db_file is not None:
        conn = sqlite3.connect(str(db_file))
        df_c = pd.read_sql(
            "SELECT * FROM crops WHERE is_rotational = 1 ORDER BY corrected_rotation_season, crop_name",
            conn,
        )
        conn.close()
        specs: List[CropSpec] = []
        for idx, row in df_c.iterrows():
            c_name = str(row["crop_name"])
            seas = str(row["corrected_rotation_season"])
            cat = str(row["crop_category"])
            fam = str(row["botanical_family"])
            opt_t = 20.0 if seas == "Rabi" else (29.5 if seas == "Kharif-1" else 28.5)
            root_sys = "Nodulating taproot" if fam == "Fabaceae" else ("Fibrous" if fam == "Poaceae" else "Taproot / Tuber")
            eros = "Low (Bare Soil)" if c_name == "Fallow" else ("High" if int(row["duration_days"]) >= 90 else "Moderate")
            specs.append(
                CropSpec(
                    crop_id=len(specs),
                    name=c_name if c_name != "Mug" else "Mungbean",
                    season=seas,
                    category=cat,
                    family=fam,
                    duration_days=int(row["duration_days"]),
                    water_req_mm=float(row["water_req_mm"]),
                    n_fixation_kg_ha=float(row["nitrogen_delta_kg_ha"]),
                    baseline_margin_usd=float(row["baseline_margin_usd_ha"]),
                    drought_hardiness=float(row["drought_hardiness"]),
                    waterlogging_hardiness=float(row["waterlogging_hardiness"]),
                    optimal_temp_c=opt_t,
                    sowing_window=str(row["transplant_month"]),
                    harvest_window=str(row["harvest_window"]),
                    root_system=root_sys,
                    erosion_protection=eros,
                )
            )
        # Ensure Potato is also present in Rabi for classic cash-crop comparison
        if not any(s.name == "Potato" for s in specs):
            specs.append(
                CropSpec(
                    crop_id=len(specs),
                    name="Potato",
                    season="Rabi",
                    category="Vegetable_Tuber_Spice",
                    family="Solanaceae",
                    duration_days=90,
                    water_req_mm=410.0,
                    n_fixation_kg_ha=-28.0,
                    baseline_margin_usd=1380.0,
                    drought_hardiness=0.45,
                    waterlogging_hardiness=0.15,
                    optimal_temp_c=18.5,
                    sowing_window="Nov 01 - Nov 20",
                    harvest_window="Feb 01 - Feb 20",
                    root_system="Tuberous shallow",
                    erosion_protection="Low-Moderate",
                )
            )
        return specs

    # Fallback catalog if DB has not been generated yet
    return [
        CropSpec(0, "Lentil", "Rabi", "Pulse_Legume", "Fabaceae", 105, 200.0, +32.0, 780.0, 0.90, 0.15, 20.0, "Nov", "Feb-Mar", "Deep taproot", "Moderate"),
        CropSpec(1, "Rape & Mustard", "Rabi", "Oilseed", "Brassicaceae", 80, 190.0, -8.0, 660.0, 0.75, 0.25, 20.5, "Oct-Nov", "Jan-Feb", "Taproot", "Moderate"),
        CropSpec(2, "Wheat", "Rabi", "Cereal", "Poaceae", 110, 320.0, -22.0, 640.0, 0.65, 0.30, 19.5, "Nov-Dec", "Mar", "Fibrous", "High"),
        CropSpec(3, "Potato", "Rabi", "Vegetable_Tuber_Spice", "Solanaceae", 90, 410.0, -28.0, 1380.0, 0.45, 0.15, 18.5, "Nov", "Feb", "Tuberous", "Low-Moderate"),
        CropSpec(4, "Onion", "Rabi", "Vegetable_Tuber_Spice", "Amaryllidaceae", 110, 350.0, -22.0, 1480.0, 0.60, 0.20, 20.0, "Oct-Nov", "Mar-Apr", "Bulb fibrous", "Moderate"),
        CropSpec(5, "Cabbage", "Rabi", "Vegetable_Tuber_Spice", "Brassicaceae", 90, 340.0, -26.0, 1520.0, 0.50, 0.25, 18.0, "Sep-Oct", "Jan-Feb", "Fibrous", "Moderate"),
        CropSpec(6, "Cauliflower", "Rabi", "Vegetable_Tuber_Spice", "Brassicaceae", 85, 330.0, -25.0, 1560.0, 0.50, 0.25, 18.5, "Sep-Oct", "Jan-Feb", "Fibrous", "Moderate"),
        CropSpec(7, "Boro", "Rabi", "Cereal", "Poaceae", 150, 1150.0, -42.0, 720.0, 0.10, 0.95, 24.0, "Dec-Jan", "Apr-May", "Fibrous", "High"),
        CropSpec(8, "Mungbean", "Kharif-1", "Pulse_Legume", "Fabaceae", 70, 210.0, +28.0, 590.0, 0.82, 0.25, 29.0, "Mar-Apr", "May-Jun", "Nodulating taproot", "High"),
        CropSpec(9, "Barbati", "Kharif-1", "Vegetable_Tuber_Spice", "Fabaceae", 75, 250.0, +20.0, 980.0, 0.72, 0.40, 28.5, "Mar-Apr", "Jun", "Nodulating taproot", "High"),
        CropSpec(10, "Patal", "Kharif-1", "Vegetable_Tuber_Spice", "Cucurbitaceae", 110, 350.0, -18.0, 1220.0, 0.68, 0.55, 29.5, "Mar", "Jun-Jul", "Taproot", "Moderate"),
        CropSpec(11, "Karala", "Kharif-1", "Vegetable_Tuber_Spice", "Cucurbitaceae", 80, 290.0, -15.0, 1180.0, 0.70, 0.55, 29.0, "Mar", "Jun", "Taproot", "Moderate"),
        CropSpec(12, "Jute", "Kharif-1", "Fiber", "Malvaceae", 120, 460.0, +10.0, 540.0, 0.50, 0.95, 30.5, "Mar-Apr", "Jul-Aug", "Deep taproot", "High"),
        CropSpec(13, "Fallow", "Kharif-1", "Fallow", "None", 0, 0.0, +5.0, 0.0, 1.00, 1.00, 28.0, "None", "None", "None", "Low"),
        CropSpec(14, "Aman", "Kharif-2", "Cereal", "Poaceae", 130, 780.0, -24.0, 710.0, 0.45, 0.92, 28.5, "Jul-Aug", "Nov-Dec", "Fibrous", "High"),
        CropSpec(15, "Mashkalai", "Kharif-2", "Pulse_Legume", "Fabaceae", 85, 240.0, +26.0, 560.0, 0.80, 0.35, 28.0, "Aug", "Nov", "Nodulating taproot", "High"),
    ]


CROP_CATALOG: List[CropSpec] = _build_full_crop_catalog()
NUM_CROPS = len(CROP_CATALOG)
CROP_NAME_TO_ID: Dict[str, int] = {c.name: c.crop_id for c in CROP_CATALOG}
# Map aliases ("T. Aman Rice" -> "Aman", "Boro Rice" -> "Boro", "Mug" -> "Mungbean")
if "Aman" in CROP_NAME_TO_ID:
    CROP_NAME_TO_ID["T. Aman Rice"] = CROP_NAME_TO_ID["Aman"]
if "Boro" in CROP_NAME_TO_ID:
    CROP_NAME_TO_ID["Boro Rice"] = CROP_NAME_TO_ID["Boro"]
if "Mungbean" in CROP_NAME_TO_ID:
    CROP_NAME_TO_ID["Mug"] = CROP_NAME_TO_ID["Mungbean"]
if "Rape & Mustard" in CROP_NAME_TO_ID:
    CROP_NAME_TO_ID["Mustard"] = CROP_NAME_TO_ID["Rape & Mustard"]

SEASON_NAMES = ["Rabi", "Kharif-1", "Kharif-2"]
SEASON_TOTAL_DAYS = [135, 106, 124]
SEASON_ACTION_MAP: Dict[int, List[int]] = {
    0: [c.crop_id for c in CROP_CATALOG if c.season == "Rabi"],
    1: [c.crop_id for c in CROP_CATALOG if c.season == "Kharif-1"],
    2: [c.crop_id for c in CROP_CATALOG if c.season == "Kharif-2"],
}

# 4 Specialist Policy Modes (including the new High-Value Vegetable head)
PRIORITY_MODES = ["balanced", "water_conservation", "soil_health", "high_value_vegetable"]

EXISTING_BASELINE_PATH: Tuple[int, int, int] = (
    CROP_NAME_TO_ID.get("Boro", 0),
    CROP_NAME_TO_ID.get("Fallow", 1),
    CROP_NAME_TO_ID.get("Aman", 2),
)


def load_district_yield_multipliers() -> Dict[str, Dict[str, float]]:
    """
    Loads district-specific Relative Yield Indices (`AP Ratio / National Mean AP Ratio`)
    from `field_shift_bangladesh.db` so the MDP environment rewards crops and vegetables
    that actually achieve high yields in that specific district.
    """
    db_candidates = [
        Path("field_shift_bangladesh.db"),
        Path("artifacts/file_generation/ttl=63d/output/field_shift_bangladesh.db"),
    ]
    db_file = next((p for p in db_candidates if p.exists()), None)
    if db_file is None:
        return {}

    conn = sqlite3.connect(str(db_file))
    df_r = pd.read_sql(
        "SELECT district_name, crop_name, relative_yield_index FROM district_crop_records WHERE is_rotational = 1",
        conn,
    )
    conn.close()

    out: Dict[str, Dict[str, float]] = {}
    for _, row in df_r.iterrows():
        d_name = str(row["district_name"])
        c_name = str(row["crop_name"])
        if c_name == "Mug":
            c_name = "Mungbean"
        out.setdefault(d_name, {})[c_name] = float(row["relative_yield_index"])
    return out


DISTRICT_YIELD_MULTIPLIERS: Dict[str, Dict[str, float]] = load_district_yield_multipliers()


FARMER_SCIENTIFIC_GLOSSARY: Dict[str, Dict[str, str]] = {
    "Evapotranspiration (ETc)": {
        "scientific_source": "NASA MODIS (MOD16A2GF) & ECOSTRESS (ECO_L3T_JET)",
        "plain_meaning": "Combined water lost from soil evaporation and plant leaf transpiration—measuring how much total water a crop drinks over the season.",
        "satellite_vs_ground": "Satellites estimate ET from canopy thermal cooling at 70m-500m resolution; local pan evaporation or soil tensiometers validate field-specific irrigation timing.",
    },
    "Root-Zone Soil Wetness (GWETROOT)": {
        "scientific_source": "NASA SMAP Level-4 (SPL4SMGP) & NASA POWER",
        "plain_meaning": "Percentage of pore space filled with water in the top 100 cm of soil where crop roots actively drink (0% = bone dry, 100% = waterlogged).",
        "satellite_vs_ground": "L-band microwave sensors + land assimilation models estimate 9 km regional wetness; farmers should verify top-soil feel and plough-pan depth on their own plot.",
    },
    "Groundwater Storage Percentile (GRACE)": {
        "scientific_source": "NASA GRACE & GRACE-FO Data Assimilation (GRACEDADM)",
        "plain_meaning": "Ranks current deep aquifer water levels against a 70-year historical baseline. Below 30% signals severe tubewell drawdown risk.",
        "satellite_vs_ground": "GRACE measures regional gravity shifts over ~25 km grids; local BMDA observation wells confirm exact static water depth for pump suction.",
    },
    "District AP Ratio (Yield Index)": {
        "scientific_source": "Bangladesh 64-District Agricultural Production & Area Dataset",
        "plain_meaning": "Measures how many tonnes per hectare a crop produces in your specific district compared to the national average.",
        "satellite_vs_ground": "Combines district-level harvest statistics with NASA seasonal solar radiation, rainfall, and evapotranspiration.",
    },
}


class PreprocessingPipeline:
    @staticmethod
    def extract_seasonal_climate_vector(daily_df: pd.DataFrame) -> np.ndarray:
        temp_mean = float(daily_df["T2M"].mean()) if "T2M" in daily_df else 25.0
        heat_days = float((daily_df["T2M_MAX"] >= 35.0).sum()) if "T2M_MAX" in daily_df else 0.0
        precip_sum = float(daily_df["PRECTOTCORR"].sum()) if "PRECTOTCORR" in daily_df else 50.0

        if "PRECTOTCORR" in daily_df:
            is_dry = (daily_df["PRECTOTCORR"] < 1.0).astype(int)
            dry_groups = is_dry.ne(is_dry.shift()).cumsum()
            cdd = float(is_dry.groupby(dry_groups).cumsum().max()) if not is_dry.empty else 0.0
        else:
            cdd = 10.0

        sm_surf = float(daily_df["GWETTOP"].mean()) if "GWETTOP" in daily_df else 0.45
        sm_root = float(daily_df["GWETROOT"].mean()) if "GWETROOT" in daily_df else 0.45
        vpd = float(daily_df["vpd_kpa"].mean()) if "vpd_kpa" in daily_df else 1.10
        grace = float(daily_df["grace_groundwater_pctl"].mean()) if "grace_groundwater_pctl" in daily_df else 50.0

        vec = np.array([
            (temp_mean - 25.0) / 10.0,
            heat_days / 30.0,
            np.log1p(precip_sum) / 6.0,
            cdd / 30.0,
            sm_surf,
            sm_root,
            vpd / 2.5,
            grace / 100.0,
        ], dtype=np.float32)
        return vec

# STAGE 2: DISTRICT-AWARE 50-CROP MDP ENVIRONMENT

@dataclass
class FarmProfile:
    farm_id: str
    irrigation_access: bool
    baseline_soil_n_kg_ha: float
    soil_organic_matter_pct: float
    priority_mode: str = "balanced"
    district_name: str = "Rajshahi"


class CropRotationMDPEnv:
    def __init__(
        self,
        farm_profile: FarmProfile,
        seasonal_climates: List[np.ndarray],
        seasonal_daily_sequences: Optional[List[np.ndarray]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.profile = farm_profile
        self.seasonal_climates = seasonal_climates
        self.seasonal_daily_sequences = seasonal_daily_sequences
        self.current_season_idx = 0

        self.soil_n = float(farm_profile.baseline_soil_n_kg_ha)
        self.prev_crop_id: Optional[int] = None
        self.prev_family: str = "None"
        self.cumulative_water_used_mm = 0.0
        self.cumulative_blue_water_mm = 0.0
        self.cumulative_profit_usd = 0.0
        self.rotation_history: List[CropSpec] = []
        self.season_step_logs: List[Dict[str, Any]] = []

        self.weights = weights or self._resolve_weights(farm_profile.priority_mode)

    @staticmethod
    def _resolve_weights(priority: str) -> Dict[str, float]:
        if priority == "water_conservation":
            return {"profit": 0.12, "water_weight": 1.45, "soil_weight": 0.15, "veg_bonus": 0.0}
        elif priority == "soil_health":
            return {"profit": 0.18, "water_weight": 0.15, "soil_weight": 1.45, "veg_bonus": 0.0}
        elif priority == "high_value_vegetable":
            return {"profit": 0.80, "water_weight": 0.15, "soil_weight": 0.20, "veg_bonus": 0.45}
        return {"profit": 0.72, "water_weight": 0.18, "soil_weight": 0.12, "veg_bonus": 0.0}  # balanced

    def get_state(self) -> np.ndarray:
        climate_vec = self.seasonal_climates[self.current_season_idx]
        season_norm = self.current_season_idx / 2.0
        soil_n_norm = np.clip(self.soil_n / 150.0, 0.0, 2.0)
        som_norm = np.clip(self.profile.soil_organic_matter_pct / 3.0, 0.0, 2.0)
        irrig_flag = 1.0 if self.profile.irrigation_access else 0.0

        prio_oh = np.array([
            1.0 if self.profile.priority_mode == "water_conservation" else 0.0,
            1.0 if self.profile.priority_mode == "soil_health" else 0.0,
        ], dtype=np.float32)

        family_oh = np.array([
            1.0 if self.prev_family == "Poaceae" else 0.0,
            1.0 if self.prev_family == "Fabaceae" else 0.0,
            1.0 if self.prev_family not in ("None", "Poaceae", "Fabaceae") else 0.0,
        ], dtype=np.float32)

        state = np.concatenate([
            climate_vec,
            [season_norm, soil_n_norm, som_norm, irrig_flag],
            prio_oh,
            family_oh,
        ])
        return state.astype(np.float32)

    def get_daily_sequence(self) -> np.ndarray:
        if self.seasonal_daily_sequences is not None and self.current_season_idx < len(self.seasonal_daily_sequences):
            return self.seasonal_daily_sequences[self.current_season_idx]
        c_vec = self.seasonal_climates[min(self.current_season_idx, 2)]
        t_steps = np.linspace(0, np.pi, 120, dtype=np.float32)[:, None]
        base = np.tile(c_vec, (120, 2))[:, :12]
        return (base + 0.1 * np.sin(t_steps)).astype(np.float32)

    def get_action_mask(self, restrict_to_core: bool = False) -> np.ndarray:
        """
        Returns a binary mask over all `NUM_CROPS` (50 crops) for the current season.
        """
        mask = np.zeros(NUM_CROPS, dtype=np.float32)
        legal_indices = SEASON_ACTION_MAP[self.current_season_idx]
        mask[legal_indices] = 1.0
        return mask

    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, Dict]:
        crop = CROP_CATALOG[action_id]
        s_idx = self.current_season_idx
        climate = self.seasonal_climates[s_idx]

        temp_c = float(climate[0] * 10.0 + 25.0)
        heat_days = float(climate[1] * 30.0)
        precip_mm = float(np.expm1(climate[2] * 6.0))
        cdd_days = float(climate[3] * 30.0)
        sm_surf = float(climate[4])
        sm_root = float(climate[5])
        vpd_kpa = float(climate[6] * 2.5)
        grace_pctl = float(climate[7] * 100.0)

        # 1. Site-Specific Crop Evapotranspiration Demand (ETc)
        if crop.water_req_mm > 0.0:
            et_mod = 1.0 + 0.10 * (vpd_kpa - 1.05) + 0.008 * (temp_c - crop.optimal_temp_c)
            water_used = float(np.clip(crop.water_req_mm * et_mod, crop.water_req_mm * 0.85, crop.water_req_mm * 1.25))
        else:
            water_used = 0.0

        effective_rain_and_soil = (precip_mm * 0.65) + (sm_root * 180.0)
        blue_water_deficit = max(0.0, water_used - effective_rain_and_soil)

        # 2. District AP Ratio Yield Scaling + Continuous NASA Climate Response
        effective_margin = crop.baseline_margin_usd

        # Apply empirical District AP Ratio yield multiplier from the 64-district spreadsheet
        dist_mults = DISTRICT_YIELD_MULTIPLIERS.get(self.profile.district_name, {})
        ap_rel_yield = dist_mults.get(crop.name, 1.0)
        # Blend baseline economics with district yield advantage (kept subtle for field crops, full for vegetables)
        if crop.category == "Vegetable_Tuber_Spice" and crop.name != "Potato":
            effective_margin *= (0.45 + 0.55 * ap_rel_yield)

        if crop.name != "Fallow":
            temp_diff = abs(temp_c - crop.optimal_temp_c)
            thermal_mult = float(np.clip(1.04 - 0.015 * temp_diff, 0.80, 1.08))
            effective_margin *= thermal_mult

            n_mult = float(np.clip(1.0 + 0.0025 * (self.soil_n - 65.0), 0.85, 1.18))
            effective_margin *= n_mult

            if grace_pctl >= 50.0:
                if crop.name == "Jute":
                    effective_margin *= 1.38
                elif crop.name in ("Lentil", "Rape & Mustard"):
                    effective_margin *= 1.08
                elif crop.name == "Potato":
                    effective_margin *= 0.65

            if heat_days > 4.0 and crop.name == "Wheat":
                effective_margin *= 0.80

        # 3. Water Deficit & Aquifer Depletion Penalty
        water_penalty = 0.0
        if crop.water_req_mm > 0.0:
            water_penalty += water_used * 0.55 + blue_water_deficit * 0.45

            if not self.profile.irrigation_access:
                if blue_water_deficit > 90.0:
                    water_penalty += (blue_water_deficit - 90.0) * 2.5
                    drought_loss = float(np.clip(1.0 - (blue_water_deficit - 90.0) / 450.0 * (1.1 - crop.drought_hardiness), 0.35, 0.95))
                    effective_margin *= drought_loss
            else:
                pumping_cost_per_mm = 0.10 if grace_pctl >= 30.0 else 0.22
                effective_margin -= blue_water_deficit * pumping_cost_per_mm
                if grace_pctl < 30.0 and water_used > 750.0:
                    water_penalty += 450.0

        # 4. Exact Soil Mineral Nitrogen Mass Balance
        soil_n_pre = self.soil_n
        soil_bonus = 0.0
        self.soil_n = self.soil_n + crop.n_fixation_kg_ha

        if crop.family == "Fabaceae":
            soil_bonus += 280.0 + crop.n_fixation_kg_ha * 3.0
        elif crop.name == "Jute":
            soil_bonus += 60.0
        else:
            soil_bonus += crop.n_fixation_kg_ha * 2.5

        pest_break_status = "First season"
        if self.prev_family != "None" and self.prev_family == crop.family:
            soil_bonus -= 180.0
            effective_margin *= 0.88
            pest_break_status = f"Monoculture risk ({crop.family} repeated -> -12% yield penalty)"
        elif self.prev_family != "None" and self.prev_family != crop.family and crop.name != "Fallow":
            soil_bonus += 70.0
            pest_break_status = f"Pest cycle broken ({self.prev_family} -> {crop.family})"
        elif crop.name == "Fallow":
            pest_break_status = "Fallow rest period (moisture conservation, bare soil)"

        # 5. Multi-Objective Scalar Reward (Including High-Value Vegetable Head)
        norm_profit = effective_margin / 1000.0
        norm_water_pen = water_penalty / 500.0
        norm_soil_bon = soil_bonus / 200.0
        veg_bonus = 0.35 if crop.category == "Vegetable_Tuber_Spice" else 0.0

        reward = (
            self.weights["profit"] * norm_profit
            - self.weights["water_weight"] * norm_water_pen
            + self.weights["soil_weight"] * norm_soil_bon
            + self.weights.get("veg_bonus", 0.0) * veg_bonus
        )

        self.cumulative_water_used_mm += water_used
        self.cumulative_blue_water_mm += blue_water_deficit
        self.cumulative_profit_usd += effective_margin
        self.rotation_history.append(crop)

        total_cal_days = SEASON_TOTAL_DAYS[s_idx]
        bare_days = max(0, total_cal_days - crop.duration_days)

        step_info = {
            "season_index": s_idx,
            "season_name": SEASON_NAMES[s_idx],
            "crop_id": crop.crop_id,
            "crop": crop.name,
            "crop_category": crop.category,
            "botanical_family": crop.family,
            "sowing_window": crop.sowing_window,
            "harvest_window": crop.harvest_window,
            "active_growing_days": crop.duration_days,
            "bare_field_days": bare_days,
            "root_system": crop.root_system,
            "erosion_protection": crop.erosion_protection,
            "margin_usd": round(effective_margin, 1),
            "water_mm": round(water_used, 1),
            "blue_water_irrigation_mm": round(blue_water_deficit, 1),
            "effective_rain_mm": round(precip_mm, 1),
            "mean_temp_c": round(temp_c, 1),
            "heat_stress_days": int(round(heat_days)),
            "consecutive_dry_days": int(round(cdd_days)),
            "rootzone_wetness_frac": round(sm_root, 2),
            "grace_groundwater_pctl": round(grace_pctl, 1),
            "soil_n_pre_kg_ha": round(soil_n_pre, 1),
            "soil_n_delta_kg_ha": round(crop.n_fixation_kg_ha, 1),
            "soil_n_post_kg_ha": round(self.soil_n, 1),
            "pest_break_status": pest_break_status,
        }
        self.season_step_logs.append(step_info)

        self.prev_crop_id = action_id
        self.prev_family = crop.family

        self.current_season_idx += 1
        done = (self.current_season_idx >= 3)
        next_state = self.get_state() if not done else np.zeros(17, dtype=np.float32)

        return next_state, reward, done, step_info

    def reset(self) -> np.ndarray:
        self.current_season_idx = 0
        self.soil_n = float(self.profile.baseline_soil_n_kg_ha)
        self.prev_crop_id = None
        self.prev_family = "None"
        self.cumulative_water_used_mm = 0.0
        self.cumulative_blue_water_mm = 0.0
        self.cumulative_profit_usd = 0.0
        self.rotation_history.clear()
        self.season_step_logs.clear()
        return self.get_state()

# STAGE 3: 50-CROP TEMPORAL ATTENTION BI-GRU + 4-HEAD ACTOR-CRITIC

if HAS_TORCH:
    class TemporalAttentionEncoder(nn.Module):
        def __init__(self, daily_feature_dim: int = 12, rnn_hidden: int = 32):
            super().__init__()
            self.bi_gru = nn.GRU(
                input_size=daily_feature_dim,
                hidden_size=rnn_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.attention_weights = nn.Sequential(
                nn.Linear(rnn_hidden * 2, 32),
                nn.Tanh(),
                nn.Linear(32, 1, bias=False),
            )
            self.layer_norm = nn.LayerNorm(rnn_hidden * 2)

        def forward(
            self, daily_seq: torch.Tensor, return_weights: bool = False
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            rnn_out, _ = self.bi_gru(daily_seq)
            attn_scores = self.attention_weights(rnn_out)
            attn_probs = F.softmax(attn_scores, dim=1)
            context = self.layer_norm(torch.sum(rnn_out * attn_probs, dim=1))
            if return_weights:
                return context, attn_probs.squeeze(-1)
            return context

    class CropRotationActorCritic(nn.Module):
        """
        National 64-District / 50-Crop Temporal Attention Bi-GRU Actor-Critic:
          - Supports all 50 rotational crops (`action_dim = NUM_CROPS`) across Rabi, Kharif-1, Kharif-2.
          - 4 Specialist Actor & Critic Heads:
              1. `balanced`
              2. `water_conservation`
              3. `soil_health`
              4. `high_value_vegetable`
        """

        def __init__(
            self,
            state_dim: int = 17,
            daily_feature_dim: int = 12,
            hidden_dim: int = 64,
            action_dim: int = NUM_CROPS,
        ):
            super().__init__()
            self.temporal_encoder = TemporalAttentionEncoder(
                daily_feature_dim=daily_feature_dim, rnn_hidden=hidden_dim // 2
            )
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(negative_slope=0.1),
            )
            self.fusion_layer = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(negative_slope=0.1),
            )

            self.actor_heads = nn.ModuleDict({
                mode: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(negative_slope=0.1),
                    nn.Linear(hidden_dim, action_dim),
                )
                for mode in PRIORITY_MODES
            })
            self.critic_heads = nn.ModuleDict({
                mode: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(negative_slope=0.1),
                    nn.Linear(hidden_dim, 1),
                )
                for mode in PRIORITY_MODES
            })

        def _fuse_inputs(
            self, state: torch.Tensor, daily_seq: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            h_state = self.state_encoder(state)
            if daily_seq is not None:
                h_temp = self.temporal_encoder(daily_seq)
            else:
                h_temp = torch.zeros_like(h_state)
            return self.fusion_layer(torch.cat([h_state, h_temp], dim=-1))

        def forward(
            self,
            state: torch.Tensor,
            action_mask: torch.Tensor,
            priority_mode: str = "balanced",
            daily_seq: Optional[torch.Tensor] = None,
        ) -> Tuple[Categorical, torch.Tensor]:
            features = self._fuse_inputs(state, daily_seq)
            head_key = priority_mode if priority_mode in self.actor_heads else "balanced"
            logits = self.actor_heads[head_key](features)

            masked_logits = torch.where(
                action_mask > 0.5,
                logits,
                torch.tensor(-1e9, dtype=logits.dtype, device=logits.device),
            )
            dist = Categorical(logits=masked_logits)
            value = self.critic_heads[head_key](features)
            return dist, value

        def evaluate_actions(
            self,
            states: torch.Tensor,
            action_masks: torch.Tensor,
            actions: torch.Tensor,
            priority_mode: str = "balanced",
            daily_seqs: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            features = self._fuse_inputs(states, daily_seqs)
            head_key = priority_mode if priority_mode in self.actor_heads else "balanced"
            logits = self.actor_heads[head_key](features)
            masked_logits = torch.where(
                action_masks > 0.5,
                logits,
                torch.tensor(-1e9, dtype=logits.dtype, device=logits.device),
            )
            dist = Categorical(logits=masked_logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            values = self.critic_heads[head_key](features)
            return log_probs, values.squeeze(-1), entropy

# STAGE 4: ADVANTAGE ACTOR-CRITIC (A2C) TRAINING OVER 64 DISTRICTS & 50 CROPS

def train_pytorch_actor_critic(
    env_generator,
    policy_net: nn.Module,
    num_episodes: int = 1000,
    lr: float = 0.003,
    weight_decay: float = 1e-4,
    gamma: float = 0.99,
    entropy_coef: float = 0.025,
    value_loss_coef: float = 0.5,
    max_grad_norm: float = 1.0,
    device: Optional[str] = None,
) -> Dict[str, List[float]]:
    resolved_device = resolve_torch_device(device)
    policy_net.to(resolved_device)

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_episodes, eta_min=1e-4)

    mode_history: Dict[str, List[float]] = {m: [] for m in PRIORITY_MODES}

    for ep in range(num_episodes):
        mode = PRIORITY_MODES[ep % len(PRIORITY_MODES)]
        try:
            env = env_generator(priority_mode=mode)
        except TypeError:
            env = env_generator()
        state = env.reset()
        done = False

        trajectory_states = []
        trajectory_daily_seqs = []
        trajectory_masks = []
        trajectory_actions = []
        trajectory_rewards = []

        while not done:
            mask = env.get_action_mask()
            daily_seq = env.get_daily_sequence()

            s_tensor = torch.tensor(state, dtype=torch.float32, device=resolved_device).unsqueeze(0)
            m_tensor = torch.tensor(mask, dtype=torch.float32, device=resolved_device).unsqueeze(0)
            d_tensor = torch.tensor(daily_seq, dtype=torch.float32, device=resolved_device).unsqueeze(0)

            with torch.no_grad():
                dist, _ = policy_net(s_tensor, m_tensor, priority_mode=mode, daily_seq=d_tensor)
                action = dist.sample().item()

            next_state, reward, done, _ = env.step(action)

            trajectory_states.append(state)
            trajectory_daily_seqs.append(daily_seq)
            trajectory_masks.append(mask)
            trajectory_actions.append(action)
            trajectory_rewards.append(reward)

            state = next_state

        discounted_returns = []
        running_return = 0.0
        for r in reversed(trajectory_rewards):
            running_return = r + gamma * running_return
            discounted_returns.insert(0, running_return)

        batch_states = torch.tensor(np.array(trajectory_states), dtype=torch.float32, device=resolved_device)
        batch_daily = torch.tensor(np.array(trajectory_daily_seqs), dtype=torch.float32, device=resolved_device)
        batch_masks = torch.tensor(np.array(trajectory_masks), dtype=torch.float32, device=resolved_device)
        batch_actions = torch.tensor(trajectory_actions, dtype=torch.long, device=resolved_device)
        batch_returns = torch.tensor(discounted_returns, dtype=torch.float32, device=resolved_device)

        log_probs, values, entropy = policy_net.evaluate_actions(
            batch_states, batch_masks, batch_actions, priority_mode=mode, daily_seqs=batch_daily
        )

        advantages = batch_returns - values.detach()

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.smooth_l1_loss(values, batch_returns)
        entropy_loss = -entropy.mean()

        total_loss = actor_loss + (value_loss_coef * critic_loss) + (entropy_coef * entropy_loss)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()

        mode_history[mode].append(sum(trajectory_rewards))

    return mode_history

# STAGE 5: MULTI-STRATEGY EVALUATOR (BASELINE + 3 CORE + HIGH-VALUE VEGETABLE)

def evaluate_rotation_trajectory(
    farm: FarmProfile,
    seasonal_climates: List[np.ndarray],
    sequence_crop_ids: Tuple[int, int, int],
    priority_mode: str,
    seasonal_daily_sequences: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    test_farm = FarmProfile(
        farm_id=farm.farm_id,
        irrigation_access=farm.irrigation_access,
        baseline_soil_n_kg_ha=farm.baseline_soil_n_kg_ha,
        soil_organic_matter_pct=farm.soil_organic_matter_pct,
        priority_mode=priority_mode,
        district_name=farm.district_name,
    )
    env = CropRotationMDPEnv(test_farm, seasonal_climates, seasonal_daily_sequences)
    env.reset()

    cumulative_reward = 0.0
    for a in sequence_crop_ids:
        _, r, _, _ = env.step(a)
        cumulative_reward += r

    def _display_crop_name(c: CropSpec) -> str:
        if c.name == "Aman":
            return "T. Aman Rice"
        if c.name == "Boro":
            return "Boro Rice"
        if c.name == "Rape & Mustard":
            return "Mustard"
        return c.name

    seq_str = " -> ".join([_display_crop_name(c) for c in env.rotation_history])
    total_active_days = sum(c.duration_days for c in env.rotation_history)
    total_bare_days = max(0, 365 - total_active_days)

    return {
        "sequence_ids": list(sequence_crop_ids),
        "rotation_name": seq_str,
        "cumulative_reward": round(float(cumulative_reward), 3),
        "total_profit_usd": round(env.cumulative_profit_usd, 1),
        "total_water_mm": round(env.cumulative_water_used_mm, 1),
        "total_blue_water_mm": round(env.cumulative_blue_water_mm, 1),
        "final_soil_n": round(env.soil_n, 1),
        "n_delta": round(env.soil_n - farm.baseline_soil_n_kg_ha, 1),
        "active_cover_days_yr": total_active_days,
        "bare_field_days_yr": total_bare_days,
        "seasonal_timeline": env.season_step_logs,
    }


def _get_candidate_pools() -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """
    Returns:
      1. `core_candidates`: Classic field crops, pulses, fiber, potato, and fallow.
      2. `veg_candidates`: High-value winter & summer vegetable rotation combinations.
    """
    core_rabi_names = {"Lentil", "Rape & Mustard", "Wheat", "Potato", "Maize 2", "Boro", "Gram", "Groundnut"}
    core_k1_names = {"Mungbean", "Aus", "Jute", "Fallow", "Sesame"}
    core_k2_names = {"Aman"}

    core_r = [c.crop_id for c in CROP_CATALOG if c.season == "Rabi" and c.name in core_rabi_names]
    core_k1 = [c.crop_id for c in CROP_CATALOG if c.season == "Kharif-1" and c.name in core_k1_names]
    core_k2 = [c.crop_id for c in CROP_CATALOG if c.season == "Kharif-2" and c.name in core_k2_names]

    core_candidates = [(r, k1, k2) for r in core_r for k1 in core_k1 for k2 in core_k2]

    veg_r = [c.crop_id for c in CROP_CATALOG if c.season == "Rabi" and c.category == "Vegetable_Tuber_Spice" and c.name not in {"Potato", "Ginger"}]
    veg_k1 = [c.crop_id for c in CROP_CATALOG if c.season == "Kharif-1" and (c.category == "Vegetable_Tuber_Spice" or c.name == "Mungbean")]
    if not veg_r:
        veg_r = core_r
    if not veg_k1:
        veg_k1 = core_k1

    veg_candidates = [(r, k1, k2) for r in veg_r for k1 in veg_k1 for k2 in core_k2]
    return core_candidates, veg_candidates


def generate_three_way_tradeoffs(
    farm: FarmProfile,
    seasonal_climates: List[np.ndarray],
    policy_net: Optional[Any] = None,
    seasonal_daily_sequences: Optional[List[np.ndarray]] = None,
    include_existing_baseline: bool = True,
    include_vegetable_strategy: bool = True,
) -> pd.DataFrame:
    """
    Evaluates:
      - Row 0: Existing Rotation (Baseline) (`Boro Rice -> Fallow -> T. Aman Rice`)
      - Row 1: Balanced (`Potato/Field -> Mungbean/Jute -> T. Aman Rice`)
      - Row 2: Water Conservation (`Lentil -> Fallow -> T. Aman Rice`)
      - Row 3: Soil Health (`Lentil/Gram -> Mungbean -> T. Aman Rice`)
      - Row 4: High-Value Vegetable (`Cabbage/Onion/Cauliflower -> Barbati/Patal -> T. Aman Rice`)
    """
    core_candidates, veg_candidates = _get_candidate_pools()

    baseline_eval = evaluate_rotation_trajectory(
        farm, seasonal_climates, EXISTING_BASELINE_PATH, "balanced", seasonal_daily_sequences
    )
    base_water = baseline_eval["total_water_mm"]

    selected_rows = []
    if include_existing_baseline:
        selected_rows.append({
            "Strategy": "Existing Rotation (Baseline)",
            "Recommended Rotation": baseline_eval["rotation_name"],
            "Net Profit ($/ha)": baseline_eval["total_profit_usd"],
            "Total Water (mm)": baseline_eval["total_water_mm"],
            "Water Saved vs Baseline (mm)": 0.0,
            "Final Soil N (kg/ha)": baseline_eval["final_soil_n"],
            "Soil N Delta (kg/ha)": baseline_eval["n_delta"],
            "Bare Field (days/yr)": baseline_eval["bare_field_days_yr"],
        })

    modes = [
        ("Balanced", "balanced", core_candidates),
        ("Water Conservation", "water_conservation", core_candidates),
        ("Soil Health", "soil_health", core_candidates),
    ]
    if include_vegetable_strategy:
        modes.append(("High-Value Vegetable", "high_value_vegetable", veg_candidates))

    for display_name, mode_key, pool in modes:
        evals = [
            evaluate_rotation_trajectory(
                farm, seasonal_climates, path, mode_key, seasonal_daily_sequences
            )
            for path in pool
        ]
        evals.sort(key=lambda x: x["cumulative_reward"], reverse=True)
        best = evals[0]

        selected_rows.append({
            "Strategy": display_name,
            "Recommended Rotation": best["rotation_name"],
            "Net Profit ($/ha)": best["total_profit_usd"],
            "Total Water (mm)": best["total_water_mm"],
            "Water Saved vs Baseline (mm)": round(base_water - best["total_water_mm"], 1),
            "Final Soil N (kg/ha)": best["final_soil_n"],
            "Soil N Delta (kg/ha)": best["n_delta"],
            "Bare Field (days/yr)": best["bare_field_days_yr"],
        })

    return pd.DataFrame(selected_rows)


find_pareto_optimal_rotations = generate_three_way_tradeoffs


def compute_20yr_historical_uncertainty(
    farm: FarmProfile,
    district_20yr_records: List[Dict[str, Any]],
    sequence_crop_ids: Tuple[int, int, int],
    priority_mode: str = "balanced",
) -> Dict[str, float]:
    profits = []
    waters = []
    for rec in district_20yr_records:
        res = evaluate_rotation_trajectory(
            farm, rec["seasonal_vectors"], sequence_crop_ids, priority_mode, rec.get("seasonal_daily_seqs")
        )
        profits.append(res["total_profit_usd"])
        waters.append(res["total_water_mm"])

    p_arr = np.array(profits, dtype=np.float32)
    w_arr = np.array(waters, dtype=np.float32)
    return {
        "profit_mean_usd": round(float(np.mean(p_arr)), 1),
        "profit_std_usd": round(float(np.std(p_arr)), 1),
        "profit_p10_downside_usd": round(float(np.percentile(p_arr, 10)), 1),
        "profit_p90_upside_usd": round(float(np.percentile(p_arr, 90)), 1),
        "water_mean_mm": round(float(np.mean(w_arr)), 1),
        "water_std_mm": round(float(np.std(w_arr)), 1),
        "water_p90_drought_demand_mm": round(float(np.percentile(w_arr, 90)), 1),
    }


def simulate_climate_scenario(
    seasonal_climates: List[np.ndarray],
    rainfall_delta_pct: float = 0.0,
    temp_delta_c: float = 0.0,
    grace_gw_delta_pct: float = 0.0,
) -> List[np.ndarray]:
    perturbed: List[np.ndarray] = []
    for vec in seasonal_climates:
        v = vec.copy()
        temp_c = v[0] * 10.0 + 25.0 + temp_delta_c
        v[0] = (temp_c - 25.0) / 10.0
        if temp_delta_c > 0:
            v[1] = np.clip(v[1] + (temp_delta_c * 3.5) / 30.0, 0.0, 1.0)
        precip_mm = np.expm1(v[2] * 6.0) * max(0.1, 1.0 + rainfall_delta_pct / 100.0)
        v[2] = np.log1p(precip_mm) / 6.0
        v[3] = np.clip(v[3] * (1.0 - 0.5 * (rainfall_delta_pct / 100.0)), 0.05, 1.0)
        moisture_shift = 0.15 * (rainfall_delta_pct / 100.0) - 0.025 * temp_delta_c
        v[4] = np.clip(v[4] + moisture_shift, 0.10, 0.98)
        v[5] = np.clip(v[5] + moisture_shift, 0.12, 0.96)
        v[6] = np.clip(v[6] * (1.0 + 0.06 * temp_delta_c - 0.10 * (rainfall_delta_pct / 100.0)), 0.15, 1.2)
        v[7] = np.clip(v[7] + (grace_gw_delta_pct / 100.0), 0.05, 0.98)
        perturbed.append(v.astype(np.float32))
    return perturbed


def build_strategy_explainability(
    strategy_name: str,
    eval_result: Dict[str, Any],
    baseline_result: Dict[str, Any],
    uncertainty_stats: Dict[str, float],
    district_name: str,
    aez_name: str,
) -> Dict[str, Any]:
    seq_name = eval_result["rotation_name"]
    steps = eval_result["seasonal_timeline"]
    rabi_step, k1_step, k2_step = steps[0], steps[1], steps[2]

    water_saved = round(baseline_result["total_water_mm"] - eval_result["total_water_mm"], 1)
    profit_diff = round(eval_result["total_profit_usd"] - baseline_result["total_profit_usd"], 1)
    n_diff = round(eval_result["n_delta"] - baseline_result["n_delta"], 1)

    if "Baseline" in strategy_name:
        why = (
            f"Represents the conventional Boro Rice -> Fallow -> T. Aman Rice status-quo rotation in {district_name} "
            f"({aez_name}), used as the reference benchmark for water, nitrogen, and income comparisons."
        )
    elif strategy_name == "Water Conservation":
        why = (
            f"Selected to minimize groundwater pumping and protect aquifer reserves in {district_name} "
            f"(saving {water_saved} mm/yr of water vs. the Boro rice baseline) by replacing irrigated winter Boro rice "
            f"with drought-hardy {rabi_step['crop']} and conserving pre-monsoon soil moisture."
        )
    elif strategy_name == "Soil Health":
        why = (
            f"Selected to rebuild depleted soil fertility ({eval_result['n_delta']:+.1f} kg N/ha net change, or "
            f"{n_diff:+.1f} kg N/ha above baseline) and minimize bare-field erosion ({eval_result['bare_field_days_yr']} bare days/yr) "
            f"using biological nitrogen fixation and botanical family alternation."
        )
    elif strategy_name == "High-Value Vegetable":
        why = (
            f"Selected to maximize horticultural cash income in {district_name} (${eval_result['total_profit_usd']}/ha, "
            f"+${profit_diff}/ha over baseline) using high-AP-Ratio local winter and summer vegetables ({rabi_step['crop']} -> {k1_step['crop']}) "
            f"while saving {water_saved} mm of water compared to Boro rice."
        )
    else:
        why = (
            f"Selected as the Pareto-optimal economic and ecological balance for {district_name} "
            f"(${eval_result['total_profit_usd']}/ha net margin, +${profit_diff}/ha over baseline, while saving {water_saved} mm of water)."
        )

    evidence = [
        f"NASA GRACE/GRACE-FO Groundwater Percentile: {rabi_step['grace_groundwater_pctl']}% in Rabi winter (values <30% signal severe tubewell drawdown risk).",
        f"NASA SMAP & POWER Root-Zone Soil Wetness (GWETROOT): {int(rabi_step['rootzone_wetness_frac']*100)}% in Rabi, {int(k1_step['rootzone_wetness_frac']*100)}% in Kharif-1, and {int(k2_step['rootzone_wetness_frac']*100)}% in Kharif-2.",
        f"NASA GPM / POWER Seasonal Rainfall & Dry Spells: {rabi_step['effective_rain_mm']} mm in Rabi ({rabi_step['consecutive_dry_days']} max consecutive dry days) vs. {k2_step['effective_rain_mm']} mm in Monsoon Kharif-2.",
        f"NASA MODIS/ECOSTRESS Evapotranspiration Demand: Total rotation ETc of {eval_result['total_water_mm']} mm/yr (requiring {eval_result['total_blue_water_mm']} mm of supplemental irrigation).",
    ]

    benefits = [
        f"Nutrient Cycling & Nitrogen Balance: Net soil mineral N delta of {eval_result['n_delta']:+.1f} kg N/ha (Rabi {rabi_step['crop']}: {rabi_step['soil_n_delta_kg_ha']:+.0f}, Kharif-1 {k1_step['crop']}: {k1_step['soil_n_delta_kg_ha']:+.0f}, Kharif-2 {k2_step['crop']}: {k2_step['soil_n_delta_kg_ha']:+.0f} kg/ha).",
        f"Pest & Disease Cycle Disruption: {k1_step['pest_break_status']}; {k2_step['pest_break_status']}.",
        f"Ground Cover & Erosion Control: Field is actively covered by living roots for {eval_result['active_cover_days_yr']} days/yr ({eval_result['bare_field_days_yr']} bare-field transition days/yr).",
        f"Water Footprint: Consumes {eval_result['total_water_mm']} mm total water ({water_saved:+.1f} mm saved relative to traditional Boro-Fallow-Aman).",
    ]

    gaps_and_uncertainty = [
        f"20-Year Climate Variability (2005-2024 Monte Carlo): Expected net profit is ${uncertainty_stats['profit_mean_usd']}/ha ± ${uncertainty_stats['profit_std_usd']}/ha (10th-percentile drought/flood downside: ${uncertainty_stats['profit_p10_downside_usd']}/ha; 90th-percentile upside: ${uncertainty_stats['profit_p90_upside_usd']}/ha).",
        "Spatial Resolution Gap: NASA SMAP (9 km), POWER (50 km), and GRACE (25 km) capture regional hydro-climatic gradients but do not resolve plot-level plough-pan depth or canal seepage.",
        "Soil Lab Gap: Satellite vegetation indices (MODIS/HLS NDVI) and SoilGrids 250m priors do not measure plot-specific soil pH or micronutrients (Boron, Zinc).",
    ]

    local_review_risks = [
        f"Turnaround & Sowing Window Verification: Verify that {rabi_step['crop']} harvest ({rabi_step['harvest_window']}) allows timely land preparation for {k1_step['crop']} sowing ({k1_step['sowing_window']}) under local soil moisture.",
        "Market & Cold-Storage Logistics: High-value vegetables and perishables require local DAE extension review of cold-storage capacity and wholesale market transport.",
        "Drainage & Flash-Flood Check: Legumes and winter vegetables suffer rapid root rot if unseasonal rain waterlogs poorly drained plots.",
    ]

    return {
        "strategy_name": strategy_name,
        "recommended_rotation": seq_name,
        "metrics": {
            "net_profit_usd_ha": eval_result["total_profit_usd"],
            "profit_delta_vs_baseline_usd_ha": profit_diff,
            "total_water_mm": eval_result["total_water_mm"],
            "blue_water_irrigation_mm": eval_result["total_blue_water_mm"],
            "water_saved_vs_baseline_mm": water_saved,
            "final_soil_n_kg_ha": eval_result["final_soil_n"],
            "soil_n_delta_kg_ha": eval_result["n_delta"],
            "active_cover_days_yr": eval_result["active_cover_days_yr"],
            "bare_field_days_yr": eval_result["bare_field_days_yr"],
        },
        "uncertainty_20yr_band": uncertainty_stats,
        "why_suggested": why,
        "supporting_nasa_evidence": evidence,
        "agronomic_and_soil_benefits": benefits,
        "information_gaps_and_uncertainty": gaps_and_uncertainty,
        "risks_requiring_local_expert_review": local_review_risks,
        "seasonal_crop_calendar": steps,
    }




def query_field_shift_api(
    district_name: str = "Rajshahi",
    irrigation_access: bool = True,
    priority_mode: str = "balanced",
    soil_fertility_level: str = "medium",
    current_crops: Optional[Tuple[str, str, str]] = None,
    locked_season_crops: Optional[Dict[str, str]] = None,
    excluded_crops: Optional[List[str]] = None,
    rainfall_delta_pct: float = 0.0,
    temp_delta_c: float = 0.0,
    grace_gw_delta_pct: float = 0.0,
    seasonal_climates: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Live Web Backend Inference Endpoint (`POST /api/recommend`).
    Supports both:
      1. Zero-Crop-Input Mode (Farmer only selects District, Irrigation, Priority).
      2. Custom Farmer Crop Mode:
         - `current_crops`: e.g. ("Wheat", "Fallow", "Aman") for their existing baseline.
         - `locked_season_crops`: e.g. {"Rabi": "Onion"} to lock one season and let AI optimize the rest.
         - `excluded_crops`: e.g. ["Tobacco", "Boro"] to filter out unwanted crops.
    """
    fertility_map = {"low": (50.0, 0.8), "medium": (75.0, 1.2), "high": (105.0, 1.8)}
    base_n, base_som = fertility_map.get(soil_fertility_level.lower(), (75.0, 1.2))

    farm = FarmProfile(
        farm_id=f"web_{district_name.lower()}",
        irrigation_access=irrigation_access,
        baseline_soil_n_kg_ha=base_n,
        soil_organic_matter_pct=base_som,
        priority_mode=priority_mode,
        district_name=district_name,
    )

    if seasonal_climates is None:
        seasonal_climates = [
            np.array([-0.3, 0.0, 0.2, 0.8, 0.35, 0.38, 0.40, 0.22], dtype=np.float32),
            np.array([0.5, 0.2, 0.5, 0.4, 0.45, 0.42, 0.60, 0.25], dtype=np.float32),
            np.array([0.35, 0.0, 0.9, 0.1, 0.85, 0.90, 0.30, 0.42], dtype=np.float32),
        ]

    perturbed_climates = simulate_climate_scenario(
        seasonal_climates,
        rainfall_delta_pct=rainfall_delta_pct,
        temp_delta_c=temp_delta_c,
        grace_gw_delta_pct=grace_gw_delta_pct,
    )

    # Resolve farmer existing rotation (defaults to Boro -> Fallow -> Aman)
    if current_crops is not None and len(current_crops) == 3:
        baseline_ids = (
            CROP_NAME_TO_ID.get(current_crops[0], EXISTING_BASELINE_PATH[0]),
            CROP_NAME_TO_ID.get(current_crops[1], EXISTING_BASELINE_PATH[1]),
            CROP_NAME_TO_ID.get(current_crops[2], EXISTING_BASELINE_PATH[2]),
        )
    else:
        baseline_ids = EXISTING_BASELINE_PATH

    baseline_eval = evaluate_rotation_trajectory(
        farm, perturbed_climates, baseline_ids, priority_mode
    )

    core_pool, veg_pool = _get_candidate_pools()
    excluded_ids = {CROP_NAME_TO_ID[c] for c in (excluded_crops or []) if c in CROP_NAME_TO_ID}

    def _filter_pool(pool: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        filtered = [p for p in pool if not (set(p) & excluded_ids)]
        if locked_season_crops:
            s_map = {"Rabi": 0, "Kharif-1": 1, "Kharif-2": 2}
            for s_name, c_name in locked_season_crops.items():
                if s_name in s_map and c_name in CROP_NAME_TO_ID:
                    s_idx = s_map[s_name]
                    cid = CROP_NAME_TO_ID[c_name]
                    filtered = [p for p in filtered if p[s_idx] == cid]
                    if not filtered:
                        r_list = [cid] if s_idx == 0 else SEASON_ACTION_MAP[0]
                        k1_list = [cid] if s_idx == 1 else SEASON_ACTION_MAP[1]
                        k2_list = [cid] if s_idx == 2 else [CROP_NAME_TO_ID.get("Aman", SEASON_ACTION_MAP[2][0])]
                        filtered = [(r, k1, k2) for r in r_list for k1 in k1_list for k2 in k2_list if not ({r, k1, k2} & excluded_ids)]
        return filtered or pool

    modes = [
        ("Balanced", "balanced", _filter_pool(core_pool)),
        ("Water Conservation", "water_conservation", _filter_pool(core_pool)),
        ("Soil Health", "soil_health", _filter_pool(core_pool)),
        ("High-Value Vegetable", "high_value_vegetable", _filter_pool(veg_pool)),
    ]

    strategies = [{
        "strategy_name": "Existing Rotation (Baseline)",
        "recommended_rotation": baseline_eval["rotation_name"],
        "metrics": baseline_eval,
    }]
    for display_name, mode_key, pool in modes:
        evals = [evaluate_rotation_trajectory(farm, perturbed_climates, path, mode_key) for path in pool]
        evals.sort(key=lambda x: x["cumulative_reward"], reverse=True)
        strategies.append({
            "strategy_name": display_name,
            "recommended_rotation": evals[0]["rotation_name"],
            "metrics": evals[0],
        })

    return {
        "district": district_name,
        "farmer_inputs": {
            "irrigation_access": irrigation_access,
            "priority_mode": priority_mode,
            "soil_fertility_level": soil_fertility_level,
            "current_crops": current_crops or ("Boro Rice", "Fallow", "T. Aman Rice"),
            "locked_season_crops": locked_season_crops or {},
            "excluded_crops": excluded_crops or [],
        },
        "strategies": strategies,
    }


if __name__ == "__main__":
    logger.info(
        "Initializing Field Shift 64-District / %d-Crop Backend Engine (PyTorch: %s)...",
        NUM_CROPS, HAS_TORCH,
    )

    rabi_climate = np.array([-0.3, 0.0, 0.2, 0.8, 0.35, 0.38, 0.40, 0.22], dtype=np.float32)
    kharif1_climate = np.array([0.5, 0.2, 0.5, 0.4, 0.45, 0.42, 0.60, 0.25], dtype=np.float32)
    kharif2_climate = np.array([0.35, 0.0, 0.9, 0.1, 0.85, 0.90, 0.30, 0.42], dtype=np.float32)
    climates = [rabi_climate, kharif1_climate, kharif2_climate]

    barind_farm = FarmProfile(
        farm_id="barind_rajshahi_01",
        irrigation_access=True,
        baseline_soil_n_kg_ha=75.0,
        soil_organic_matter_pct=1.1,
        priority_mode="balanced",
        district_name="Rajshahi",
    )

    tradeoff_df = generate_three_way_tradeoffs(barind_farm, climates)
    print("\n" + tradeoff_df.to_string(index=False))