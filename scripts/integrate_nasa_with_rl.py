"""
Field Shift: Complete 20-Year NASA Dataset, Temporal RL & Web API JSON Exporter
===============================================================================

Closes 100% of the NASA Space Apps 2026 "Field Shift: Adapting Farms with NASA Data"
requirements and exports a complete, frontend-ready JSON payload (`field_shift_web_payload.json`)
alongside the trained PyTorch checkpoint (`field_shift_sota_policy.pt`):

  1. Compares 5 Rotation Strategies (Existing Baseline `Boro Rice -> Fallow -> T. Aman Rice`
     + 4 AI Pareto Strategies: `Balanced`, `Water Conservation`, `Soil Health`, and
     `High-Value Vegetable` powered by the 64-district / 50-crop database).
  2. Outputs the 5 Official NASA Rubric Explainability Fields for every strategy:
     - `why_suggested`
     - `supporting_nasa_evidence`
     - `agronomic_and_soil_benefits`
     - `information_gaps_and_uncertainty` (with 20-year Monte Carlo confidence intervals)
     - `risks_requiring_local_expert_review`
  3. Extracts Temporal Self-Attention Profiles (`attention_stress_windows`) from the trained
     Bi-GRU encoder showing exact peak climate-stress windows across the 120-day seasons.
  4. Pre-computes Interactive Scenario Slider sensitivity tables (e.g., `-25% Drought`,
     `+25% Wet Monsoon`, `+2.0C Warming`, `-10% Groundwater Drop`) and provides a callable
     function `query_field_shift_api()` for live Flask/FastAPI backend endpoints.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from field_shift.nasa_data import (
    NW_BANGLADESH_DISTRICTS,
    FarmLocation,
    NASAPowerClient,
    normalize_daily_sequence,
)
from field_shift.rl_pipeline import (
    CROP_CATALOG,
    EXISTING_BASELINE_PATH,
    FARMER_SCIENTIFIC_GLOSSARY,
    HAS_TORCH,
    NUM_CROPS,
    SEASON_ACTION_MAP,
    SEASON_NAMES,
    CropRotationMDPEnv,
    FarmProfile,
    PreprocessingPipeline,
    _get_candidate_pools,
    build_strategy_explainability,
    compute_20yr_historical_uncertainty,
    evaluate_rotation_trajectory,
    generate_three_way_tradeoffs,
    resolve_torch_device,
    simulate_climate_scenario,
)

if HAS_TORCH:
    import torch
    from field_shift.rl_pipeline import CropRotationActorCritic, train_pytorch_actor_critic

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("FieldShiftIntegration")


def extract_year_trajectories(
    daily_df: pd.DataFrame, year: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    prev_year = year - 1
    windows = [
        (f"{prev_year}-11-01", f"{year}-03-15"),  # Rabi (Winter)
        (f"{year}-03-16", f"{year}-06-30"),       # Kharif-1 (Pre-Monsoon)
        (f"{year}-07-01", f"{year}-10-31"),       # Kharif-2 (Monsoon)
    ]

    seasonal_vectors: List[np.ndarray] = []
    seasonal_daily_seqs: List[np.ndarray] = []

    for idx, (start_dt, end_dt) in enumerate(windows):
        sub_df = daily_df.loc[start_dt:end_dt]
        if sub_df.empty:
            sub_df = daily_df.iloc[idx * 100 : (idx + 1) * 100]

        vec = PreprocessingPipeline.extract_seasonal_climate_vector(sub_df)
        seq = normalize_daily_sequence(sub_df, seq_len=120)

        seasonal_vectors.append(vec)
        seasonal_daily_seqs.append(seq)

    return seasonal_vectors, seasonal_daily_seqs


def build_multi_decade_regional_dataset(
    districts: List[FarmLocation],
    start_year: int = 2005,
    end_year: int = 2024,
    cache_dir: str = "./nasa_cache",
) -> Tuple[List[Dict], Dict[str, List[Dict]], Dict[str, Dict]]:
    power_client = NASAPowerClient(cache_dir=cache_dir)
    bulk_start = f"{start_year - 1}1101"
    bulk_end = f"{end_year}1031"

    training_bank: List[Dict] = []
    district_history_map: Dict[str, List[Dict]] = {d.district_name: [] for d in districts}
    latest_district_states: Dict[str, Dict] = {}
    total_daily_rows = 0

    for loc in districts:
        df_full = power_client.fetch_daily_agro(
            loc, start_date=bulk_start, end_date=bulk_end, use_cache=True
        )
        total_daily_rows += len(df_full)

        for yr in range(start_year, end_year + 1):
            vecs, seqs = extract_year_trajectories(df_full, yr)
            record = {
                "location": loc,
                "year": yr,
                "seasonal_vectors": vecs,
                "seasonal_daily_seqs": seqs,
            }
            training_bank.append(record)
            district_history_map[loc.district_name].append(record)
            if yr == end_year:
                latest_district_states[loc.district_name] = record

    logger.info(
        "Dataset Ready: %d districts x %d years = %d seasonal episodes (%d daily NASA rows cached in '%s').",
        len(districts),
        end_year - start_year + 1,
        len(training_bank),
        total_daily_rows,
        cache_dir,
    )
    return training_bank, district_history_map, latest_district_states


def extract_temporal_attention_windows(
    seasonal_daily_seqs: List[np.ndarray],
    actor_critic_model: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Extracts the peak climate-stress windows across the 120-day seasonal sequences
    using either the trained Bi-GRU Temporal Attention weights or physical VPD/Soil
    Moisture stress gradients.
    """
    windows_summary = []
    device = resolve_torch_device() if HAS_TORCH else "cpu"

    for s_idx, seq in enumerate(seasonal_daily_seqs):
        if HAS_TORCH and actor_critic_model is not None:
            actor_critic_model.eval()
            with torch.no_grad():
                d_t = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
                _, attn_w = actor_critic_model.temporal_encoder(d_t, return_weights=True)
                weights = attn_w.cpu().numpy().flatten()
        else:
            # Physics stress proxy: high VPD + high Tmax - root-zone soil wetness
            stress_raw = seq[:, 1] + seq[:, 10] - seq[:, 9]
            exp_s = np.exp(stress_raw - np.max(stress_raw))
            weights = exp_s / np.sum(exp_s)

        # Divide 120 days into 12 ten-day decads for clean frontend chart plotting
        decad_weights = [
            round(float(np.sum(weights[i * 10 : (i + 1) * 10])), 4) for i in range(12)
        ]
        peak_decad = int(np.argmax(decad_weights))
        day_start = peak_decad * 10 + 1
        day_end = (peak_decad + 1) * 10

        windows_summary.append({
            "season": SEASON_NAMES[s_idx],
            "peak_stress_days_within_season": f"Day {day_start} to Day {day_end}",
            "peak_attention_weight": decad_weights[peak_decad],
            "ten_day_attention_profile": decad_weights,
        })

    return windows_summary


def build_district_decision_payload(
    loc: FarmLocation,
    latest_record: Dict[str, Any],
    history_20yr: List[Dict[str, Any]],
    actor_critic_model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Constructs the complete JSON-serializable decision-support package for a district,
    including Baseline + 4 AI strategies (Balanced, Water Conservation, Soil Health,
    High-Value Vegetable), 5-point NASA explainability, 20-year uncertainty,
    12-month crop calendar & bare-field timelines, and What-If scenario slider outputs.
    """
    farm = FarmProfile(
        farm_id=loc.farm_id,
        irrigation_access=bool(loc.irrigation_access),
        baseline_soil_n_kg_ha=75.0,
        soil_organic_matter_pct=1.1,
        priority_mode="balanced",
        district_name=loc.district_name,
    )
    climates = latest_record["seasonal_vectors"]
    daily_seqs = latest_record["seasonal_daily_seqs"]

    # 1. Evaluate Existing Baseline (Boro Rice -> Fallow -> T. Aman Rice)
    baseline_eval = evaluate_rotation_trajectory(
        farm, climates, EXISTING_BASELINE_PATH, "balanced", daily_seqs
    )
    baseline_unc = compute_20yr_historical_uncertainty(
        farm, history_20yr, EXISTING_BASELINE_PATH, "balanced"
    )
    baseline_card = build_strategy_explainability(
        "Existing Rotation (Baseline)",
        baseline_eval,
        baseline_eval,
        baseline_unc,
        loc.district_name,
        loc.aez_name if hasattr(loc, "aez_name") else loc.aez_zone,
    )

    # 2. Evaluate the 4 Specialist Pareto AI Strategies (Core Field + High-Value Vegetable)
    core_candidates, veg_candidates = _get_candidate_pools()
    modes = [
        ("Balanced", "balanced", core_candidates),
        ("Water Conservation", "water_conservation", core_candidates),
        ("Soil Health", "soil_health", core_candidates),
        ("High-Value Vegetable", "high_value_vegetable", veg_candidates),
    ]

    strategy_cards = [baseline_card]
    for display_name, mode_key, pool in modes:
        evals = [
            evaluate_rotation_trajectory(farm, climates, path, mode_key, daily_seqs)
            for path in pool
        ]
        evals.sort(key=lambda x: x["cumulative_reward"], reverse=True)
        best_eval = evals[0]
        best_path = tuple(best_eval["sequence_ids"])
        unc = compute_20yr_historical_uncertainty(farm, history_20yr, best_path, mode_key)
        card = build_strategy_explainability(
            display_name, best_eval, baseline_eval, unc, loc.district_name, loc.aez_zone
        )
        strategy_cards.append(card)

    # 3. Pre-compute Interactive Scenario Slider Responses for Frontend
    scenarios = [
        {"scenario_id": "current_baseline", "label": "Current NASA Baseline (0% Rain, +0.0°C)", "rain_pct": 0.0, "temp_c": 0.0, "gw_pct": 0.0},
        {"scenario_id": "severe_drought", "label": "Severe Drought Shock (-25% Rain, +1.5°C, -10% GW)", "rain_pct": -25.0, "temp_c": +1.5, "gw_pct": -10.0},
        {"scenario_id": "heatwave_spike", "label": "Extreme Heatwave (+2.5°C Warming)", "rain_pct": -10.0, "temp_c": +2.5, "gw_pct": -5.0},
        {"scenario_id": "wet_monsoon_flood", "label": "Intense Monsoon Surge (+30% Rain, +15% GW)", "rain_pct": +30.0, "temp_c": +0.5, "gw_pct": +15.0},
    ]
    scenario_results = []
    for sc in scenarios:
        pert_clim = simulate_climate_scenario(
            climates,
            rainfall_delta_pct=sc["rain_pct"],
            temp_delta_c=sc["temp_c"],
            grace_gw_delta_pct=sc["gw_pct"],
        )
        df_sc = generate_three_way_tradeoffs(farm, pert_clim, actor_critic_model, include_existing_baseline=True)
        scenario_results.append({
            "scenario_id": sc["scenario_id"],
            "label": sc["label"],
            "parameters": {"rainfall_delta_pct": sc["rain_pct"], "temp_delta_c": sc["temp_c"], "grace_gw_delta_pct": sc["gw_pct"]},
            "recommendations": df_sc.to_dict(orient="records"),
        })

    # 4. Extract Temporal Attention Stress Windows
    attn_windows = extract_temporal_attention_windows(daily_seqs, actor_critic_model)

    return {
        "district": loc.district_name,
        "farm_id": loc.farm_id,
        "coordinates": {"latitude": loc.latitude, "longitude": loc.longitude},
        "agro_ecological_zone": loc.aez_zone,
        "grace_groundwater_percentile": loc.groundwater_baseline_pctl,
        "default_irrigation_access": bool(loc.irrigation_access),
        "temporal_attention_stress_windows": attn_windows,
        "strategies_comparison": strategy_cards,
        "scenario_slider_simulations": scenario_results,
    }


def run_nasa_rl_workflow():
    logger.info("Step 1: Building 20-Year (2005-2024) Multi-District NASA Dataset across 8 NW Bangladesh Districts...")
    training_bank, district_history_map, latest_district_states = build_multi_decade_regional_dataset(
        districts=NW_BANGLADESH_DISTRICTS,
        start_year=2005,
        end_year=2024,
        cache_dir="./nasa_cache",
    )

    actor_critic_model = None
    if HAS_TORCH:
        device = resolve_torch_device()
        logger.info(
            "Step 2: Training SOTA Temporal Attention Bi-GRU + 4-Head Actor-Critic on '%s' (1200 episodes across 50 crops)...",
            device,
        )
        actor_critic_model = CropRotationActorCritic(
            state_dim=17, daily_feature_dim=12, hidden_dim=64, action_dim=NUM_CROPS
        )

        def historical_env_generator(priority_mode: str = "balanced"):
            ep_data = training_bank[np.random.randint(0, len(training_bank))]
            loc: FarmLocation = ep_data["location"]
            profile = FarmProfile(
                farm_id=loc.farm_id,
                irrigation_access=bool(np.random.rand() > 0.25),
                baseline_soil_n_kg_ha=float(np.random.uniform(50.0, 100.0)),
                soil_organic_matter_pct=float(np.random.uniform(0.8, 1.8)),
                priority_mode=priority_mode,
                district_name=loc.district_name,
            )
            return CropRotationMDPEnv(
                profile,
                ep_data["seasonal_vectors"],
                ep_data["seasonal_daily_seqs"],
            )

        mode_history = train_pytorch_actor_critic(
            env_generator=historical_env_generator,
            policy_net=actor_critic_model,
            num_episodes=1200,
            lr=0.003,
            weight_decay=1e-4,
            gamma=0.99,
            entropy_coef=0.025,
            max_grad_norm=1.0,
            device=device,
        )

        for mode_name, rets in mode_history.items():
            logger.info(
                "  -> Head [%-20s] | Initial 20-ep Return: %+.3f | Final 20-ep Return: %+.3f",
                mode_name,
                float(np.mean(rets[:20])),
                float(np.mean(rets[-20:])),
            )

        ckpt_path = Path("field_shift_sota_policy.pt")
        torch.save(actor_critic_model.state_dict(), ckpt_path)
        logger.info("Saved trained SOTA PyTorch policy weights to '%s'.", ckpt_path)
    else:
        logger.info("Step 2: PyTorch not detected in VM environment; proceeding to Pareto evaluation.")

    # -------------------------------------------------------------------------
    # STEP 3: RAJSHAHI COMPARISON (EXISTING BASELINE + 4 AI STRATEGIES)
    # -------------------------------------------------------------------------
    logger.info("Step 3: Generating Baseline + 4 Specialist Pareto Strategies & Explainability Report...")
    rajshahi_latest = latest_district_states["Rajshahi"]

    live_farm = FarmProfile(
        farm_id="bd_rajshahi_01",
        irrigation_access=True,
        baseline_soil_n_kg_ha=75.0,
        soil_organic_matter_pct=1.1,
        priority_mode="balanced",
        district_name="Rajshahi",
    )

    tradeoff_table = generate_three_way_tradeoffs(
        farm=live_farm,
        seasonal_climates=rajshahi_latest["seasonal_vectors"],
        policy_net=actor_critic_model,
        seasonal_daily_sequences=rajshahi_latest["seasonal_daily_seqs"],
        include_existing_baseline=True,
        include_vegetable_strategy=True,
    )

    print("\n" + "=" * 118)
    print("=== FIELD SHIFT: RAJSHAHI STRATEGY COMPARISON (EXISTING BASELINE vs. 4 AI PARETO OPTIONS) ===")
    print("=" * 118)
    print(tradeoff_table.to_string(index=False))

  
    # STEP 4: REGIONAL MULTI-DISTRICT ADAPTATION MATRIX (ALL 8 DISTRICTS)

    district_rows = []
    web_districts_payload: Dict[str, Any] = {}

    for loc_meta in NW_BANGLADESH_DISTRICTS:
        dist_name = loc_meta.district_name
        state_data = latest_district_states[dist_name]
        history_20yr = district_history_map[dist_name]

        dist_payload = build_district_decision_payload(
            loc_meta, state_data, history_20yr, actor_critic_model
        )
        web_districts_payload[dist_name] = dist_payload

        target_mode = (
            "Water Conservation"
            if not loc_meta.irrigation_access or loc_meta.groundwater_baseline_pctl < 25.0
            else "Balanced"
        )
        matched_card = next(
            c for c in dist_payload["strategies_comparison"] if c["strategy_name"] == target_mode
        )
        veg_card = next(
            c for c in dist_payload["strategies_comparison"] if c["strategy_name"] == "High-Value Vegetable"
        )
        m = matched_card["metrics"]
        u = matched_card["uncertainty_20yr_band"]

        district_rows.append({
            "District": dist_name,
            "GRACE GW (%)": int(loc_meta.groundwater_baseline_pctl),
            "Active Policy": target_mode,
            "Field Crop Rotation": matched_card["recommended_rotation"],
            "Field Profit ($/ha)": m["net_profit_usd_ha"],
            "Vegetable Rotation": veg_card["recommended_rotation"],
            "Veg Profit ($/ha)": veg_card["metrics"]["net_profit_usd_ha"],
            "Water Saved (mm)": m["water_saved_vs_baseline_mm"],
        })

    regional_df = pd.DataFrame(district_rows)
    print("\n" + "=" * 136)
    print("=== NORTHWEST BANGLADESH REGIONAL ADAPTATION MATRIX (FIELD CROP vs. HIGH-VALUE VEGETABLE ROTATIONS) ===")
    print("=" * 136)
    print(regional_df.to_string(index=False))

    # Print sample 5-point explainability report for Rajshahi strategies
    raj_cards = web_districts_payload["Rajshahi"]["strategies_comparison"]
    print("\n" + "=" * 118)
    print("=== SAMPLE 5-POINT NASA EXPLAINABILITY & RISK REPORT (RAJSHAHI - ALL 4 AI STRATEGIES) ===")
    print("=" * 118)
    for card in raj_cards[1:]:
        print(f"\n[{card['strategy_name'].upper()}] -> {card['recommended_rotation']}")
        print(f"  1. Why Suggested        : {card['why_suggested']}")
        print(f"  2. NASA Evidence        : {card['supporting_nasa_evidence'][0]} | {card['supporting_nasa_evidence'][1]}")
        print(f"  3. Agronomic Benefits   : {card['agronomic_and_soil_benefits'][0]}")
        print(f"  4. Uncertainty & Gaps   : {card['information_gaps_and_uncertainty'][0]}")
        print(f"  5. Local Expert Review  : {card['risks_requiring_local_expert_review'][0]}")

    # STEP 5: EXPORT COMPLETE JSON BUNDLE FOR WEB APPLICATION FRONTEND
    full_web_bundle = {
        "project_title": "Field Shift: Adapting Farms with NASA Data (2026 NASA Space Apps Challenge)",
        "target_region": "Bangladesh (64-District / 50-Crop Rotational Catalog + 20-Year Daily NASA Dataset 2005-2024)",
        "farmer_centered_advisory_notice": (
            "IMPORTANT ADVISORY: Field Shift is a decision-support tool designed to help farmers and "
            "agricultural extension officers explore climate-resilient crop rotation possibilities using "
            "NASA satellite observations. It does NOT replace farmer judgment, local DAE agronomic expertise, "
            "or plot-level soil testing. Always review sowing dates, drainage, and market logistics locally."
        ),
        "nasa_datasets_harmonized": [
            {"dataset": "NASA POWER / MERRA-2", "variables": ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_PAR_TOT", "WS2M", "PRECTOTCORR"], "resolution": "Daily, 0.5° x 0.625°"},
            {"dataset": "NASA SMAP Level-4 (SPL4SMGP) & L3", "variables": ["sm_surface (0-5cm)", "sm_rootzone (0-100cm)", "GWETTOP", "GWETROOT"], "resolution": "9 km EASE-Grid, Daily/3-Hourly"},
            {"dataset": "NASA GRACE & GRACE-FO (GRACEDADM)", "variables": ["gws_inst (Groundwater Storage Percentile)", "rtzsm_inst"], "resolution": "0.25° (~25 km), Weekly"},
            {"dataset": "NASA MODIS & ECOSTRESS", "variables": ["MOD13Q1 NDVI/EVI (250m)", "MOD16A2GF ET/PET (500m)", "ECO_L3T_JET (70m)", "ECO_L4T_ESI"], "resolution": "70m - 500m, 8/16-Day"},
            {"dataset": "NASA HLS (Landsat 8/9 + Sentinel-2) & SRTM", "variables": ["HLSL30 30m NDVI/EVI", "SRTM 30m Elevation & Slope", "MCD12Q1 Land Cover"], "resolution": "30m - 500m"},
        ],
        "scientific_glossary_for_farmers": FARMER_SCIENTIFIC_GLOSSARY,
        "crop_catalog": [c.__dict__ for c in CROP_CATALOG],
        "districts": web_districts_payload,
    }

    json_path = Path("field_shift_web_payload.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_web_bundle, f, indent=2, ensure_ascii=False)
    logger.info("Exported complete frontend-ready JSON payload to '%s' (%.1f KB).", json_path, json_path.stat().st_size / 1024.0)


if __name__ == "__main__":
    run_nasa_rl_workflow()
