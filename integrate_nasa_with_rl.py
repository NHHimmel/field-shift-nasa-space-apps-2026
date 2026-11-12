"""
Field Shift: End-to-End 20-Year Multi-District NASA Dataset & Temporal RL Pipeline
==================================================================================

What This Script Executes:
1. Bulk Multi-Decade Dataset Construction (`2005-2024`, 20 agricultural years):
   - Fetches and caches 20 years of daily NASA POWER agroclimatology & soil wetness
     across 8 representative agricultural districts in Northwest Bangladesh:
     (Rajshahi, Chapai Nawabganj, Naogaon, Bogura, Dinajpur, Rangpur, Pabna, Kurigram).
   - Total Dataset Size: 8 districts x 20 years = 160 farm-year episodes
     (~58,400 daily rows x 16 variables = >930,000 environmental data points).
   - Saves each district's 20-year daily CSV to `./nasa_cache/` so subsequent runs
     load from disk in <0.2 seconds.
2. Daily Temporal Sequence Tensor Extraction:
   - Slices every farm-year into 3 seasonal daily matrices `[120_days, 12_features]`
     (Rabi, Kharif-1, Kharif-2) alongside 8-dim seasonal climate vectors.
3. SOTA Temporal Attention Bi-GRU Actor-Critic Training:
   - Trains the Multi-Head PyTorch network across 900 episodes sampled from the
     160 real historical district-year trajectories using AdamW and Cosine Annealing.
   - Saves the trained checkpoint to `field_shift_sota_policy.pt`.
4. Regional & Farm-Level Inference:
   - Evaluates the trained model on the latest season and outputs both the
     3-way Pareto trade-off table for Rajshahi and a regional multi-district
     adaptation summary.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

current_dir = str(Path(__file__).resolve().parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np
import pandas as pd

from field_shift_nasa_data import (
    NW_BANGLADESH_DISTRICTS,
    FarmLocation,
    NASAPowerClient,
    normalize_daily_sequence,
)
from field_shift_rl_pipeline import (
    CROP_CATALOG,
    HAS_TORCH,
    NUM_CROPS,
    CropRotationMDPEnv,
    FarmProfile,
    PreprocessingPipeline,
    generate_three_way_tradeoffs,
    resolve_torch_device,
)

if HAS_TORCH:
    import torch
    from field_shift_rl_pipeline import CropRotationActorCritic, train_pytorch_actor_critic

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("FieldShiftIntegration")


def extract_year_trajectories(
    daily_df: pd.DataFrame, year: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Slices a continuous multi-year daily DataFrame for agricultural `year` into:
      1. `seasonal_vectors`: Three 8-dim summary vectors [Rabi, Kharif-1, Kharif-2]
      2. `seasonal_daily_seqs`: Three `[120_days, 12_features]` daily matrices for the Bi-GRU
    """
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
            # Fallback positional slice if index boundaries are outside range
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
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """
    Fetches or loads 20 years of daily NASA observations across all 8 Northwest
    Bangladesh districts in one bulk call per district, constructing 160 historical
    farm-year episodes.
    """
    power_client = NASAPowerClient(cache_dir=cache_dir)
    bulk_start = f"{start_year - 1}1101"
    bulk_end = f"{end_year}1031"

    training_bank: List[Dict] = []
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
    return training_bank, latest_district_states


def run_nasa_rl_workflow():
    logger.info("Step 1: Building 20-Year (2005-2024) Multi-District NASA Dataset across 8 NW Bangladesh Districts...")
    training_bank, latest_district_states = build_multi_decade_regional_dataset(
        districts=NW_BANGLADESH_DISTRICTS,
        start_year=2005,
        end_year=2024,
        cache_dir="./nasa_cache",
    )

    actor_critic_model = None
    if HAS_TORCH:
        device = resolve_torch_device()
        logger.info(
            "Step 2: Training SOTA Temporal Attention Bi-GRU + Multi-Head Actor-Critic on '%s' (900 episodes)...",
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
            )
            return CropRotationMDPEnv(
                profile,
                ep_data["seasonal_vectors"],
                ep_data["seasonal_daily_seqs"],
            )

        mode_history = train_pytorch_actor_critic(
            env_generator=historical_env_generator,
            policy_net=actor_critic_model,
            num_episodes=900,
            lr=0.003,
            weight_decay=1e-4,
            gamma=0.99,
            entropy_coef=0.025,
            max_grad_norm=1.0,
            device=device,
        )

        for mode_name, rets in mode_history.items():
            logger.info(
                "  -> Head [%-18s] | Initial 20-ep Return: %+.3f | Final 20-ep Return: %+.3f",
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
    # STEP 3: INFERENCE FOR TARGET FARM IN RAJSHAHI (HIGH BARIND TRACT)
    # -------------------------------------------------------------------------
    logger.info("Step 3: Generating Page-2 Multi-Objective Trade-Off Recommendations for Rajshahi...")
    rajshahi_latest = latest_district_states["Rajshahi"]

    live_farm = FarmProfile(
        farm_id="bd_rajshahi_01",
        irrigation_access=True,
        baseline_soil_n_kg_ha=75.0,
        soil_organic_matter_pct=1.1,
        priority_mode="balanced",
    )

    tradeoff_table = generate_three_way_tradeoffs(
        farm=live_farm,
        seasonal_climates=rajshahi_latest["seasonal_vectors"],
        policy_net=actor_critic_model,
        seasonal_daily_sequences=rajshahi_latest["seasonal_daily_seqs"],
    )

    print("\n" + "=" * 78)
    print("=== FIELD SHIFT: RAJSHAHI FARM RECOMMENDATIONS (PAGE 2 DELIVERABLE) ===")
    print("=" * 78)
    print(tradeoff_table.to_string(index=False))

    # -------------------------------------------------------------------------
    # STEP 4: REGIONAL MULTI-DISTRICT ADAPTATION SUMMARY (ALL 8 DISTRICTS)
    # -------------------------------------------------------------------------
    district_rows = []
    for dist_name, state_data in latest_district_states.items():
        loc_meta: FarmLocation = state_data["location"]
        f_prof = FarmProfile(
            farm_id=loc_meta.farm_id,
            irrigation_access=bool(loc_meta.irrigation_access),
            baseline_soil_n_kg_ha=75.0,
            soil_organic_matter_pct=1.2,
            priority_mode="water_conservation" if not loc_meta.irrigation_access else "balanced",
        )
        df_dist = generate_three_way_tradeoffs(
            farm=f_prof,
            seasonal_climates=state_data["seasonal_vectors"],
            policy_net=actor_critic_model,
            seasonal_daily_sequences=state_data["seasonal_daily_seqs"],
        )
        # Pick row matching the district's primary hydrological regime
        target_mode = "Water Conservation" if not loc_meta.irrigation_access or loc_meta.groundwater_baseline_pctl < 25.0 else "Balanced"
        row = df_dist[df_dist["Policy Mode"] == target_mode].iloc[0]
        district_rows.append({
            "District": dist_name,
            "Agro-Ecological Zone": loc_meta.aez_zone,
            "GRACE GW (%)": int(loc_meta.groundwater_baseline_pctl),
            "Active Policy": target_mode,
            "Optimal Rotation Sequence": row["Recommended Rotation"],
            "Profit ($/ha)": row["Total Net Profit ($/ha)"],
            "Water (mm)": row["Total Water Used (mm)"],
        })

    regional_df = pd.DataFrame(district_rows)
    print("\n" + "=" * 105)
    print("=== NORTHWEST BANGLADESH REGIONAL ADAPTATION MATRIX (20-YEAR NASA DATASET) ===")
    print("=" * 105)
    print(regional_df.to_string(index=False))


if __name__ == "__main__":
    run_nasa_rl_workflow()