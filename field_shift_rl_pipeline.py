"""
Field Shift: SOTA Temporal Attention Bi-GRU Actor-Critic & MDP Crop Rotation System

Target Region: Northwest Bangladesh (8 Agricultural Districts across Barind & Tista AEZs)
Cropping Horizon: Finite Horizon T=3 Seasons (Rabi -> Kharif-1 -> Kharif-2)

Agronomic & Physics Calibrations:
  1. Continuous NASA Climate-Driven Yield & ETc Physics:
     - Crop water demand (ETc) scales dynamically with each district's seasonal NASA
       Vapor Pressure Deficit (VPD) and temperature, while irrigation deficit accounts
       for effective seasonal rainfall and SMAP root-zone soil moisture storage.
     - Crop gross margin responds continuously to NASA solar radiation, thermal optima
       (e.g., cooler northern winter boost for Potato/Wheat in Bogura/Dinajpur), and
       soil saturation/waterlogging risk (favoring Jute on active floodplains like Kurigram).
  2. Exact, Unclipped Soil Mineral Nitrogen Mass Balance:
     - Models net mineral N carryover after standard maintenance fertilization so
       multi-season N deltas sum exactly without hitting artificial clipping floors.
  3. End-to-End Temporal Attention Bi-GRU + Multi-Head Specialist Actor-Critic:
     - Ingests `[Batch, 120_days, 12_features]` daily NASA sequences alongside soil state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
    """
    Safely probes CUDA / cuBLAS compatibility and falls back to CPU if the local GPU
    architecture (e.g., GeForce MX330 sm_61) mismatches the installed CUDA binary.
    """
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


# STAGE 1: AGRONOMIC CATALOG & ACTION SPACE SPECIFICATION

@dataclass(frozen=True)
class CropSpec:
    crop_id: int
    name: str
    season: str              # 'Rabi', 'Kharif-1', 'Kharif-2'
    family: str              # 'Fabaceae', 'Poaceae', 'Brassicaceae', 'Solanaceae', 'Malvaceae', 'None'
    duration_days: int
    water_req_mm: float      # Reference seasonal crop evapotranspiration requirement (ETc)
    n_fixation_kg_ha: float  # Net residual soil N change (kg N/ha) after standard management
    baseline_margin_usd: float # Reference gross margin USD/ha
    drought_hardiness: float # 0.0 to 1.0
    waterlogging_hardiness: float # 0.0 to 1.0
    optimal_temp_c: float    # Optimal seasonal mean temperature (C)


# Calibrated Crop Catalog for Northwest Bangladesh (BARI / BRRI / DAE standards)
CROP_CATALOG: List[CropSpec] = [
    # --- RABI (Winter, Indices 0 - 5) ---
    CropSpec(0, "Lentil", "Rabi", "Fabaceae", duration_days=105, water_req_mm=200.0, n_fixation_kg_ha=+32.0, baseline_margin_usd=780.0, drought_hardiness=0.90, waterlogging_hardiness=0.15, optimal_temp_c=20.0),
    CropSpec(1, "Mustard", "Rabi", "Brassicaceae", duration_days=80, water_req_mm=190.0, n_fixation_kg_ha=-8.0, baseline_margin_usd=660.0, drought_hardiness=0.75, waterlogging_hardiness=0.25, optimal_temp_c=20.5),
    CropSpec(2, "Wheat", "Rabi", "Poaceae", duration_days=110, water_req_mm=320.0, n_fixation_kg_ha=-22.0, baseline_margin_usd=640.0, drought_hardiness=0.65, waterlogging_hardiness=0.30, optimal_temp_c=19.5),
    CropSpec(3, "Potato", "Rabi", "Solanaceae", duration_days=90, water_req_mm=410.0, n_fixation_kg_ha=-28.0, baseline_margin_usd=1380.0, drought_hardiness=0.45, waterlogging_hardiness=0.15, optimal_temp_c=18.5),
    CropSpec(4, "Maize", "Rabi", "Poaceae", duration_days=140, water_req_mm=520.0, n_fixation_kg_ha=-35.0, baseline_margin_usd=1080.0, drought_hardiness=0.60, waterlogging_hardiness=0.40, optimal_temp_c=21.5),
    CropSpec(5, "Boro Rice", "Rabi", "Poaceae", duration_days=150, water_req_mm=1150.0, n_fixation_kg_ha=-42.0, baseline_margin_usd=720.0, drought_hardiness=0.10, waterlogging_hardiness=0.95, optimal_temp_c=24.0),

    # --- KHARIF-1 (Pre-Monsoon, Indices 6 - 9) ---
    CropSpec(6, "Mungbean", "Kharif-1", "Fabaceae", duration_days=70, water_req_mm=210.0, n_fixation_kg_ha=+28.0, baseline_margin_usd=590.0, drought_hardiness=0.82, waterlogging_hardiness=0.25, optimal_temp_c=29.0),
    CropSpec(7, "Aus Rice", "Kharif-1", "Poaceae", duration_days=110, water_req_mm=540.0, n_fixation_kg_ha=-20.0, baseline_margin_usd=520.0, drought_hardiness=0.55, waterlogging_hardiness=0.80, optimal_temp_c=30.0),
    CropSpec(8, "Jute", "Kharif-1", "Malvaceae", duration_days=120, water_req_mm=460.0, n_fixation_kg_ha=+10.0, baseline_margin_usd=540.0, drought_hardiness=0.50, waterlogging_hardiness=0.95, optimal_temp_c=30.5),
    CropSpec(9, "Fallow", "Kharif-1", "None", duration_days=0, water_req_mm=0.0, n_fixation_kg_ha=+5.0, baseline_margin_usd=0.0, drought_hardiness=1.00, waterlogging_hardiness=1.00, optimal_temp_c=28.0),

    # --- KHARIF-2 (Monsoon, Index 10) ---
    CropSpec(10, "T. Aman Rice", "Kharif-2", "Poaceae", duration_days=130, water_req_mm=780.0, n_fixation_kg_ha=-24.0, baseline_margin_usd=710.0, drought_hardiness=0.45, waterlogging_hardiness=0.92, optimal_temp_c=28.5),
]

NUM_CROPS = len(CROP_CATALOG)
SEASON_NAMES = ["Rabi", "Kharif-1", "Kharif-2"]
SEASON_ACTION_MAP: Dict[int, List[int]] = {
    0: [0, 1, 2, 3, 4, 5],      # Rabi
    1: [6, 7, 8, 9],            # Kharif-1
    2: [10],                    # Kharif-2
}
PRIORITY_MODES = ["balanced", "water_conservation", "soil_health"]


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


# STAGE 2: CALIBRATED AGRICULTURAL MDP ENVIRONMENT

@dataclass
class FarmProfile:
    farm_id: str
    irrigation_access: bool        # True if farm has motorized deep/shallow tubewell
    baseline_soil_n_kg_ha: float   # Initial soil mineral nitrogen pool (e.g. 75 kg/ha)
    soil_organic_matter_pct: float # Baseline Soil Organic Carbon % (from SoilGrids 250m)
    priority_mode: str             # 'balanced', 'water_conservation', 'soil_health'


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
        self.cumulative_profit_usd = 0.0
        self.rotation_history: List[CropSpec] = []

        self.weights = weights or self._resolve_weights(farm_profile.priority_mode)

    @staticmethod
    def _resolve_weights(priority: str) -> Dict[str, float]:
        if priority == "water_conservation":
            return {"profit": 0.12, "water_weight": 1.45, "soil_weight": 0.15}
        elif priority == "soil_health":
            return {"profit": 0.18, "water_weight": 0.15, "soil_weight": 1.45}
        return {"profit": 0.72, "water_weight": 0.18, "soil_weight": 0.12}  # balanced

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
            1.0 if self.prev_family in ["Brassicaceae", "Solanaceae", "Malvaceae"] else 0.0,
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

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(NUM_CROPS, dtype=np.float32)
        legal_indices = SEASON_ACTION_MAP[self.current_season_idx]
        mask[legal_indices] = 1.0
        return mask

    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Executes a single seasonal transition with continuous NASA climate physics.
        """
        crop = CROP_CATALOG[action_id]
        climate = self.seasonal_climates[self.current_season_idx]

        # De-normalize continuous NASA climate features for current season
        temp_c = climate[0] * 10.0 + 25.0
        heat_days = climate[1] * 30.0
        precip_mm = np.expm1(climate[2] * 6.0)
        cdd_days = climate[3] * 30.0
        sm_surf = float(climate[4])
        sm_root = float(climate[5])
        vpd_kpa = climate[6] * 2.5
        grace_pctl = climate[7] * 100.0

        # --- 1. Site-Specific Crop Evapotranspiration Demand (ETc) ---
        if crop.water_req_mm > 0.0:
            et_mod = 1.0 + 0.10 * (vpd_kpa - 1.05) + 0.008 * (temp_c - crop.optimal_temp_c)
            water_used = float(np.clip(crop.water_req_mm * et_mod, crop.water_req_mm * 0.85, crop.water_req_mm * 1.25))
        else:
            water_used = 0.0

        effective_rain_and_soil = (precip_mm * 0.65) + (sm_root * 180.0)
        blue_water_deficit = max(0.0, water_used - effective_rain_and_soil)

        # --- 2. Continuous Agro-Ecological Yield & Margin Realization ---
        effective_margin = crop.baseline_margin_usd

        if crop.name != "Fallow":
            # A. Thermal suitability curve
            temp_diff = abs(temp_c - crop.optimal_temp_c)
            thermal_mult = np.clip(1.04 - 0.015 * temp_diff, 0.80, 1.08)
            effective_margin *= thermal_mult

            # B. Soil Nitrogen carryover response
            n_mult = np.clip(1.0 + 0.0025 * (self.soil_n - 65.0), 0.85, 1.18)
            effective_margin *= n_mult

            # C. Active Floodplain vs. Well-Drained Terrace Suitability:
            # Active Tista Floodplain (Kurigram: GRACE >= 50%) has abundant surface water for
            # Jute retting (+35% Jute margin) but high waterlogging/blight risk for Potato (-35%).
            if grace_pctl >= 50.0:
                if crop.name == "Jute":
                    effective_margin *= 1.38
                elif crop.name == "Lentil" or crop.name == "Mustard":
                    effective_margin *= 1.08
                elif crop.name == "Potato":
                    effective_margin *= 0.65

            if heat_days > 4.0 and crop.name == "Wheat":
                effective_margin *= 0.80

        # --- 3. Water Deficit & Aquifer Depletion Penalty ---
        water_penalty = 0.0
        if crop.water_req_mm > 0.0:
            water_penalty += water_used * 0.55 + blue_water_deficit * 0.45

            if not self.profile.irrigation_access:
                if blue_water_deficit > 90.0:
                    water_penalty += (blue_water_deficit - 90.0) * 2.5
                    drought_loss = np.clip(1.0 - (blue_water_deficit - 90.0) / 450.0 * (1.1 - crop.drought_hardiness), 0.35, 0.95)
                    effective_margin *= drought_loss
            else:
                pumping_cost_per_mm = 0.10 if grace_pctl >= 30.0 else 0.22
                effective_margin -= blue_water_deficit * pumping_cost_per_mm
                if grace_pctl < 30.0 and water_used > 750.0:
                    water_penalty += 450.0

        # --- 4. Exact Soil Mineral Nitrogen Mass Balance (No Clipping Floor) ---
        soil_bonus = 0.0
        self.soil_n = self.soil_n + crop.n_fixation_kg_ha

        if crop.family == "Fabaceae":
            soil_bonus += 280.0 + crop.n_fixation_kg_ha * 3.0
        elif crop.name == "Jute":
            soil_bonus += 60.0
        else:
            soil_bonus += crop.n_fixation_kg_ha * 2.5

        if self.prev_family != "None" and self.prev_family == crop.family:
            soil_bonus -= 180.0
            effective_margin *= 0.88
        elif self.prev_family != "None" and self.prev_family != crop.family and crop.name != "Fallow":
            soil_bonus += 70.0

        # --- 5. Multi-Objective Scalar Reward ---
        norm_profit = effective_margin / 1000.0
        norm_water_pen = water_penalty / 500.0
        norm_soil_bon = soil_bonus / 200.0

        reward = (
            self.weights["profit"] * norm_profit
            - self.weights["water_weight"] * norm_water_pen
            + self.weights["soil_weight"] * norm_soil_bon
        )

        self.cumulative_water_used_mm += water_used
        self.cumulative_profit_usd += effective_margin
        self.rotation_history.append(crop)
        self.prev_crop_id = action_id
        self.prev_family = crop.family

        self.current_season_idx += 1
        done = (self.current_season_idx >= 3)
        next_state = self.get_state() if not done else np.zeros(17, dtype=np.float32)

        info = {
            "crop": crop.name,
            "margin_usd": effective_margin,
            "water_mm": water_used,
            "soil_n_post": self.soil_n,
        }
        return next_state, reward, done, info

    def reset(self) -> np.ndarray:
        self.current_season_idx = 0
        self.soil_n = float(self.profile.baseline_soil_n_kg_ha)
        self.prev_crop_id = None
        self.prev_family = "None"
        self.cumulative_water_used_mm = 0.0
        self.cumulative_profit_usd = 0.0
        self.rotation_history.clear()
        return self.get_state()


# STAGE 3: SOTA TEMPORAL ATTENTION BI-GRU + MULTI-HEAD ACTOR-CRITIC

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

        def forward(self, daily_seq: torch.Tensor) -> torch.Tensor:
            rnn_out, _ = self.bi_gru(daily_seq)
            attn_scores = self.attention_weights(rnn_out)
            attn_probs = F.softmax(attn_scores, dim=1)
            context = torch.sum(rnn_out * attn_probs, dim=1)
            return self.layer_norm(context)

    class CropRotationActorCritic(nn.Module):
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
            logits = self.actor_heads[priority_mode](features)

            masked_logits = torch.where(
                action_mask > 0.5,
                logits,
                torch.tensor(-1e9, dtype=logits.dtype, device=logits.device),
            )
            dist = Categorical(logits=masked_logits)
            value = self.critic_heads[priority_mode](features)
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
            logits = self.actor_heads[priority_mode](features)
            masked_logits = torch.where(
                action_masks > 0.5,
                logits,
                torch.tensor(-1e9, dtype=logits.dtype, device=logits.device),
            )
            dist = Categorical(logits=masked_logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            values = self.critic_heads[priority_mode](features)
            return log_probs, values.squeeze(-1), entropy


# STAGE 4: ADVANTAGE ACTOR-CRITIC (A2C) TRAINING OVER DAILY SEQUENCES

def train_pytorch_actor_critic(
    env_generator,
    policy_net: nn.Module,
    num_episodes: int = 900,
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


# STAGE 5: MULTI-OBJECTIVE TRADE-OFF EVALUATOR

def evaluate_rotation_trajectory(
    farm: FarmProfile,
    seasonal_climates: List[np.ndarray],
    sequence_crop_ids: Tuple[int, int, int],
    priority_mode: str,
    seasonal_daily_sequences: Optional[List[np.ndarray]] = None,
) -> Dict[str, Union[str, float]]:
    test_farm = FarmProfile(
        farm_id=farm.farm_id,
        irrigation_access=farm.irrigation_access,
        baseline_soil_n_kg_ha=farm.baseline_soil_n_kg_ha,
        soil_organic_matter_pct=farm.soil_organic_matter_pct,
        priority_mode=priority_mode,
    )
    env = CropRotationMDPEnv(test_farm, seasonal_climates, seasonal_daily_sequences)
    env.reset()

    cumulative_reward = 0.0
    for a in sequence_crop_ids:
        _, r, _, _ = env.step(a)
        cumulative_reward += r

    seq_str = " -> ".join([c.name for c in env.rotation_history])
    return {
        "sequence_ids": sequence_crop_ids,
        "rotation_name": seq_str,
        "cumulative_reward": cumulative_reward,
        "total_profit_usd": round(env.cumulative_profit_usd, 1),
        "total_water_mm": round(env.cumulative_water_used_mm, 1),
        "final_soil_n": round(env.soil_n, 1),
        "n_delta": round(env.soil_n - farm.baseline_soil_n_kg_ha, 1),
    }


def generate_three_way_tradeoffs(
    farm: FarmProfile,
    seasonal_climates: List[np.ndarray],
    policy_net: Optional[Any] = None,
    seasonal_daily_sequences: Optional[List[np.ndarray]] = None,
) -> pd.DataFrame:
    legal_rabi = SEASON_ACTION_MAP[0]
    legal_k1 = SEASON_ACTION_MAP[1]
    legal_k2 = SEASON_ACTION_MAP[2]

    candidates = [
        (r, k1, k2)
        for r in legal_rabi
        for k1 in legal_k1
        for k2 in legal_k2
    ]

    modes = [
        ("Balanced", "balanced"),
        ("Water Conservation", "water_conservation"),
        ("Soil Health", "soil_health"),
    ]

    selected_rows = []
    for display_name, mode_key in modes:
        evals = [
            evaluate_rotation_trajectory(
                farm, seasonal_climates, path, mode_key, seasonal_daily_sequences
            )
            for path in candidates
        ]
        evals.sort(key=lambda x: x["cumulative_reward"], reverse=True)
        best = evals[0]

        selected_rows.append({
            "Policy Mode": display_name,
            "Recommended Rotation": best["rotation_name"],
            "Total Net Profit ($/ha)": best["total_profit_usd"],
            "Total Water Used (mm)": best["total_water_mm"],
            "Final Soil Nitrogen (kg/ha)": best["final_soil_n"],
            "Nitrogen Delta (kg/ha)": best["n_delta"],
        })

    return pd.DataFrame(selected_rows)


find_pareto_optimal_rotations = generate_three_way_tradeoffs


if __name__ == "__main__":
    logger.info("Initializing Field Shift Calibrated SOTA Pipeline (PyTorch: %s)...", HAS_TORCH)

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
    )

    tradeoff_df = generate_three_way_tradeoffs(barind_farm, climates)
    print("\n" + tradeoff_df.to_string(index=False))