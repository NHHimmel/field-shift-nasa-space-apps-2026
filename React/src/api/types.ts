// Shapes returned by the FastAPI backend (FastApi/app/engine.py, explainer.py).

export type Lang = 'bn' | 'en'
export type SeasonKey = 'rabi' | 'kharif1' | 'kharif2'
export type StrategyKey = 'baseline' | 'balanced' | 'water_conservation' | 'soil_health' | 'high_value_vegetable'
export type Priority = Exclude<StrategyKey, 'baseline'>
export type SoilLevel = 'low' | 'medium' | 'high' | 'unknown'
export type ScenarioId = 'normal' | 'drought' | 'heatwave' | 'wet_monsoon'

export const SEASONS: SeasonKey[] = ['rabi', 'kharif1', 'kharif2']
export const PRIORITIES: Priority[] = ['balanced', 'water_conservation', 'soil_health', 'high_value_vegetable']

export interface DistrictInfo {
  key: string
  name_en: string
  name_bn: string
  division_en: string
  division_bn: string
  agro_ecological_zone: string
  latitude: number
  longitude: number
  groundwater_pctl: number
  default_irrigation: boolean
}

export interface CropInfo {
  key: string
  name_en: string
  name_bn: string
  season: SeasonKey | null
  category: string
  family: string
}

export interface Meta {
  districts: DistrictInfo[]
  crops: CropInfo[]
  season_crops: Record<SeasonKey, string[]>
  default_current_crops: Record<SeasonKey, string>
  units: { usd_to_bdt: number; ha_per_bigha: number }
}

export interface Health {
  ok: boolean
  gemini_configured: boolean
  gemini_model: string
  policy_model: string
}

export interface Pest {
  kind: 'broken' | 'repeat' | 'fallow' | 'first'
  from?: string | null
  to?: string | null
  family?: string | null
}

export interface SeasonStep {
  season: SeasonKey
  crop: string
  category: string
  family: string
  sowing_window: string
  harvest_window: string
  season_days: number
  active_days: number
  bare_days: number
  water_mm: number
  irrigation_mm: number
  rain_mm: number
  mean_temp_c: number
  heat_stress_days: number
  dry_spell_days: number
  rootzone_wetness_pct: number
  groundwater_pctl: number
  soil_n_change_kg_bigha: number
  pest: Pest
}

export interface Metrics {
  profit_bdt_bigha: number
  profit_change_bdt_bigha: number
  water_mm: number
  irrigation_mm: number
  water_saved_mm: number
  soil_n_kg_bigha: number
  soil_n_change_kg_bigha: number
  soil_n_change_vs_baseline_kg_bigha: number
  cover_days: number
  bare_days: number
}

export interface Range20 {
  profit_typical_bdt_bigha: number
  profit_spread_bdt_bigha: number
  profit_bad_year_bdt_bigha: number
  profit_good_year_bdt_bigha: number
  water_typical_mm: number
  water_dry_year_mm: number
}

export interface StrategyCard {
  key: StrategyKey
  crops: string[]
  metrics: Metrics
  range_20yr: Range20
  seasons: SeasonStep[]
}

export interface ScenarioRow {
  key: StrategyKey
  crops: string[]
  metrics: Metrics
}

export interface Scenario {
  id: ScenarioId
  rain_pct: number
  temp_c: number
  gw_pct: number
  strategies: ScenarioRow[]
}

export interface StressSeason {
  season: SeasonKey
  profile: number[]
  peak_index: number
  peak_start: string // "MM-DD"
  peak_end: string
}

export interface Recommendation {
  district: DistrictInfo
  inputs: {
    irrigation: boolean
    irrigation_was_default: boolean
    soil_fertility: Exclude<SoilLevel, 'unknown'>
    current_crops: Record<SeasonKey, string>
    excluded_crops: string[]
  }
  strategies: StrategyCard[]
  scenarios: Scenario[]
  stress_windows: { source: 'trained_attention' | 'physics_proxy'; seasons: StressSeason[] }
  units: { currency: string; area: string; usd_to_bdt: number; ha_per_bigha: number }
  data_info: {
    climate_years: [number, number]
    latest_year: number
    crop_catalog_size: number
    nasa_sources?: NASADataSource[]
    inference?: InferenceInfo
  }
  recommended_strategy: Priority
}

export interface NASADataSource {
  name: string
  role: string
  status: 'active' | 'derived' | 'fallback'
}

export interface InferenceInfo {
  recommendation_method: 'deterministic_rotation_search' | 'trained_neural_policy'
  neural_policy_used: boolean
  policy_status: string
  stress_window_source: 'trained_attention' | 'physics_proxy'
  checkpoint_name?: string | null
}

export interface RecommendRequest {
  district: string
  irrigation: boolean | null
  priority: Priority
  soil_fertility: SoilLevel
  current_crops: Record<SeasonKey, string> | null
  excluded_crops: string[]
}

export interface ExplanationSections {
  headline: string
  why: string
  nasa_evidence: string[]
  soil_benefits: string[]
  confidence: string
  expert_checks: string[]
}

export interface Explanation {
  strategy: StrategyKey
  lang: Lang
  source: 'gemini' | 'gemini_cached' | 'template'
  sections: ExplanationSections
}
