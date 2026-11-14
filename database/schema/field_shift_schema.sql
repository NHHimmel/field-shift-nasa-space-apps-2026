-- ============================================================================
-- Field Shift: Adapting Farms with NASA Data (2026 NASA Space Apps Challenge)
-- Complete Relational Database Schema for MySQL 8.0+ (All 64 Districts & 72 Crops)
-- ============================================================================

CREATE DATABASE IF NOT EXISTS field_shift_db
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE field_shift_db;

-- 1. Districts Table (All 64 Administrative Districts of Bangladesh + NASA Climate Baselines)
CREATE TABLE IF NOT EXISTS districts (
    district_id INT AUTO_INCREMENT PRIMARY KEY,
    district_name VARCHAR(64) NOT NULL UNIQUE,
    division_name VARCHAR(32) NOT NULL,
    agro_ecological_zone VARCHAR(96) NOT NULL,
    latitude DECIMAL(8, 4) NOT NULL,
    longitude DECIMAL(8, 4) NOT NULL,
    grace_groundwater_pctl DECIMAL(5, 2) NOT NULL COMMENT 'NASA GRACE/GRACE-FO Groundwater Percentile (0-100)',
    rabi_rainfall_mm DECIMAL(8, 2) NOT NULL,
    kharif1_rainfall_mm DECIMAL(8, 2) NOT NULL,
    kharif2_rainfall_mm DECIMAL(8, 2) NOT NULL,
    rabi_et_mm_day DECIMAL(5, 2) NOT NULL,
    kharif1_et_mm_day DECIMAL(5, 2) NOT NULL,
    kharif2_et_mm_day DECIMAL(5, 2) NOT NULL,
    default_irrigation_access BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_district_division (division_name),
    INDEX idx_district_gw (grace_groundwater_pctl)
) ENGINE=InnoDB;

-- 2. Crops Catalog Table (All 72 Crops: 52 Seasonal Rotation Crops + 20 Perennial Fruits/Plantation)
CREATE TABLE IF NOT EXISTS crops (
    crop_id INT AUTO_INCREMENT PRIMARY KEY,
    crop_name VARCHAR(64) NOT NULL UNIQUE,
    crop_category ENUM(
        'Cereal',
        'Pulse_Legume',
        'Oilseed',
        'Fiber',
        'Vegetable_Tuber_Spice',
        'Perennial_Fruit_Orchard',
        'Fallow'
    ) NOT NULL,
    botanical_family VARCHAR(48) NOT NULL,
    is_rotational BOOLEAN NOT NULL DEFAULT TRUE COMMENT 'TRUE for 60-150d annuals; FALSE for multi-year fruit trees',
    raw_sheet_season VARCHAR(24) NOT NULL,
    corrected_rotation_season ENUM('Rabi', 'Kharif-1', 'Kharif-2', 'Perennial') NOT NULL,
    transplant_month VARCHAR(32),
    growth_window VARCHAR(48),
    harvest_window VARCHAR(48),
    duration_days INT NOT NULL,
    water_req_mm DECIMAL(7, 2) NOT NULL COMMENT 'Seasonal crop evapotranspiration demand ETc (mm)',
    nitrogen_delta_kg_ha DECIMAL(6, 2) NOT NULL COMMENT 'Net soil mineral N change (+ for legumes, - for depleters)',
    baseline_margin_usd_ha DECIMAL(8, 2) NOT NULL COMMENT 'Reference gross margin (USD/ha) at national mean AP Ratio',
    national_mean_ap_ratio DECIMAL(8, 4) NOT NULL,
    drought_hardiness DECIMAL(4, 2) NOT NULL,
    waterlogging_hardiness DECIMAL(4, 2) NOT NULL,
    INDEX idx_crop_season (corrected_rotation_season, is_rotational),
    INDEX idx_crop_category (crop_category)
) ENGINE=InnoDB;

-- 3. District-Crop Historical Records (4,607 Rows from Uploaded Spreadsheet + NASA Weather)
CREATE TABLE IF NOT EXISTS district_crop_records (
    record_id INT AUTO_INCREMENT PRIMARY KEY,
    district_id INT NOT NULL,
    crop_id INT NOT NULL,
    raw_season VARCHAR(24) NOT NULL,
    corrected_season VARCHAR(24) NOT NULL,
    cultivated_area_ha DECIMAL(12, 2) NOT NULL,
    production_tonnes DECIMAL(12, 2) NOT NULL,
    ap_ratio DECIMAL(8, 4) NOT NULL COMMENT 'Production / Area yield ratio',
    relative_yield_index DECIMAL(6, 3) NOT NULL COMMENT 'District AP Ratio / National Mean AP Ratio',
    avg_temp_c DECIMAL(5, 2),
    max_temp_c DECIMAL(5, 2),
    min_temp_c DECIMAL(5, 2),
    avg_humidity_pct DECIMAL(5, 2),
    seasonal_rainfall_mm DECIMAL(8, 2),
    seasonal_solar_radiation DECIMAL(6, 3),
    seasonal_wind_speed DECIMAL(6, 3),
    seasonal_et_mm_day DECIMAL(6, 3),
    estimated_margin_usd_ha DECIMAL(8, 2) NOT NULL,
    FOREIGN KEY (district_id) REFERENCES districts(district_id) ON DELETE CASCADE,
    FOREIGN KEY (crop_id) REFERENCES crops(crop_id) ON DELETE CASCADE,
    INDEX idx_dc_lookup (district_id, corrected_season, crop_id)
) ENGINE=InnoDB;

-- 4. AI Crop-Rotation Recommendations Table (5 Strategies per District = 320 Rows across 64 Districts)
CREATE TABLE IF NOT EXISTS rotation_recommendations (
    recommendation_id INT AUTO_INCREMENT PRIMARY KEY,
    district_id INT NOT NULL,
    strategy_mode ENUM(
        'Existing Baseline',
        'Balanced',
        'Water Conservation',
        'Soil Health',
        'High-Value Vegetable'
    ) NOT NULL,
    rabi_crop_name VARCHAR(64) NOT NULL,
    kharif1_crop_name VARCHAR(64) NOT NULL,
    kharif2_crop_name VARCHAR(64) NOT NULL,
    rotation_sequence VARCHAR(160) NOT NULL,
    net_profit_usd_ha DECIMAL(8, 2) NOT NULL,
    profit_delta_vs_baseline_usd DECIMAL(8, 2) NOT NULL,
    total_water_mm DECIMAL(8, 2) NOT NULL,
    water_saved_vs_baseline_mm DECIMAL(8, 2) NOT NULL,
    final_soil_n_kg_ha DECIMAL(6, 2) NOT NULL,
    soil_n_delta_kg_ha DECIMAL(6, 2) NOT NULL,
    bare_field_days_yr INT NOT NULL,
    why_suggested TEXT NOT NULL,
    supporting_nasa_evidence JSON NOT NULL,
    agronomic_benefits JSON NOT NULL,
    uncertainty_and_gaps JSON NOT NULL,
    local_expert_review_risks JSON NOT NULL,
    FOREIGN KEY (district_id) REFERENCES districts(district_id) ON DELETE CASCADE,
    UNIQUE KEY uq_district_strategy (district_id, strategy_mode)
) ENGINE=InnoDB;

-- 5. Perennial Fruit & Orchard Suitability Table (Top Ranked Fruit Trees per District)
CREATE TABLE IF NOT EXISTS district_orchard_suitability (
    suitability_id INT AUTO_INCREMENT PRIMARY KEY,
    district_id INT NOT NULL,
    crop_id INT NOT NULL,
    rank_in_district INT NOT NULL,
    cultivated_area_ha DECIMAL(12, 2) NOT NULL,
    production_tonnes DECIMAL(12, 2) NOT NULL,
    ap_ratio DECIMAL(8, 4) NOT NULL,
    relative_yield_vs_national DECIMAL(6, 2) NOT NULL COMMENT 'Percentage vs national average yield',
    estimated_annual_margin_usd_ha DECIMAL(8, 2) NOT NULL,
    harvest_window VARCHAR(48),
    agronomic_recommendation VARCHAR(255) NOT NULL,
    FOREIGN KEY (district_id) REFERENCES districts(district_id) ON DELETE CASCADE,
    FOREIGN KEY (crop_id) REFERENCES crops(crop_id) ON DELETE CASCADE,
    INDEX idx_orchard_district (district_id, rank_in_district)
) ENGINE=InnoDB;