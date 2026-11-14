import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "FastApi"))

from app import explainer  # noqa: E402
from app.explainer import (  # noqa: E402
    SECTION_KEYS,
    build_facts,
    fill_sections,
    template_sections,
    validate_sections,
)


def _season(key, crop, category, family, pest):
    return {
        "season": key, "crop": crop, "category": category, "family": family,
        "sowing_window": "Nov 01 - Nov 20", "harvest_window": "Feb-Mar",
        "season_days": 135, "active_days": 105, "bare_days": 30,
        "water_mm": 200.0, "irrigation_mm": 80.0, "rain_mm": 45.5, "mean_temp_c": 20.1,
        "heat_stress_days": 3, "dry_spell_days": 41, "rootzone_wetness_pct": 38,
        "groundwater_pctl": 22.0, "soil_n_change_kg_bigha": 4.3, "pest": pest,
    }


def _card(key, crops, profit, change, water_saved, n_change):
    seasons = [
        _season("rabi", crops[0], "Pulse_Legume", "Fabaceae", {"kind": "first"}),
        _season("kharif1", crops[1], "Fiber", "Malvaceae", {"kind": "broken", "from": "Fabaceae", "to": "Malvaceae"}),
        _season("kharif2", crops[2], "Cereal", "Poaceae", {"kind": "broken", "from": "Malvaceae", "to": "Poaceae"}),
    ]
    return {
        "key": key,
        "crops": list(crops),
        "metrics": {
            "profit_bdt_bigha": profit, "profit_change_bdt_bigha": change,
            "water_mm": 1240.0, "irrigation_mm": 610.5, "water_saved_mm": water_saved,
            "soil_n_kg_bigha": 12.1, "soil_n_change_kg_bigha": n_change,
            "soil_n_change_vs_baseline_kg_bigha": 6.0, "cover_days": 355, "bare_days": 10,
        },
        "range_20yr": {
            "profit_typical_bdt_bigha": 23450, "profit_spread_bdt_bigha": 1200,
            "profit_bad_year_bdt_bigha": 21000, "profit_good_year_bdt_bigha": 25010,
            "water_typical_mm": 1250.0, "water_dry_year_mm": 1390.0,
        },
        "seasons": seasons,
    }


@pytest.fixture()
def payload():
    return {
        "district": {"key": "Rajshahi"},
        "inputs": {"irrigation": True},
        "strategies": [
            _card("baseline", ("Boro", "Fallow", "Aman"), 20000, 0, 0.0, -9.1),
            _card("water_conservation", ("Lentil", "Jute", "Aman"), 23560, 3560, 655.0, 2.1),
        ],
    }


def test_validator_rejects_digits_in_any_script():
    allowed = {"profit"}
    ok = {k: ([] if k in explainer.LIST_SECTIONS else "Profit is {profit}.") for k in SECTION_KEYS}
    assert validate_sections(ok, allowed) == []

    bad_ascii = dict(ok, why="Profit is 23,560 taka.")
    bad_bangla = dict(ok, why="লাভ ২৩,৫৬০ টাকা।")
    unknown = dict(ok, why="Profit is {made_up}.")
    missing = {k: v for k, v in ok.items() if k != "why"}
    assert validate_sections(bad_ascii, allowed)
    assert validate_sections(bad_bangla, allowed)
    assert validate_sections(unknown, allowed)
    assert validate_sections(missing, allowed)


def test_placeholder_names_with_digits_are_allowed(payload):
    values, _, _ = build_facts(payload, "water_conservation", "en")
    text = {k: ([] if k in explainer.LIST_SECTIONS else "{crop_kharif1} in {kharif1}.") for k in SECTION_KEYS}
    assert validate_sections(text, set(values)) == []


@pytest.mark.parametrize("lang", ["en", "bn"])
def test_filled_numbers_match_engine_values_exactly(payload, lang):
    values, _, q = build_facts(payload, "water_conservation", lang)
    sections = template_sections("water_conservation", q, lang)
    assert validate_sections(sections, set(values)) == []
    filled = fill_sections(sections, values)
    flat = " ".join(filled[k] if isinstance(filled[k], str) else " ".join(filled[k]) for k in SECTION_KEYS)
    expected_profit = "৳23,560 per bigha" if lang == "en" else "বিঘাপ্রতি ৳২৩,৫৬০"
    expected_change = "৳3,560 per bigha" if lang == "en" else "বিঘাপ্রতি ৳৩,৫৬০"
    assert expected_profit in flat
    assert expected_change in flat
    assert "{" not in flat and "}" not in flat


def test_template_is_used_without_api_key(payload, monkeypatch):
    from dataclasses import replace

    monkeypatch.setattr(explainer, "settings", replace(explainer.settings, gemini_api_key=""))
    out = explainer.explain(payload, "water_conservation", "bn")
    assert out["source"] == "template"
    assert "রবি" in out["sections"]["headline"]
