import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "FastApi"))

from app.units import (  # noqa: E402
    fmt_money,
    fmt_window,
    format_number,
    kg_per_bigha,
    money_bdt_per_bigha,
)


def test_money_conversion_rounds_to_10_taka():
    # 1000 USD/ha * 122 BDT/USD * 0.13355 ha/bigha = 16,293.1 -> 16,290
    assert money_bdt_per_bigha(1000.0, usd_to_bdt=122.0, ha_per_bigha=0.13355) == 16290
    assert money_bdt_per_bigha(0.0, usd_to_bdt=122.0, ha_per_bigha=0.13355) == 0
    assert money_bdt_per_bigha(-250.0, usd_to_bdt=122.0, ha_per_bigha=0.13355) == -4070


def test_kg_conversion():
    assert kg_per_bigha(75.0, ha_per_bigha=0.13355) == 10.0
    assert kg_per_bigha(32.0, ha_per_bigha=0.13355) == 4.3
    assert str(kg_per_bigha(0.0001, ha_per_bigha=0.13355)) == "0.0"  # never "-0.0"


def test_south_asian_grouping_and_bangla_digits():
    assert format_number(1234567, "en") == "12,34,567"
    assert format_number(1234567, "bn") == "১২,৩৪,৫৬৭"
    assert format_number(-4070, "en") == "-4,070"
    assert format_number(4.30, "en", decimals=1) == "4.3"
    assert format_number(10.0, "en", decimals=1) == "10"


def test_money_strings():
    assert fmt_money(16290, "en") == "৳16,290 per bigha"
    assert fmt_money(16290, "bn") == "বিঘাপ্রতি ৳১৬,২৯০"


def test_window_translation():
    assert fmt_window("Nov 01 - Nov 20", "en") == "Nov 01 - Nov 20"
    assert fmt_window("Nov 01 - Nov 20", "bn") == "নভেম্বর ০১ - নভেম্বর ২০"
    assert fmt_window("Feb-Mar", "bn") == "ফেব্রুয়ারি-মার্চ"
    assert fmt_window("None", "bn") == "—"
