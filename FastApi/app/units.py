"""Deterministic unit conversion and number formatting.

The model works in USD per hectare and kg per hectare. Farmers in Bangladesh think in
Taka per bigha, so every money / per-area value is converted here, ONCE, before it
reaches the frontend or the LLM. After this point the values are never changed.
"""

from __future__ import annotations

import re
from typing import Optional

from .config import settings

_BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")


def money_bdt_per_bigha(
    usd_per_ha: float,
    usd_to_bdt: Optional[float] = None,
    ha_per_bigha: Optional[float] = None,
) -> int:
    """USD/ha -> Taka/bigha, rounded to the nearest 10 Taka."""
    rate = settings.usd_to_bdt if usd_to_bdt is None else usd_to_bdt
    area = settings.ha_per_bigha if ha_per_bigha is None else ha_per_bigha
    return int(round(float(usd_per_ha) * rate * area / 10.0)) * 10


def kg_per_bigha(kg_per_ha: float, ha_per_bigha: Optional[float] = None) -> float:
    """kg/ha -> kg/bigha, rounded to 0.1 kg."""
    area = settings.ha_per_bigha if ha_per_bigha is None else ha_per_bigha
    return _clean(round(float(kg_per_ha) * area, 1))


def round1(value: float) -> float:
    return _clean(round(float(value), 1))


def _clean(value: float) -> float:
    # Avoid "-0.0" showing up in the UI.
    return value + 0.0 if value != 0 else 0.0


# ---------------------------------------------------------------------------
# Formatting (South-Asian digit grouping: 12,34,567 — same as Intl 'bn-BD'/'en-IN')
# ---------------------------------------------------------------------------

def _group(int_digits: str) -> str:
    if len(int_digits) <= 3:
        return int_digits
    head, tail = int_digits[:-3], int_digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups + [tail])


def format_number(value: float, lang: str, decimals: int = 0) -> str:
    v = float(value)
    sign = "-" if v < 0 else ""
    v = abs(v)
    text = f"{v:.{decimals}f}"
    if decimals:
        int_part, frac = text.split(".")
        frac = frac.rstrip("0")
        text = _group(int_part) + (f".{frac}" if frac else "")
    else:
        text = _group(text)
    text = sign + text
    return to_lang_digits(text, lang)


def to_lang_digits(text: str, lang: str) -> str:
    return text.translate(_BN_DIGITS) if lang == "bn" else text


def fmt_money(value: float, lang: str) -> str:
    n = format_number(value, lang)
    return f"বিঘাপ্রতি ৳{n}" if lang == "bn" else f"৳{n} per bigha"


def fmt_kg(value: float, lang: str) -> str:
    n = format_number(value, lang, decimals=1)
    return f"বিঘাপ্রতি {n} কেজি" if lang == "bn" else f"{n} kg per bigha"


def fmt_mm(value: float, lang: str) -> str:
    n = format_number(value, lang, decimals=1)
    return f"{n} মিমি" if lang == "bn" else f"{n} mm"


def fmt_days(value: float, lang: str) -> str:
    n = format_number(value, lang)
    return f"{n} দিন" if lang == "bn" else f"{n} days"


def fmt_pct(value: float, lang: str) -> str:
    return f"{format_number(value, lang, decimals=1)}%"


_MONTHS_BN = {
    "Jan": "জানুয়ারি", "Feb": "ফেব্রুয়ারি", "Mar": "মার্চ", "Apr": "এপ্রিল", "May": "মে",
    "Jun": "জুন", "Jul": "জুলাই", "Aug": "আগস্ট", "Sep": "সেপ্টেম্বর", "Oct": "অক্টোবর",
    "Nov": "নভেম্বর", "Dec": "ডিসেম্বর",
}
_MONTH_RE = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b")


def fmt_window(window: str, lang: str) -> str:
    """Translate sowing/harvest windows such as 'Nov 01 - Nov 20' or 'Feb-Mar'."""
    if not window or window.strip().lower() == "none":
        return "—"
    if lang != "bn":
        return window
    return to_lang_digits(_MONTH_RE.sub(lambda m: _MONTHS_BN[m.group(1)], window), lang)
