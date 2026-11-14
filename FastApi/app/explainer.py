"""Plain-language (Bangla / English) explanations of a recommendation.

Number safety
-------------
The LLM never writes a number. It receives the facts with *named placeholders*
(e.g. `{profit}`) and must reference every quantity, date, crop, season and district
only through those placeholders. The backend then:

  1. rejects any LLM text that still contains a digit (ASCII 0-9 or Bangla ০-৯) or an
     unknown placeholder, and retries once with the error message;
  2. fills the placeholders with the exact, pre-formatted values computed by the
     engine — so the numbers the farmer reads are byte-for-byte the model's numbers;
  3. falls back to a deterministic bilingual template if Gemini is not configured,
     unreachable, or keeps breaking the rules.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from . import cache
from .config import settings
from .names import SEASON_KEYS, crop_name, district_name, season_name
from .units import fmt_days, fmt_kg, fmt_mm, fmt_money, fmt_pct, fmt_window, to_lang_digits

logger = logging.getLogger("field_shift_web.explainer")

PROMPT_VERSION = "v1"
SECTION_KEYS = ("headline", "why", "nasa_evidence", "soil_benefits", "confidence", "expert_checks")
LIST_SECTIONS = {"nasa_evidence", "soil_benefits", "expert_checks"}

PLACEHOLDER_RE = re.compile(r"\{([a-z0-9_]+)\}")
DIGIT_RE = re.compile(r"[0-9০-৯]")

STRATEGY_GOALS = {
    "baseline": "the farmer's CURRENT rotation, used as the comparison benchmark",
    "balanced": "best overall balance of income, water use and soil health",
    "water_conservation": "use less water and protect groundwater",
    "soil_health": "rebuild soil fertility (nitrogen) and keep the field covered",
    "high_value_vegetable": "earn more from high-value vegetables",
}


# ---------------------------------------------------------------------------
# Facts + placeholders
# ---------------------------------------------------------------------------

def _direction(value: float, pos: str, neg: str, same: str = "same") -> str:
    if value > 0:
        return pos
    if value < 0:
        return neg
    return same


def build_facts(payload: Dict[str, Any], strategy_key: str, lang: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Any]]:
    """Returns (values_for_lang, meanings, qualitative_facts)."""
    cards = {s["key"]: s for s in payload["strategies"]}
    card = cards[strategy_key]
    base = cards["baseline"]
    m, r, seasons = card["metrics"], card["range_20yr"], card["seasons"]
    d = payload["district"]

    values: Dict[str, str] = {}
    meanings: Dict[str, str] = {}

    def put(name: str, value: str, meaning: str) -> None:
        values[name] = value
        meanings[name] = meaning

    put("district", district_name(d["key"], lang), "the farmer's district")
    put("years", to_lang_digits("20", lang) + (" বছর" if lang == "bn" else " years"),
        "length of the NASA weather history used")
    for i, sk in enumerate(SEASON_KEYS):
        s = seasons[i]
        put(sk, season_name(sk, lang), f"name of the {sk} season")
        put(f"crop_{sk}", crop_name(s["crop"], lang), f"recommended crop in the {sk} season")
        put(f"base_crop_{sk}", crop_name(base["seasons"][i]["crop"], lang), f"farmer's current crop in the {sk} season")
        put(f"sowing_{sk}", fmt_window(s["sowing_window"], lang), f"sowing/transplanting window of the {sk} crop")
        put(f"harvest_{sk}", fmt_window(s["harvest_window"], lang), f"harvest window of the {sk} crop")
        put(f"wetness_{sk}", fmt_pct(s["rootzone_wetness_pct"], lang), f"NASA SMAP root-zone soil wetness in {sk}")
        put(f"rain_{sk}", fmt_mm(s["rain_mm"], lang), f"NASA POWER rainfall in {sk}")
    put("dry_days_rabi", fmt_days(seasons[0]["dry_spell_days"], lang), "longest run of dry days in rabi (NASA POWER)")
    put("heat_days_kharif1", fmt_days(seasons[1]["heat_stress_days"], lang), "days above 35°C in kharif1")
    put("groundwater", fmt_pct(seasons[0]["groundwater_pctl"], lang),
        "NASA GRACE groundwater level percentile vs. history (below 30% = low)")

    put("profit", fmt_money(m["profit_bdt_bigha"], lang), "expected net profit per year of this plan")
    put("base_profit", fmt_money(base["metrics"]["profit_bdt_bigha"], lang), "net profit per year of the current crops")
    put("profit_change", fmt_money(abs(m["profit_change_bdt_bigha"]), lang),
        "size of the profit difference vs. current crops (always positive, see profit_vs_current)")
    put("water", fmt_mm(m["water_mm"], lang), "total crop water need per year")
    put("irrigation", fmt_mm(m["irrigation_mm"], lang), "part of the water need that must come from irrigation")
    put("water_saved", fmt_mm(abs(m["water_saved_mm"]), lang),
        "size of the water difference vs. current crops (always positive, see water_vs_current)")
    put("soil_n_change", fmt_kg(abs(m["soil_n_change_kg_bigha"]), lang),
        "size of the soil nitrogen change over the year (always positive, see soil_nitrogen_trend)")
    put("cover_days", fmt_days(m["cover_days"], lang), "days per year the field has a living crop")
    put("bare_days", fmt_days(m["bare_days"], lang), "days per year the field is bare")
    put("typical_profit", fmt_money(r["profit_typical_bdt_bigha"], lang), "average profit over 20 years of NASA weather")
    put("bad_year_profit", fmt_money(r["profit_bad_year_bdt_bigha"], lang), "profit in a bad year (10th percentile)")
    put("good_year_profit", fmt_money(r["profit_good_year_bdt_bigha"], lang), "profit in a good year (90th percentile)")
    put("dry_year_water", fmt_mm(r["water_dry_year_mm"], lang), "water need in a dry year")

    categories = [s["category"] for s in seasons]
    pests = [s["pest"] for s in seasons[1:]]
    qualitative = {
        "strategy_goal": STRATEGY_GOALS[strategy_key],
        "is_current_rotation": strategy_key == "baseline",
        "irrigation_access": payload["inputs"]["irrigation"],
        "profit_vs_current": _direction(m["profit_change_bdt_bigha"], "higher", "lower"),
        "water_vs_current": _direction(m["water_saved_mm"], "uses less water", "uses more water"),
        "soil_nitrogen_trend": _direction(m["soil_n_change_kg_bigha"], "adds nitrogen", "removes nitrogen", "no change"),
        "soil_vs_current": _direction(m["soil_n_change_vs_baseline_kg_bigha"], "better", "worse"),
        "groundwater_is_low": seasons[0]["groundwater_pctl"] < 30,
        "season_crop_types": dict(zip(SEASON_KEYS, categories)),
        "pest_cycle_between_seasons": [
            {"kind": p["kind"], **({"from_family": p.get("from"), "to_family": p.get("to")} if p["kind"] == "broken" else {})}
            for p in pests
        ],
        "has_vegetables": "Vegetable_Tuber_Spice" in categories,
        "has_legume": "Pulse_Legume" in categories or any(s["family"] == "Fabaceae" for s in seasons),
        "kharif1_is_fallow": seasons[1]["crop"] == "Fallow",
        "needs_irrigation": m["irrigation_mm"] > 0,
    }
    return values, meanings, qualitative


# ---------------------------------------------------------------------------
# Validation + filling
# ---------------------------------------------------------------------------

def validate_sections(sections: Any, allowed: set) -> List[str]:
    errors: List[str] = []
    if not isinstance(sections, dict):
        return ["response is not a JSON object"]
    for key in SECTION_KEYS:
        if key not in sections:
            errors.append(f"missing section '{key}'")
            continue
        val = sections[key]
        items = val if key in LIST_SECTIONS else [val]
        if key in LIST_SECTIONS and not isinstance(val, list):
            errors.append(f"section '{key}' must be a list")
            continue
        for text in items:
            if not isinstance(text, str):
                errors.append(f"section '{key}' contains a non-string")
                continue
            unknown = set(PLACEHOLDER_RE.findall(text)) - allowed
            if unknown:
                errors.append(f"unknown placeholders {sorted(unknown)} in '{key}'")
            stripped = PLACEHOLDER_RE.sub("", text)
            if DIGIT_RE.search(stripped):
                errors.append(f"digits written directly in '{key}': {text!r}")
            if "{" in stripped or "}" in stripped:
                errors.append(f"malformed placeholder in '{key}': {text!r}")
    return errors


def fill_sections(sections: Dict[str, Any], values: Dict[str, str]) -> Dict[str, Any]:
    def fill(text: str) -> str:
        return PLACEHOLDER_RE.sub(lambda mt: values[mt.group(1)], text)

    out: Dict[str, Any] = {}
    for key in SECTION_KEYS:
        val = sections[key]
        out[key] = [fill(t) for t in val if t.strip()] if key in LIST_SECTIONS else fill(val)
    return out


# ---------------------------------------------------------------------------
# Deterministic bilingual template (fallback / no API key)
# ---------------------------------------------------------------------------

_T = {
    "en": {
        "headline_base": "This is what you grow now: {crop_rabi}, then {crop_kharif1}, then {crop_kharif2}.",
        "headline": "Grow {crop_rabi} in {rabi}, {crop_kharif1} in {kharif1} and {crop_kharif2} in {kharif2}.",
        "goal": {
            "baseline": "We use your current crops to compare the other plans.",
            "balanced": "This plan gives the best balance of income, water use and soil health for {district}.",
            "water_conservation": "This plan is chosen to use less water and protect groundwater in {district}.",
            "soil_health": "This plan is chosen to rebuild soil fertility in {district}.",
            "high_value_vegetable": "This plan is chosen to earn more from high-value vegetables in {district}.",
        },
        "profit_base": "Expected profit is {profit}.",
        "profit_higher": "Expected profit is {profit}, which is {profit_change} more than your current crops.",
        "profit_lower": "Expected profit is {profit}, which is {profit_change} less than your current crops.",
        "profit_same": "Expected profit is {profit}, about the same as now.",
        "water_less": "It needs {water_saved} less water in a year.",
        "water_more": "It needs {water_saved} more water in a year.",
        "ev_gw": "NASA GRACE satellites (they weigh underground water) show groundwater at the {groundwater} level compared with past years.",
        "ev_gw_low": " This is low, so tubewells may struggle in the dry season.",
        "ev_smap": "NASA SMAP (soil moisture from space): the root zone is {wetness_rabi} wet in {rabi}, {wetness_kharif1} in {kharif1} and {wetness_kharif2} in {kharif2}.",
        "ev_rain": "NASA POWER rainfall: {rain_rabi} in {rabi}, with up to {dry_days_rabi} without rain in a row, compared with {rain_kharif2} in {kharif2}.",
        "ev_water": "The crops need {water} of water in total; {irrigation} of it must come from irrigation.",
        "ev_water_rain": "The crops need {water} of water in total, and rain can cover all of it.",
        "soil_add": "Soil nitrogen goes up by {soil_n_change} over the year.",
        "soil_remove": "Soil nitrogen goes down by {soil_n_change} over the year, so plan for fertiliser or compost.",
        "soil_flat": "Soil nitrogen stays about the same over the year.",
        "pest_broken": "Changing crop families between seasons breaks pest and disease cycles.",
        "pest_repeat": "The same crop family is grown twice in a row, which can build up pests.",
        "cover": "Living roots cover the field for {cover_days} a year; it is bare for {bare_days}.",
        "confidence": "Using the last {years} of NASA weather, profit is about {typical_profit} in a normal year, {bad_year_profit} in a bad year and {good_year_profit} in a good year.",
        "chk_turnaround": "Check that {crop_rabi} can be harvested ({harvest_rabi}) in time to sow {crop_kharif1} ({sowing_kharif1}).",
        "chk_market": "Ask your local agriculture officer (DAE) about markets and cold storage for vegetables.",
        "chk_drain": "Make sure the field drains well; sudden rain can rot pulses and winter vegetables.",
        "chk_soil": "Satellite data covers a wide area. Test your own soil before you decide.",
    },
    "bn": {
        "headline_base": "এটি আপনার বর্তমান চাষ: {crop_rabi}, তারপর {crop_kharif1}, তারপর {crop_kharif2}।",
        "headline": "{rabi} মৌসুমে {crop_rabi}, {kharif1} মৌসুমে {crop_kharif1} এবং {kharif2} মৌসুমে {crop_kharif2} চাষ করুন।",
        "goal": {
            "baseline": "অন্য পরিকল্পনাগুলোর সাথে তুলনা করতে আমরা আপনার বর্তমান চাষ ব্যবহার করি।",
            "balanced": "{district} জেলার জন্য এই পরিকল্পনায় আয়, পানি ও মাটির স্বাস্থ্যের মধ্যে সবচেয়ে ভালো ভারসাম্য আছে।",
            "water_conservation": "{district} জেলায় কম পানি খরচ ও ভূগর্ভস্থ পানি রক্ষার জন্য এই পরিকল্পনা বেছে নেওয়া হয়েছে।",
            "soil_health": "{district} জেলায় মাটির উর্বরতা ফিরিয়ে আনার জন্য এই পরিকল্পনা বেছে নেওয়া হয়েছে।",
            "high_value_vegetable": "{district} জেলায় দামি সবজি চাষ করে বেশি আয়ের জন্য এই পরিকল্পনা বেছে নেওয়া হয়েছে।",
        },
        "profit_base": "সম্ভাব্য লাভ {profit}।",
        "profit_higher": "সম্ভাব্য লাভ {profit}, যা আপনার বর্তমান চাষের চেয়ে {profit_change} বেশি।",
        "profit_lower": "সম্ভাব্য লাভ {profit}, যা আপনার বর্তমান চাষের চেয়ে {profit_change} কম।",
        "profit_same": "সম্ভাব্য লাভ {profit}, যা এখনকার মতোই।",
        "water_less": "এতে বছরে {water_saved} কম পানি লাগে।",
        "water_more": "এতে বছরে {water_saved} বেশি পানি লাগে।",
        "ev_gw": "নাসার GRACE স্যাটেলাইট (যা মাটির নিচের পানি মাপে) অনুযায়ী অতীতের তুলনায় আপনার এলাকার ভূগর্ভস্থ পানি {groundwater} স্তরে আছে।",
        "ev_gw_low": " এটি কম, তাই শুকনো মৌসুমে নলকূপে পানি কমে যেতে পারে।",
        "ev_smap": "নাসার SMAP (মহাকাশ থেকে মাটির আর্দ্রতা মাপে): শিকড়ের মাটিতে আর্দ্রতা {rabi} মৌসুমে {wetness_rabi}, {kharif1} মৌসুমে {wetness_kharif1} এবং {kharif2} মৌসুমে {wetness_kharif2}।",
        "ev_rain": "নাসার POWER বৃষ্টির তথ্য: {rabi} মৌসুমে {rain_rabi} বৃষ্টি, টানা সর্বোচ্চ {dry_days_rabi} বৃষ্টিহীন; আর {kharif2} মৌসুমে {rain_kharif2}।",
        "ev_water": "ফসলের মোট পানির চাহিদা {water}, যার মধ্যে {irrigation} সেচ দিয়ে দিতে হবে।",
        "ev_water_rain": "ফসলের মোট পানির চাহিদা {water}, যা বৃষ্টির পানিতেই মিটে যায়।",
        "soil_add": "সারা বছরে মাটিতে নাইট্রোজেন {soil_n_change} বাড়ে।",
        "soil_remove": "সারা বছরে মাটি থেকে নাইট্রোজেন {soil_n_change} কমে, তাই সার বা কম্পোস্টের পরিকল্পনা রাখুন।",
        "soil_flat": "সারা বছরে মাটির নাইট্রোজেন প্রায় একই থাকে।",
        "pest_broken": "মৌসুমে মৌসুমে ভিন্ন পরিবারের ফসল চাষ করলে পোকামাকড় ও রোগের চক্র ভেঙে যায়।",
        "pest_repeat": "পরপর একই পরিবারের ফসল চাষ করায় পোকামাকড় বাড়তে পারে।",
        "cover": "বছরে {cover_days} জমিতে ফসল থাকে; {bare_days} জমি খালি থাকে।",
        "confidence": "গত {years}ের নাসার আবহাওয়া তথ্য অনুযায়ী, স্বাভাবিক বছরে লাভ প্রায় {typical_profit}, খারাপ বছরে {bad_year_profit} এবং ভালো বছরে {good_year_profit}।",
        "chk_turnaround": "{crop_rabi} কাটা ({harvest_rabi}) শেষ করে সময়মতো {crop_kharif1} বোনা ({sowing_kharif1}) সম্ভব কিনা যাচাই করুন।",
        "chk_market": "সবজির বাজার ও হিমাগার সম্পর্কে স্থানীয় কৃষি কর্মকর্তার (DAE) সাথে কথা বলুন।",
        "chk_drain": "জমিতে পানি নিষ্কাশন ভালো কিনা দেখুন; হঠাৎ বৃষ্টিতে ডাল ও শীতের সবজির শিকড় পচে যেতে পারে।",
        "chk_soil": "স্যাটেলাইটের তথ্য বড় এলাকার। সিদ্ধান্ত নেওয়ার আগে নিজের জমির মাটি পরীক্ষা করুন।",
    },
}


def template_sections(strategy_key: str, q: Dict[str, Any], lang: str) -> Dict[str, Any]:
    t = _T[lang]
    is_base = strategy_key == "baseline"

    why = [t["goal"][strategy_key]]
    if is_base:
        why.append(t["profit_base"])
    else:
        why.append({"higher": t["profit_higher"], "lower": t["profit_lower"]}.get(q["profit_vs_current"], t["profit_same"]))
        if q["water_vs_current"] == "uses less water":
            why.append(t["water_less"])
        elif q["water_vs_current"] == "uses more water":
            why.append(t["water_more"])

    evidence = [
        t["ev_gw"] + (t["ev_gw_low"] if q["groundwater_is_low"] else ""),
        t["ev_smap"],
        t["ev_rain"],
        t["ev_water"] if q["needs_irrigation"] else t["ev_water_rain"],
    ]

    soil = [{"adds nitrogen": t["soil_add"], "removes nitrogen": t["soil_remove"]}.get(q["soil_nitrogen_trend"], t["soil_flat"])]
    kinds = {p["kind"] for p in q["pest_cycle_between_seasons"]}
    if "repeat" in kinds:
        soil.append(t["pest_repeat"])
    elif "broken" in kinds:
        soil.append(t["pest_broken"])
    soil.append(t["cover"])

    checks = []
    if not q["kharif1_is_fallow"]:
        checks.append(t["chk_turnaround"])
    if q["has_vegetables"]:
        checks.append(t["chk_market"])
    checks += [t["chk_drain"], t["chk_soil"]]

    sep = " "
    return {
        "headline": t["headline_base"] if is_base else t["headline"],
        "why": sep.join(why),
        "nasa_evidence": evidence,
        "soil_benefits": soil,
        "confidence": t["confidence"],
        "expert_checks": checks,
    }


# ---------------------------------------------------------------------------
# Gemini (REST via httpx — no extra SDK needed)
# ---------------------------------------------------------------------------

_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "headline": {"type": "STRING"},
        "why": {"type": "STRING"},
        "nasa_evidence": {"type": "ARRAY", "items": {"type": "STRING"}},
        "soil_benefits": {"type": "ARRAY", "items": {"type": "STRING"}},
        "confidence": {"type": "STRING"},
        "expert_checks": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": list(SECTION_KEYS),
    "propertyOrdering": list(SECTION_KEYS),
}


def _system_prompt(lang: str) -> str:
    language = (
        "Bangla (standard Bengali script). Use simple, everyday village words a farmer understands; "
        "keep NASA mission names (GRACE, SMAP, POWER, MODIS) in English letters."
        if lang == "bn"
        else "simple, plain English (short words, no jargon)"
    )
    return f"""You are a friendly agricultural extension advisor in Bangladesh. You explain a crop-rotation
recommendation produced by a NASA-data model to a smallholder farmer. Write in {language}.

STRICT RULES — your answer is machine-checked and rejected if you break them:
1. NEVER write any digit or number yourself (no 0-9, no ০-৯, and no numbers spelled out as words).
2. Whenever you mention a quantity, money, percentage, date, crop, season or district, insert the matching
   placeholder EXACTLY as given, in curly braces, e.g. {{profit}} or {{crop_rabi}}. Each placeholder is
   replaced with its exact value later, so write the sentence so it reads naturally after replacement.
   Money placeholders already include "per bigha" and the Taka sign; do not add units after placeholders.
3. Only use placeholders from the provided list. Do not invent facts, numbers, crops or risks.
4. Difference placeholders (profit_change, water_saved, soil_n_change) are always positive sizes — use the
   direction facts (higher/lower, less/more water, adds/removes nitrogen) to choose your words.
5. Short sentences (max ~25 words). When naming a NASA satellite, say in a few words what it measures.
   Never use dataset product codes.

Sections:
- headline: one sentence telling the farmer what to grow in each season.
- why: two or three sentences on why this plan suits them (goal, profit vs current crops, water).
- nasa_evidence: two to four bullets on what NASA data shows (groundwater, soil wetness, rainfall, water need).
- soil_benefits: two to four bullets (nitrogen, pest/disease cycle, field cover).
- confidence: one or two sentences on how sure we are, using the normal/bad/good-year profits.
- expert_checks: two to four bullets of things to confirm with the local agriculture officer (DAE)
  or on their own field (sowing turnaround, market/storage if vegetables, drainage, soil test)."""


def _call_gemini(system: str, user: str) -> Dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": 0.4,
            "responseMimeType": "application/json",
            "responseSchema": _RESPONSE_SCHEMA,
        },
    }
    resp = httpx.post(
        url,
        headers={"x-goog-api-key": settings.gemini_api_key, "Content-Type": "application/json"},
        json=body,
        timeout=settings.gemini_timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    parts = data["candidates"][0]["content"]["parts"]
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    return json.loads(text)


def explain(payload: Dict[str, Any], strategy_key: str, lang: str) -> Dict[str, Any]:
    values, meanings, qualitative = build_facts(payload, strategy_key, lang)
    allowed = set(values)

    # For the LLM's reasoning, show English-formatted values (ASCII digits) next to each placeholder.
    en_values, _, _ = build_facts(payload, strategy_key, "en")
    llm_input = {
        "placeholders": {k: {"meaning": meanings[k], "value": en_values[k]} for k in values},
        "facts": qualitative,
    }
    user_prompt = (
        "Explain this recommendation to the farmer. Data:\n"
        + json.dumps(llm_input, ensure_ascii=False, indent=1)
    )

    cache_key = hashlib.sha256(
        json.dumps(
            {"v": PROMPT_VERSION, "model": settings.gemini_model, "lang": lang, "input": llm_input},
            sort_keys=True, ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    if settings.gemini_api_key:
        cached = cache.get(cache_key)
        if cached is not None:
            return {"sections": fill_sections(cached, values), "source": "gemini_cached"}

        prompt = user_prompt
        for attempt in range(2):
            try:
                raw = _call_gemini(_system_prompt(lang), prompt)
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                logger.warning("Gemini call failed (%s): %s", type(exc).__name__, exc)
                break
            errors = validate_sections(raw, allowed)
            if not errors:
                clean = {k: raw[k] for k in SECTION_KEYS}
                cache.put(cache_key, lang, settings.gemini_model, clean)
                return {"sections": fill_sections(clean, values), "source": "gemini"}
            logger.info("Gemini answer rejected (attempt %d): %s", attempt + 1, errors[:3])
            prompt = (
                user_prompt
                + "\n\nYour previous answer was REJECTED for these reasons:\n- "
                + "\n- ".join(errors[:8])
                + "\nRewrite it following the rules exactly."
            )

    sections = template_sections(strategy_key, qualitative, lang)
    return {"sections": fill_sections(sections, values), "source": "template"}


def gemini_configured() -> bool:
    return bool(settings.gemini_api_key)
