"""Bilingual (English / Bangla) display names for crops, districts, divisions and seasons.

Keys are the canonical names used inside the model (`CROP_CATALOG[i].name`,
`DISTRICT_METADATA` keys). The biotech team should review the Bangla crop names.
"""

from __future__ import annotations

from typing import Dict, Tuple

# crop key -> (English display, Bangla display)
CROP_NAMES: Dict[str, Tuple[str, str]] = {
    # Cereals
    "Aman": ("T. Aman Rice", "আমন ধান"),
    "Boro": ("Boro Rice", "বোরো ধান"),
    "Aus": ("Aus Rice", "আউশ ধান"),
    "Wheat": ("Wheat", "গম"),
    "Maize 1": ("Maize (summer)", "ভুট্টা (খরিফ)"),
    "Maize 2": ("Maize (winter)", "ভুট্টা (রবি)"),
    "Cheena": ("Proso millet (Cheena)", "চিনা"),
    # Pulses & legumes
    "Lentil": ("Lentil", "মসুর ডাল"),
    "Gram": ("Chickpea (Gram)", "ছোলা"),
    "Motor": ("Field pea (Motor)", "মটর"),
    "Mungbean": ("Mungbean", "মুগ ডাল"),
    "Mug": ("Mungbean", "মুগ ডাল"),
    "Mashkalai": ("Black gram (Mashkalai)", "মাষকলাই"),
    "Arhar": ("Pigeon pea (Arhar)", "অড়হর"),
    "Groundnut": ("Groundnut", "চিনাবাদাম"),
    "Beans": ("Country bean (Shim)", "শিম"),
    "Barbati": ("Yardlong bean (Barbati)", "বরবটি"),
    # Oilseeds & fibre
    "Rape & Mustard": ("Mustard", "সরিষা"),
    "Sesame": ("Sesame", "তিল"),
    "Jute": ("Jute", "পাট"),
    "Tobacco": ("Tobacco", "তামাক"),
    # Winter vegetables, tubers & spices
    "Potato": ("Potato", "আলু"),
    "Cabbage": ("Cabbage", "বাঁধাকপি"),
    "Cauliflower": ("Cauliflower", "ফুলকপি"),
    "Carrot": ("Carrot", "গাজর"),
    "Radish": ("Radish", "মুলা"),
    "Shalgom": ("Turnip (Shalgom)", "শালগম"),
    "Sweet Potato": ("Sweet potato", "মিষ্টি আলু"),
    "Onion": ("Onion", "পেঁয়াজ"),
    "Garlic": ("Garlic", "রসুন"),
    "Chili": ("Chilli", "মরিচ"),
    "Ginger": ("Ginger", "আদা"),
    "Lal Shak": ("Red amaranth (Lal Shak)", "লাল শাক"),
    "Palong Shak": ("Spinach (Palong Shak)", "পালং শাক"),
    "Laushak": ("Bottle-gourd leaves (Lau Shak)", "লাউ শাক"),
    "Pumpkin": ("Sweet pumpkin", "মিষ্টি কুমড়া"),
    "Gourd": ("Bottle gourd", "লাউ"),
    # Summer / monsoon vegetables
    "Lady’s Finger": ("Okra (Dherosh)", "ঢেঁড়স"),
    "Lady's Finger": ("Okra (Dherosh)", "ঢেঁড়স"),
    "Jhinga": ("Ridge gourd (Jhinga)", "ঝিঙা"),
    "Kakrol": ("Spiny gourd (Kakrol)", "কাঁকরোল"),
    "Chalkumra": ("Ash gourd (Chalkumra)", "চালকুমড়া"),
    "Karala": ("Bitter gourd (Korola)", "করলা"),
    "Cucumber": ("Cucumber", "শসা"),
    "Patal": ("Pointed gourd (Potol)", "পটল"),
    "Danta": ("Stem amaranth (Data)", "ডাঁটা"),
    "Danta Shak": ("Amaranth greens (Data Shak)", "ডাঁটা শাক"),
    "Kolmi Shak": ("Water spinach (Kolmi Shak)", "কলমি শাক"),
    "Puishak": ("Malabar spinach (Pui Shak)", "পুঁই শাক"),
    "Mukhi Kachu": ("Taro (Mukhi Kochu)", "মুখী কচু"),
    "Oal Kachu": ("Elephant-foot yam (Ol Kochu)", "ওল কচু"),
    # Rest
    "Fallow": ("Fallow (no crop)", "পতিত (জমি খালি)"),
}

DISTRICT_BN: Dict[str, str] = {
    "Bagerhat": "বাগেরহাট", "Bandarban": "বান্দরবান", "Barguna": "বরগুনা", "Barishal": "বরিশাল",
    "Bhola": "ভোলা", "Bogura": "বগুড়া", "Brahmanbaria": "ব্রাহ্মণবাড়িয়া", "Chandpur": "চাঁদপুর",
    "Chapai Nawabganj": "চাঁপাইনবাবগঞ্জ", "Chattogram": "চট্টগ্রাম", "Chuadanga": "চুয়াডাঙ্গা",
    "CoxsBazar": "কক্সবাজার", "Cumilla": "কুমিল্লা", "Dhaka": "ঢাকা", "Dinajpur": "দিনাজপুর",
    "Faridpur": "ফরিদপুর", "Feni": "ফেনী", "Gaibandha": "গাইবান্ধা", "Gazipur": "গাজীপুর",
    "Gopalganj": "গোপালগঞ্জ", "Habiganj": "হবিগঞ্জ", "Jamalpur": "জামালপুর", "Jashore": "যশোর",
    "Jhallokati": "ঝালকাঠি", "Jhenaidah": "ঝিনাইদহ", "Joypurhat": "জয়পুরহাট",
    "Khagrachari": "খাগড়াছড়ি", "Khulna": "খুলনা", "Kishoreganj": "কিশোরগঞ্জ", "Kurigram": "কুড়িগ্রাম",
    "Kushtia": "কুষ্টিয়া", "Lakshmipur": "লক্ষ্মীপুর", "Lalmonirhat": "লালমনিরহাট",
    "Madaripur": "মাদারীপুর", "Magura": "মাগুরা", "Manikganj": "মানিকগঞ্জ", "Meherpur": "মেহেরপুর",
    "Moulvibazar": "মৌলভীবাজার", "Munshiganj": "মুন্সীগঞ্জ", "Mymensingh": "ময়মনসিংহ",
    "Naogaon": "নওগাঁ", "Narail": "নড়াইল", "Narayanganj": "নারায়ণগঞ্জ", "Narsingdi": "নরসিংদী",
    "Natore": "নাটোর", "Netrokona": "নেত্রকোনা", "Nilphamari": "নীলফামারী", "Noakhali": "নোয়াখালী",
    "Pabna": "পাবনা", "Panchagar": "পঞ্চগড়", "Patuakhali": "পটুয়াখালী", "Pirojpur": "পিরোজপুর",
    "Rajbari": "রাজবাড়ী", "Rajshahi": "রাজশাহী", "Rangamati": "রাঙ্গামাটি", "Rangpur": "রংপুর",
    "Satkhira": "সাতক্ষীরা", "Shariatpur": "শরীয়তপুর", "Sherpur": "শেরপুর", "Sirajganj": "সিরাজগঞ্জ",
    "Sunamganj": "সুনামগঞ্জ", "Sylhet": "সিলেট", "Tangail": "টাঙ্গাইল", "Thakurgaon": "ঠাকুরগাঁও",
}

DISTRICT_EN: Dict[str, str] = {"CoxsBazar": "Cox's Bazar"}

DIVISION_BN: Dict[str, str] = {
    "Barishal": "বরিশাল", "Chattogram": "চট্টগ্রাম", "Dhaka": "ঢাকা", "Khulna": "খুলনা",
    "Mymensingh": "ময়মনসিংহ", "Rajshahi": "রাজশাহী", "Rangpur": "রংপুর", "Sylhet": "সিলেট",
}

SEASON_KEYS = ("rabi", "kharif1", "kharif2")
SEASON_MODEL_NAMES = ("Rabi", "Kharif-1", "Kharif-2")
SEASON_NAMES: Dict[str, Tuple[str, str]] = {
    "rabi": ("Rabi (winter)", "রবি (শীত)"),
    "kharif1": ("Kharif-1 (summer)", "খরিফ-১ (গ্রীষ্ম)"),
    "kharif2": ("Kharif-2 (monsoon)", "খরিফ-২ (বর্ষা)"),
}

STRATEGY_KEYS = ("baseline", "balanced", "water_conservation", "soil_health", "high_value_vegetable")


def crop_name(key: str, lang: str) -> str:
    en, bn = CROP_NAMES.get(key, (key, key))
    return bn if lang == "bn" else en


def district_name(key: str, lang: str) -> str:
    if lang == "bn":
        return DISTRICT_BN.get(key, key)
    return DISTRICT_EN.get(key, key)


def season_name(key: str, lang: str) -> str:
    en, bn = SEASON_NAMES[key]
    return bn if lang == "bn" else en
