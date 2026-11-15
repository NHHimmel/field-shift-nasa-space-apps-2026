import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frontend_dist_built():
    dist = ROOT / "React" / "dist"
    assert dist.exists(), "React/dist should exist"
    index_html = dist / "index.html"
    assert index_html.exists(), "React/dist/index.html should exist"
    content = index_html.read_text(encoding="utf-8")
    assert "<div id=\"root\">" in content or "id=\"root\"" in content


def test_nasa_products_links_valid():
    ts_file = ROOT / "React" / "src" / "lib" / "nasaProducts.ts"
    assert ts_file.exists()
    content = ts_file.read_text(encoding="utf-8")
    assert "https://power.larc.nasa.gov/" in content
    assert "https://smap.jpl.nasa.gov/" in content
    assert "https://grace.jpl.nasa.gov/" in content
    assert "https://modis.gsfc.nasa.gov/" in content
    assert "https://www.earthdata.nasa.gov/" in content


def test_i18n_strings_keys_match():
    strings_file = ROOT / "React" / "src" / "i18n" / "strings.ts"
    assert strings_file.exists()
    content = strings_file.read_text(encoding="utf-8")

    # Extract all string keys in en = { ... } and bn: Record<...> = { ... }
    en_match = re.search(r"const en = \{([\s\S]*?)\n\}", content)
    bn_match = re.search(r"const bn: Record<StringKey, string> = \{([\s\S]*?)\n\}", content)
    assert en_match and bn_match

    en_keys = set(re.findall(r"'([a-zA-Z0-9_.]+)':", en_match.group(1)))
    bn_keys = set(re.findall(r"'([a-zA-Z0-9_.]+)':", bn_match.group(1)))

    assert len(en_keys) > 80
    assert en_keys == bn_keys, f"Keys mismatch: in en only: {en_keys - bn_keys}, in bn only: {bn_keys - en_keys}"

    # Verify our new feature keys exist
    for required in [
        "loc.title",
        "loc.aez",
        "loc.coords",
        "climate.sectionTitle",
        "climate.rain.title",
        "climate.smap.title",
        "climate.temp.title",
        "climate.grace.title",
        "exp.evidenceTitle",
        "link.nasaProduct",
    ]:
        assert required in en_keys, f"Missing required key: {required}"


def test_components_exist():
    src = ROOT / "React" / "src" / "components"
    assert (src / "BangladeshMap.tsx").exists()
    assert (src / "DistrictLocationCard.tsx").exists()
    assert (src / "NasaClimateCards.tsx").exists()
    assert (src / "Explanation.tsx").exists()
    assert (src / "EvidencePanel.tsx").exists()
    assert (src / "HeroVisual.tsx").exists()


def test_planted_crop_conditioning():
    import sys
    sys.path.insert(0, str(ROOT / "FastApi"))
    from app import engine

    # Test 1: When Rabi is planted as Mustard and other seasons are auto
    res = engine.recommend("Rajshahi", True, "medium", ("Mustard", "auto", "auto"), [])
    assert res["inputs"]["planted_seasons"] == {"rabi": "Mustard"}
    for strat in res["strategies"]:
        assert strat["crops"][0] in ("Rape & Mustard", "Mustard")

    # Test 2: When both Rabi and Kharif-1 are planted
    res2 = engine.recommend("Rajshahi", True, "medium", ("Mustard", "Jute", "auto"), [])
    assert res2["inputs"]["planted_seasons"] == {"rabi": "Mustard", "kharif1": "Jute"}
    for strat in res2["strategies"]:
        assert strat["crops"][0] in ("Rape & Mustard", "Mustard")
        assert strat["crops"][1] == "Jute"

    # Test 3: Unconstrained when None
    res3 = engine.recommend("Rajshahi", True, "medium", None, [])
    assert res3["inputs"]["planted_seasons"] == {}

