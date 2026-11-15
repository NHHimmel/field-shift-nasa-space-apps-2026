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
