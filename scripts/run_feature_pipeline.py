"""Fetch and inspect one farm's NASA POWER feature table."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from field_shift.nasa_data import FarmLocation, NASAPowerClient


def main() -> None:
    farm = FarmLocation(
        farm_id="rajshahi_plot_12",
        latitude=24.3745,
        longitude=88.6042,
        buffer_meters=250.0,
        irrigation_access=True,
        farmer_priority="water_conservation",
    )

    client = NASAPowerClient(cache_dir=PROJECT_ROOT / "data" / "cache")
    daily_df = client.fetch_daily_agro(
        location=farm,
        start_date="2024-01-01",
        end_date="2024-12-31",
        use_cache=True,
    )

    preview_columns = [
        "T2M",
        "RH2M",
        "ALLSKY_SFC_SW_DWN",
        "PRECTOTCORR",
        "GWETROOT",
        "vpd_kpa",
    ]
    print(daily_df[preview_columns].head())
    print("Rows:", len(daily_df))
    print("Mean climate features:")
    print(daily_df[preview_columns].mean(numeric_only=True))


if __name__ == "__main__":
    main()
