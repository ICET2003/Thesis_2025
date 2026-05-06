from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

weather_path = ROOT / "Dataset" / "ercot_hourly_temp.csv"

weather = pd.read_csv(weather_path)

# Use local interval start as the clean hourly timestamp
weather["Hour Ending"] = pd.to_datetime(
    weather["interval_start_local"],
    errors="coerce",
    utc=True
).dt.tz_convert(None)

# Drop interval/publish timestamp columns
weather = weather.drop(
    columns=[
        "interval_start_local",
        "interval_start_utc",
        "interval_end_local",
        "interval_end_utc",
        "publish_time_local",
        "publish_time_utc"
    ],
    errors="ignore"
)

# Put timestamp first
weather = weather[["Hour Ending"] + [c for c in weather.columns if c != "Hour Ending"]]

# Optional: average duplicates if several rows share the same hour
weather_hourly = (
    weather
    .groupby("Hour Ending", as_index=False)
    .mean(numeric_only=True)
)

out_path = ROOT / "data" / "processed" / "ercot_hourly_temp_clean.csv"
weather_hourly.to_csv(out_path, index=False)

print(weather_hourly.head())
print(weather_hourly.shape)
print("Saved to:", out_path)

# Next merge into 5 regions

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


weather_path = ROOT / "data/processed" / "ercot_hourly_temp_clean.csv"
weather = pd.read_csv(weather_path)

weather["Hour Ending"] = pd.to_datetime(weather["Hour Ending"])

# Wide → long
weather_long = weather.melt(
    id_vars="Hour Ending",
    var_name="region_8",
    value_name="temperature"
)

weather_long["region_8"] = weather_long["region_8"].astype(str).str.upper().str.strip()

# 8 regions → 5 regions
region_map = {
    "WEST": "West",
    "FAR_WEST": "West",
    "FWEST": "West",

    "NORTH": "North",
    "NORTH_CENTRAL": "North",
    "NCENT": "North",
    "SCENT": "South",
    "SOUTH_CENTRAL": "South",

    "SOUTHERN": "South",

    "COAST": "Houston",

    "EAST": "Panhandle"
}

weather_long["region"] = weather_long["region_8"].map(region_map)

# Drop unmapped columns if any
weather_long = weather_long.dropna(subset=["region"])

# Collapse 8 → 5 by averaging temperature
weather_region5 = (
    weather_long
    .groupby(["Hour Ending", "region"], as_index=False)["temperature"]
    .mean()
)

out_path = ROOT / "data" / "processed" / "ercot_hourly_temp_region5.csv"
weather_region5.to_csv(out_path, index=False)

print(weather_region5.head())
print(weather_region5["region"].unique())
print(weather_region5.shape)
print("Saved to:", out_path)