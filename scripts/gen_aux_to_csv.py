from pathlib import Path
import pandas as pd
import re
import shlex

# ---------------------------
# Paths
# ---------------------------
ROOT = Path(__file__).resolve().parents[1]

AUX = ROOT / "Dataset" / "Texas2k_series2025" / "Expansion_Planning_Problem_Data" / "Texas2k_2016_with_2025_subs_gen_load.aux"

out_dir = ROOT / "data" / "processed"
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Read AUX file
# ---------------------------
text = AUX.read_text(errors="ignore")

# ---------------------------
# Find generator block
# ---------------------------
m = re.search(
    r"DATA\s*\(\s*Gen\s*,\s*\[(.*?)\]\s*\)\s*\{(.*?)\}",
    text,
    flags=re.S | re.I
)

if not m:
    raise RuntimeError("Could not find DATA (Gen, [...]) block in AUX.")

# ---------------------------
# Extract columns + rows
# ---------------------------
cols = [c.strip() for c in m.group(1).split(",")]
body = m.group(2)

rows = []
bad = 0

for line in body.splitlines():

    line = line.strip()

    if not line or line.startswith("//"):
        continue

    parts = shlex.split(line)

    if len(parts) != len(cols):
        bad += 1

        if len(parts) < len(cols):
            parts = parts + [""] * (len(cols) - len(parts))
        else:
            parts = parts[:len(cols)]

    rows.append(parts)

# ---------------------------
# Build dataframe
# ---------------------------
df = pd.DataFrame(rows, columns=cols)

# ---------------------------
# Convert numeric columns
# ---------------------------
num_cols = ["BusNum", "GenMWMax", "GenMWMin", "GenFuelCost"]

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------------------------
# Keep only active generators
# ---------------------------
df = df[df["GenStatus"] == "Closed"]

# ---------------------------
# Keep only needed columns
# ---------------------------
gen_df = df[[
    "BusNum",
    "GenMWMax",
    "GenMWMin",
    "GenFuelCost",
    "GenFuelType"
]]

# ---------------------------
# Save file
# ---------------------------
out_path = out_dir / "generators.csv"

gen_df.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Generators extracted:", len(gen_df))
print("Fuel types:", gen_df["GenFuelType"].unique())
print(gen_df.head(10))