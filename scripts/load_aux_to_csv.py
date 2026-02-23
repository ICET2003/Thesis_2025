from pathlib import Path
import pandas as pd
import re
import shlex

ROOT = Path(__file__).resolve().parents[1]
AUX = ROOT / "Dataset" / "Texas2k_series2025" / "Expansion_Planning_Problem_Data" / "Texas2k_2016_with_2025_subs_gen_load.aux"

text = AUX.read_text(errors="ignore")

m = re.search(
    r"DATA\s*\(\s*Load\s*,\s*\[(.*?)\]\s*\)\s*\{(.*?)\}",
    text,
    flags=re.S | re.I
)
if not m:
    raise RuntimeError("Could not find DATA (Load, [...]) block in AUX.")

cols = [c.strip() for c in m.group(1).split(",")]
body = m.group(2)

rows = []
bad = 0
for line in body.splitlines():
    line = line.strip()
    if not line or line.startswith("//"):
        continue

    parts = shlex.split(line)  #quote-aware split

    # Some lines may still have mismatch; make them fit the header
    if len(parts) != len(cols):
        bad += 1
        if len(parts) < len(cols):
            parts = parts + [""] * (len(cols) - len(parts))
        else:
            parts = parts[:len(cols)]

    rows.append(parts)

df = pd.DataFrame(rows, columns=cols)

df["BusNum"] = pd.to_numeric(df["BusNum"], errors="coerce")
df["LoadMW"] = pd.to_numeric(df["LoadMW"], errors="coerce").fillna(0.0)

bus_load = df.groupby("BusNum", as_index=False)["LoadMW"].sum()
bus_load.rename(columns={"LoadMW": "PD_base"}, inplace=True)

out_dir = ROOT / "data" / "processed"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "base_bus_load.csv"
bus_load.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Total system load (MW):", bus_load["PD_base"].sum())
print("Rows with token mismatch handled:", bad)
print(bus_load.head(10))