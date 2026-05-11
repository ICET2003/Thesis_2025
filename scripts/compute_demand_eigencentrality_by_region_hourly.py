from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigs

# ============================================================
# 1. Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

candidates_path = ROOT / "Dataset" / "Texas2k_series2025" / "Expansion_Planning_Problem_Data" / "Candidates.csv"
bus_level_path = ROOT / "data" / "processed" / "bus_level.csv"
native_load_path = ROOT / "data" / "processed" / "native_load_long.csv"

bus_ec_output = ROOT / "data" / "processed" / "hourly_weighted_ec_bus.csv"
region_ec_output = ROOT / "data" / "processed" / "hourly_weighted_ec_region5.csv"

# ============================================================
# 2. Read data
# ============================================================

candidates = pd.read_csv(candidates_path)
bus_level = pd.read_csv(bus_level_path)
native_load = pd.read_csv(native_load_path)
# Fix the inconsistency of timestamp
def fix_hour_ending(ts):
    ts = str(ts).strip()
    parts = ts.split()   # handles multiple spaces automatically

    date = parts[0]
    time = parts[1]

    if time == "24:00":
        new_date = pd.to_datetime(date) + pd.Timedelta(days=1)
        return f"{new_date.strftime('%m/%d/%Y')} 00:00"

    return f"{date} {time}"

native_load["Hour Ending"] = native_load["Hour Ending"].apply(fix_hour_ending)

native_load["Hour Ending"] = pd.to_datetime(native_load["Hour Ending"])


native_load["Hour Ending"] = pd.to_datetime(native_load["Hour Ending"])

# Drop ERCOT because it is total system load, not a region
native_load = native_load[native_load["region"] != "ERCOT"].copy()

# ============================================================
# 3. Build bus list and adjacency matrix A
# ============================================================

all_buses = np.sort(
    pd.Index(candidates["from_bus_number"])
    .union(pd.Index(candidates["to_bus_number"]))
    .unique()
)

bus_index = {bus: i for i, bus in enumerate(all_buses)}

rows = []
cols = []

for _, row in candidates.iterrows():
    i = bus_index[row["from_bus_number"]]
    j = bus_index[row["to_bus_number"]]

    rows.extend([i, j])
    cols.extend([j, i])

n = len(all_buses)

A = csr_matrix(
    (np.ones(len(rows)), (rows, cols)),
    shape=(n, n)
)

# ============================================================
# 4. Bus to 8-region mapping
# ============================================================
# Change this if your 8-region column has a different name

print(bus_level.columns)

BUS_REGION_8_COL = "region_8"

bus_region_8 = (
    bus_level[["BusNum", BUS_REGION_8_COL]]
    .drop_duplicates("BusNum")
    .set_index("BusNum")[BUS_REGION_8_COL]
)



# ============================================================
# 5. Recalculate weighted EC for every hour
# ============================================================

results = []

# Bus counts
bus_counts_8 = (bus_region_8.
                reindex(all_buses).
                value_counts())

print(bus_counts_8)

for hour, df_hour in native_load.groupby("Hour Ending"):

    # 8-region hourly load
    region_load = df_hour.set_index("region")["load_mw"]

    # Assign each bus its hourly load based on its 8-region
    bus_counts_8_aligned = bus_counts_8.reindex(region_load.index)

    load_per_bus = (region_load / bus_counts_8_aligned).to_dict()

    demand_t = (
        bus_region_8
        .map(load_per_bus)
        .fillna(0)
        .to_numpy()
    )

    # Skip if no demand information
    if demand_t.sum() == 0:
        continue

    # W_t = A @ D_t
    D_t = diags(demand_t)
    W_t = A @ D_t # Sum of the capacity, may collapse it to 5 => Make the network easier to calculate

    # Eigenvector centrality
    vals, vecs = eigs(W_t, k=1, which="LR")

    ec_t = np.abs(np.real(vecs[:, 0]))

    # Normalize
    if ec_t.sum() != 0:
        ec_t = ec_t / ec_t.sum()

    hour_df = pd.DataFrame({
        "Hour Ending": hour,
        "BusNum": all_buses,
        "demand_t": demand_t,
        "EC_weighted": ec_t
    })

    results.append(hour_df)

ec_hourly_bus = pd.concat(results, ignore_index=True)

ec_hourly_bus.to_csv(bus_ec_output, index=False)

print("Saved bus-level hourly EC:")
print(bus_ec_output)
print(ec_hourly_bus.head())
print(ec_hourly_bus.shape)

# ============================================================
# 6. Collapse bus-level EC into 5 regions
# ============================================================
# This assumes bus_level["region"] is your 5-region label:
# West, North, South, Houston, Panhandle

bus_region_5 = (
    bus_level[["BusNum", "region"]]
    .drop_duplicates("BusNum")
)

ec_hourly_bus = ec_hourly_bus.merge(
    bus_region_5,
    on="BusNum",
    how="left"
)

# Load-weighted regional EC
ec_hourly_bus["EC_times_demand"] = (
    ec_hourly_bus["EC_weighted"] * ec_hourly_bus["demand_t"]
)

ec_hourly_region5 = (
    ec_hourly_bus
    .dropna(subset=["region"])
    .groupby(["Hour Ending", "region"], as_index=False)
    .agg(
        EC_mean=("EC_weighted", "mean"),
        EC_sum=("EC_weighted", "sum"),
        demand_sum=("demand_t", "sum"),
        EC_times_demand_sum=("EC_times_demand", "sum"),
        n_buses=("BusNum", "nunique")
    )
)

ec_hourly_region5["EC_load_weighted"] = (
    ec_hourly_region5["EC_times_demand_sum"] /
    ec_hourly_region5["demand_sum"]
)

ec_hourly_region5.to_csv(region_ec_output, index=False)

print("Saved 5-region hourly EC:")
print(region_ec_output)
print(ec_hourly_region5.head())
print(ec_hourly_region5.shape)



# Get a heatmap

import matplotlib.pyplot as plt

# convert sparse matrix to dense
A_dense = W_t.toarray()

plt.figure(figsize=(12,10))

plt.imshow(
    A_dense,
    cmap="viridis",   # try also: plasma, inferno, magma
    aspect="auto"
)

plt.colorbar(label="Connection Strength")

plt.title("Adjacency Matrix Heatmap")
plt.xlabel("Bus Index")
plt.ylabel("Bus Index")

plt.show()