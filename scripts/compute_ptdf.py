from pathlib import Path
import pandas as pd
import numpy as np


# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------
# Load branch dataset
# ------------------------------------------------------------

branch = pd.read_csv(
    ROOT /
    "Dataset" /
    "Texas2k_series2025" /
    "Expansion_Planning_Problem_Data" /
    "Candidates.csv"
)

print("Branch shape:", branch.shape)


# ------------------------------------------------------------
# Build bus list from network topology
# ------------------------------------------------------------

bus_ids = sorted(
    set(branch["from_bus_number"]) |
    set(branch["to_bus_number"])
)

print("Total buses:", len(bus_ids))

bus_lookup = {bus: i for i, bus in enumerate(bus_ids)}

nb = len(bus_ids)
nl = branch.shape[0]


# ------------------------------------------------------------
# Build branch matrix
# ------------------------------------------------------------

branch_matrix = []

for _, row in branch.iterrows():

    f = bus_lookup[row["from_bus_number"]]
    t = bus_lookup[row["to_bus_number"]]
    x = row["x"]

    branch_matrix.append([f, t, x])

branch_matrix = np.array(branch_matrix)


# ------------------------------------------------------------
# Build Bbus matrix (bus susceptance matrix)
# ------------------------------------------------------------

Bbus = np.zeros((nb, nb))

for row in branch_matrix:

    f = int(row[0])
    t = int(row[1])
    x = row[2]

    b = 1 / x

    Bbus[f,f] += b
    Bbus[t,t] += b
    Bbus[f,t] -= b
    Bbus[t,f] -= b


print("Bbus shape:", Bbus.shape)


# ------------------------------------------------------------
# Choose slack bus
# ------------------------------------------------------------

slack = 0


# ------------------------------------------------------------
# Reduce Bbus by removing slack bus
# ------------------------------------------------------------

Bred = np.delete(Bbus, slack, axis=0)
Bred = np.delete(Bred, slack, axis=1)

print("Reduced Bbus shape:", Bred.shape)


# ------------------------------------------------------------
# Invert reduced matrix
# ------------------------------------------------------------

Bred_inv = np.linalg.inv(Bred)


# ------------------------------------------------------------
# Build Bf (branch-flow incidence matrix)
# ------------------------------------------------------------

Bf = np.zeros((nl, nb))

for k, row in enumerate(branch_matrix):

    f = int(row[0])
    t = int(row[1])
    x = row[2]

    b = 1 / x

    Bf[k,f] = b
    Bf[k,t] = -b


print("Bf shape:", Bf.shape)


# ------------------------------------------------------------
# Remove slack column from Bf
# ------------------------------------------------------------

Bf_red = np.delete(Bf, slack, axis=1)


# ------------------------------------------------------------
# Compute PTDF
# ------------------------------------------------------------

PTDF = Bf_red @ Bred_inv

print("PTDF reduced shape:", PTDF.shape)


# ------------------------------------------------------------
# Restore slack column
# ------------------------------------------------------------

PTDF_full = np.zeros((nl, nb))

PTDF_full[:,1:] = PTDF


print("Full PTDF shape:", PTDF_full.shape)


# ------------------------------------------------------------
# Save PTDF matrix
# ------------------------------------------------------------

ptdf_df = pd.DataFrame(PTDF_full)

ptdf_path = ROOT / "data" / "processed" / "ptdf_matrix.csv"

ptdf_df.to_csv(ptdf_path, index=False)

print("Saved PTDF matrix:", ptdf_path)


# ------------------------------------------------------------
# Compute PTDF exposure per bus
# ------------------------------------------------------------

ptdf_exposure = np.mean(np.abs(PTDF_full), axis=0)

bus_df = pd.DataFrame({
    "BusNum": bus_ids,
    "PTDF_exposure": ptdf_exposure
})

exposure_path = ROOT / "data" / "processed" / "ptdf_exposure.csv"

bus_df.to_csv(exposure_path, index=False)

print("Saved PTDF exposure:", exposure_path)

print(bus_df.head())