from pathlib import Path
import pandas as pd
import numpy as np

from pypower.runopf import runopf
from pypower.ppoption import ppoption


# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

load = pd.read_csv(ROOT / "data" / "processed" / "base_bus_load.csv")
gen = pd.read_csv(ROOT / "data" / "processed" / "generators_clean.csv")

branch = pd.read_csv(
    ROOT / "Dataset" /
    "Texas2k_series2025" /
    "Expansion_Planning_Problem_Data" /
    "Candidates.csv"
)

print("Load shape:", load.shape)
print("Gen shape:", gen.shape)
print("Branch shape:", branch.shape)


# ------------------------------------------------------------
# Build COMPLETE bus list from network topology
# ------------------------------------------------------------

bus_ids = sorted(
    set(branch["from_bus_number"]) |
    set(branch["to_bus_number"]) |
    set(load["BusNum"]) |
    set(gen["BusNum"])
)

print("Total buses:", len(bus_ids))

bus_lookup = {bus: i+1 for i, bus in enumerate(bus_ids)}


# ------------------------------------------------------------
# BUS MATRIX
# MATPOWER format:
# [BUS_I TYPE PD QD GS BS AREA VM VA BASEKV ZONE VMAX VMIN]
# ------------------------------------------------------------

bus_rows = []

for b in bus_ids:

    demand = load.loc[load["BusNum"] == b, "PD_base"].sum()

    bus_rows.append([
        bus_lookup[b],   # BUS_I
        1,               # TYPE = PQ
        demand,          # PD
        0,               # QD
        0,               # GS
        0,               # BS
        1,               # AREA
        1.0,             # VM
        0.0,             # VA
        230,             # BASE_KV
        1,               # ZONE
        1.1,             # VMAX
        0.9              # VMIN
    ])

bus = np.array(bus_rows)


# ------------------------------------------------------------
# GENERATOR MATRIX
# [BUS PG QG QMAX QMIN VG MBASE STATUS PMAX PMIN]
# ------------------------------------------------------------

gen_rows = []

for _, row in gen.iterrows():

    if row["BusNum"] not in bus_lookup:
        continue

    gen_rows.append([
        bus_lookup[row["BusNum"]],
        0,      # PG
        0,      # QG
        0,      # QMAX
        0,      # QMIN
        1.0,    # VG
        100,    # MBASE
        1,      # STATUS
        row["GenMWMax"],
        row["GenMWMin"]
    ])

gen_matrix = np.array(gen_rows)

print("Generator matrix:", gen_matrix.shape)


# ------------------------------------------------------------
# SET SLACK BUS
# ------------------------------------------------------------

slack_bus = gen.iloc[0]["BusNum"]
bus[bus[:, 0] == bus_lookup[slack_bus], 1] = 3

print("Slack bus:", slack_bus)


# ------------------------------------------------------------
# GENERATOR COST
# ------------------------------------------------------------

gencost_rows = []

for _, row in gen.iterrows():

    gencost_rows.append([
        2,                 # polynomial
        0,
        0,
        2,                 # linear cost
        row["marginal_cost"],
        0
    ])

gencost = np.array(gencost_rows)


# ------------------------------------------------------------
# BRANCH MATRIX
# MATPOWER format:
# [F_BUS T_BUS R X B RATE_A RATE_B RATE_C
#  TAP SHIFT STATUS ANGMIN ANGMAX]
# ------------------------------------------------------------

branch_rows = []

for _, row in branch.iterrows():

    if row["from_bus_number"] not in bus_lookup:
        continue
    if row["to_bus_number"] not in bus_lookup:
        continue

    branch_rows.append([
        bus_lookup[row["from_bus_number"]],
        bus_lookup[row["to_bus_number"]],
        0,              # R (ignored in DC)
        row["x"],       # reactance
        0,              # B
        row["s1"],      # rateA
        row["s2"],      # rateB
        row["s3"],      # rateC
        0,              # tap
        0,              # shift
        1,              # status
        -360,
        360
    ])

branch_matrix = np.array(branch_rows)

print("Branch matrix:", branch_matrix.shape)


# ------------------------------------------------------------
# SYSTEM BALANCE CHECK
# ------------------------------------------------------------

total_load = bus[:,2].sum()
total_gen = gen_matrix[:,8].sum()

print("Total load:", total_load)
print("Total generation capacity:", total_gen)


# ------------------------------------------------------------
# BUILD PYPOWER CASE
# ------------------------------------------------------------

ppc = {
    "version": "2",
    "baseMVA": 100,
    "bus": bus,
    "gen": gen_matrix,
    "branch": branch_matrix,
    "gencost": gencost
}


# ------------------------------------------------------------
# RUN DC OPF
# ------------------------------------------------------------

ppopt = ppoption(PF_DC=True, VERBOSE=1)

results = runopf(ppc, ppopt)

if not results["success"]:
    print("OPF did not converge")
    exit()

print("DC OPF solved successfully")


# ------------------------------------------------------------
# EXTRACT LMP
# ------------------------------------------------------------

lmp = results["bus"][:,13]

lmp_df = pd.DataFrame({
    "BusNum": bus_ids,
    "LMP": lmp
})


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

out_path = ROOT / "data" / "processed" / "lmp_results.csv"

lmp_df.to_csv(out_path, index=False)

print("Saved LMP results:", out_path)
print(lmp_df.head())