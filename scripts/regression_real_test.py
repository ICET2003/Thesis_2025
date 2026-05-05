from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm


# --------------------------------------------------
# Project root
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------
# Load regressors and regressives
# --------------------------------------------------

ec = pd.read_csv(ROOT / "data/processed/eigenvector_centrality.csv")
bc = pd.read_csv(ROOT / "data/processed/betweenness_centrality.csv")
ptdf = pd.read_csv(ROOT / "data/processed/ptdf_exposure.csv")
congestion = pd.read_csv(ROOT/ "data/processed/congestion_hourly.csv")

print("Datasets loaded")


# --------------------------------------------------
# Merge datasets
# --------------------------------------------------

bus = ec.merge(bc, on="BusNum").merge(ptdf, on="BusNum")

print("Merged dataset size:", len(bus))


# --------------------------------------------------
# Rename columns (safer)
# --------------------------------------------------

bus = bus.rename(columns={
    "EigenvectorCentrality": "EC",
    "BC_demand": "BC",
    "PTDF_exposure": "PTDF"
})

#bus["BC"] = 100 * bus["BC"]

# -----------------------------
# Modify the bus 
# -----------------------------

# 1. Create a new column called region_code
bus['region_code'] = bus['BusNum'].astype("str").str[0].astype("int")

# 2. Create a region name_column
region_map = { # For congestion
    1: "WEST",
    2: "NORTH",
    3: "WEST",
    4: "SOUTH",
    5: "NORTH",
    6: "SOUTH",
    7: "COAST",
    8: "EAST"
}

region_map_8 = { # For weather
    1: "FWEST",
    2: "NORTH",
    3: "WEST",
    4: "SOUTH",
    5: "NCENT",
    6: "SCENT",
    7: "COAST",
    8: "EAST"
}


bus['region'] = bus['region_code'].map(region_map)
bus['region_8'] = bus['region_code'].map(region_map_8)

print(bus.head())

bus.to_csv(ROOT/ "data/processed"/ "bus_level.csv")

# --------------------------------------------------
# Get y data
# --------------------------------------------------
#y = congestion

#model = sm.OLS(y, X).fit()

#print(model.summary())