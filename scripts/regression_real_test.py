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
region_map = {
    1: "Far West",
    2: "North",
    3: "West",
    4: "South",
    5: "North Central",
    6: "South Central",
    7: "Coast",
    8: "East"
}

bus['region'] = bus['region_code'].map(region_map)

print(bus.head())

# --------------------------------------------------
# Get y data
# --------------------------------------------------
#y = congestion

#model = sm.OLS(y, X).fit()

#print(model.summary())