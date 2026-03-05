from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm


# --------------------------------------------------
# Project root
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------
# Load regressors
# --------------------------------------------------

ec = pd.read_csv(ROOT / "data/processed/eigenvector_centrality.csv")
bc = pd.read_csv(ROOT / "data/processed/betweenness_centrality.csv")
ptdf = pd.read_csv(ROOT / "data/processed/ptdf_exposure.csv")

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

bus["BC"] = 100 * bus["BC"]

# --------------------------------------------------
# Generate structured synthetic LMP
# --------------------------------------------------

np.random.seed(42)

base_price = 25

bus["LMP"] = (
    base_price
    + 80 * bus["PTDF"]
    + 40 * bus["EC"]
    + 30 * bus["BC"]
    + np.random.normal(0, 3, len(bus))
)

print("\nSynthetic LMP generated")
print(bus["LMP"].describe())


# --------------------------------------------------
# Compute congestion
# --------------------------------------------------

system_price = bus["LMP"].mean()

bus["Congestion"] = bus["LMP"] - system_price
bus["Congestion_abs"] = bus["Congestion"].abs()


# --------------------------------------------------
# Save dataset
# --------------------------------------------------

output_path = ROOT / "data/processed/synthetic_lmp_dataset.csv"

bus.to_csv(output_path, index=False)

print("\nDataset saved to:")
print(output_path)


# --------------------------------------------------
# Run regression
# --------------------------------------------------

print("\nRunning regression...")

X = bus[["EC", "BC", "PTDF"]]

X = sm.add_constant(X)

y = bus["Congestion_abs"]

model = sm.OLS(y, X).fit()

print(model.summary())