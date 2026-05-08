# Download package
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]

candidates = pd.read_csv(ROOT / "Dataset" /  "Texas2k_series2025" / "Expansion_Planning_Problem_Data" / "Candidates.csv")
bus_region = pd.read_csv(ROOT/ "data/processed"/ "bus_level.csv")
regression_dataset_1 = pd.read_csv(ROOT/ "data/processed"/ "regression_dataset_1.csv")
# Sum the line capacity of each region

## 1. Put a candidate region and connect
 # from-bus region
candidates = candidates.merge(
    bus_region[["BusNum", "region"]],
    left_on="from_bus_number",
    right_on="BusNum",
    how="left"
).rename(columns={"region": "from_region"})

candidates = candidates.drop(columns=["BusNum"])

 # to-bus region
candidates = candidates.merge(
    bus_region[["BusNum", "region"]],
    left_on="to_bus_number",
    right_on="BusNum",
    how="left"
).rename(columns={"region": "to_region"})

candidates = candidates.drop(columns=["BusNum"])

print(candidates.head())

## 2. Calculate the sum of capacity
candidates["capacity"] = candidates["s1"]
 # For the across-region line
candidates["capacity_half"] = candidates["capacity"]/2

## 3. Create regional capacity distributions

from_side = candidates[["from_region", "capacity_half"]].copy()
from_side.columns = ["region", "capacity"]

to_side = candidates[["to_region", "capacity_half"]].copy()
to_side.columns = ["region", "capacity"]

capacity_by_region = pd.concat(
    [from_side, to_side],
    ignore_index=True
) 

print(capacity_by_region.head())

## 4. Aggregate capacoty by region

capacity_by_region = (capacity_by_region
                      .groupby("region")['capacity']
                      .sum()
                      .reset_index())

print(capacity_by_region)

# Intergrate the moving demand in by merging this into regression dataset
regression_dataset_1 = regression_dataset_1.merge(
    capacity_by_region,
    on="region",
    how="left"
)

regression_dataset_1["capacity_tightness"] = regression_dataset_1["demand_sum_x"]/ regression_dataset_1["capacity"]

print(regression_dataset_1.head())

# Save File
regression_dataset_1.to_csv(ROOT/ "data/processed"/ "regression_dataset_2.csv")