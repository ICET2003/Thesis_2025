from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
ROOT = Path(__file__).resolve().parents[1]
hourly_weighted = pd.read_csv(ROOT/ "data/processed"/ "hourly_weighted_ec_region5.csv")

# Mutate data
hourly_weighted["demand_sum_updated"] = hourly_weighted["demand_sum"]/12000

print(hourly_weighted)