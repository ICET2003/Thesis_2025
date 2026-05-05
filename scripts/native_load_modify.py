# Load the package
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load the natve load file using read_xlsx
ROOT = Path(__file__).resolve().parents[1]
native_load_hourly = pd.read_excel(ROOT/ "Dataset"/ "Native_Load_2025.xlsx")

print(native_load_hourly)
# Pivot longer
native_load_long = native_load_hourly.melt(
    id_vars=["Hour Ending"],
    var_name="region",
    value_name="load_mw"
)

print(native_load_long.head())

# Save native_load
native_load_long.to_csv(ROOT/ "data/processed"/ "native_load_long.csv")