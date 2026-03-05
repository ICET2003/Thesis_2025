import pandas as pd
from pathlib import Path

# project root = Thesis_2025/
ROOT = Path(__file__).resolve().parents[1]
gen_load_path = ROOT/ "data/processed" / "generators.csv"

print("Reading from:")
print(gen_load_path, "\n")

gen_df = pd.read_csv(gen_load_path)

# remove storage units
gen_df = gen_df[~gen_df["GenFuelType"].str.contains("Energy Storage", na=False)]

# function to assign Birchfield-style marginal costs
def assign_marginal_cost(fuel):

    fuel = str(fuel)

    if "Wind" in fuel:
        return 0

    if "Solar" in fuel:
        return 0

    if "Hydro" in fuel:
        return 5

    if "Nuclear" in fuel:
        return 10

    if "Coal" in fuel:
        return 20

    if "Natural Gas" in fuel or "NG" in fuel:
        return 35

    return 50  # fallback for oil / other fuels


# apply cost assignment
gen_df["marginal_cost"] = gen_df["GenFuelType"].apply(assign_marginal_cost)

# convert numeric columns
gen_df["GenMWMax"] = pd.to_numeric(gen_df["GenMWMax"], errors="coerce")
gen_df["GenMWMin"] = pd.to_numeric(gen_df["GenMWMin"], errors="coerce")

# keep only needed columns
gen_df_clean = gen_df[[
    "BusNum",
    "GenMWMax",
    "GenMWMin",
    "marginal_cost"
]]

# save result
gen_df_clean.to_csv("../data/processed/generators_clean.csv", index=False)

print("Generators ready for DC-OPF:", len(gen_df_clean))
print(gen_df_clean.head())