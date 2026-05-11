# First download all packages
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


# Download data
ROOT = Path(__file__).resolve().parents[1]

congestion_hourly = pd.read_csv(ROOT/ "data/processed"/ "congestion_regression.csv")
temp_hourly = pd.read_csv(ROOT/ "data/processed"/ "ercot_hourly_temp_region5.csv")
ec_hourly = pd.read_csv(ROOT/ "data/processed"/ "hourly_weighted_ec_region5.csv")
bc_hourly = pd.read_csv(ROOT/ "data/processed"/ "hourly_bc_static_x_demand_region5.csv")

# Region mapping, resolve inconsistency for temp and congestion
region_map = {
    "West": "WEST",
    "Houston": "COAST",
    "Panhandle": "EAST",
    "North": "NORTH",
    "South": "SOUTH"
}

congestion_hourly["region"] = congestion_hourly["load_region"].map(region_map)
temp_hourly["region"] = temp_hourly["region"].map(region_map)

# Rename the column of Hour Ending
congestion_hourly = congestion_hourly.rename(columns= {"Timestamp": "Hour Ending"})

# Sanity Check
print(congestion_hourly.head())
print(temp_hourly.head())

# Merge all dataset together with Hour Ending and region
regression_dataset_1 = (
    ec_hourly
    .merge(bc_hourly, how= "inner", on= ["Hour Ending", "region"])
    .merge(temp_hourly, how= "inner", on= ["Hour Ending", "region"])
    .merge(congestion_hourly, how= "inner", on = ["Hour Ending", "region"])
)

# Sanity check on number of columns and rows
print(regression_dataset_1.head())
print(regression_dataset_1.shape)

# Rename hour ending and change it to categorical by hour
regression_dataset_1 = regression_dataset_1.rename(columns= {"Hour Ending": "hour_ending"})
regression_dataset_1["hour_ending"] = pd.to_datetime(regression_dataset_1["hour_ending"])
regression_dataset_1["hour_ending_type"] = (
    regression_dataset_1["hour_ending"].dt.hour
)

# Save data
regression_dataset_1.to_csv(ROOT/ "data/processed"/ "regression_dataset_1.csv")

# Now regress (1)
model = smf.ols(
    formula="""
     congestion ~ EC_load_weighted 
                + BC_load_weighted
                + temperature
                + C(region) 
                + C(hour_ending_type) 
    """,
    data= regression_dataset_1
   ).fit(cov_type= "HC3") # Heteroskedascity robust SE

#print(model.summary())

# Now regress (2)
model_2 = smf.ols(
    formula="""
     congestion ~ EC_load_weighted 
                + temperature
                + C(region) 
                + C(hour_ending_type) 
    """,
    data= regression_dataset_1
   ).fit(cov_type= "HC3") # Heteroskedascity robust SE

#print(model_2.summary())


# Now regress (3)
model_3 = smf.ols(
    formula="""
     congestion ~ temperature
                + C(region) 
                + C(hour_ending_type) 
    """,
    data= regression_dataset_1
   ).fit(cov_type= "HC3") # Heteroskedascity robust SE

#print(model_3.summary())
# --------------------------------------------------------------------
# load data
regression_dataset_2 = pd.read_csv(ROOT/ "data/processed"/ "regression_dataset_2.csv")


# Now regress (4)
model_4 = smf.ols(
    formula="""
     congestion ~ capacity_tightness
                + EC_load_weighted 
                + BC_load_weighted
                + temperature
                + C(region) 
                + C(hour_ending_type) 
    """,
    data= regression_dataset_2
   ).fit(cov_type= "HC3") # Heteroskedascity robust SE

#print(model_4.summary())

# Now regress (5)
model_5 = smf.ols(
    formula="""
     congestion ~ capacity_tightness
                + temperature
                + C(region) 
                + C(hour_ending_type) 
    """,
    data= regression_dataset_2
   ).fit(cov_type= "HC3") # Heteroskedascity robust SE

#print(model_5.summary())

# Now regress (6)
regression_dataset_2["log_congestion"] = np.log1p(regression_dataset_2["congestion"])

model_6 = smf.ols(
    formula="""
     log_congestion ~ capacity_tightness
                + EC_load_weighted 
                + BC_load_weighted
                + temperature
                + C(region) 
                + C(hour_ending_type) 
    """,
    data= regression_dataset_2
   ).fit(cov_type= "HC3") # Heteroskedascity robust SE

#print(model_6.summary())

# -------------------------------------------------------- (Congestion Abs)
regression_dataset_2["abs_congestion"] = np.abs(regression_dataset_2["congestion"])
regression_dataset_2["abs_log_congestion"] = np.log1p(regression_dataset_2["abs_congestion"])

model_7 = smf.ols(
    formula="""
     abs_congestion ~ temperature
                + C(region) 
                + C(hour_ending_type) 
    """,
    data= regression_dataset_2
   ).fit(cov_type= "HC3")

print(model_7.summary())

model_8 = smf.ols(
    formula="""
     abs_log_congestion ~ temperature
                + C(region) 
                + C(hour_ending_type) 
    """,
    data= regression_dataset_2
   ).fit(cov_type= "HC3") # Heteroskedascity robust SE

print(model_8.summary())

model_9 = smf.ols(
    formula="""
     abs_congestion ~ capacity_tightness
                + temperature
                + C(region) 
                + C(hour_ending_type) 
    """,
    data= regression_dataset_2
   ).fit(cov_type= "HC3")

print(model_9.summary())

model_10 = smf.ols(
    formula="""
     abs_log_congestion ~ capacity_tightness
                + temperature
                + C(region) 
                + C(hour_ending_type) 
    """,
    data= regression_dataset_2
   ).fit(cov_type= "HC3") # Heteroskedascity robust SE

print(model_10.summary())

model_11 = smf.ols(
    formula="""
     abs_congestion ~ capacity_tightness
                + EC_load_weighted 
                + BC_load_weighted
                + temperature
                + C(region) 
                + C(hour_ending_type) 
    """,
    data= regression_dataset_2
   ).fit(cov_type= "HC3") # Heteroskedascity robust SE

print(model_11.summary())

model_12 = smf.ols(
    formula="""
     abs_log_congestion ~ capacity_tightness
                + EC_load_weighted 
                + BC_load_weighted
                + temperature
                + C(region) 
                + C(hour_ending_type) 
    """,
    data= regression_dataset_2
   ).fit(cov_type= "HC3") # Heteroskedascity robust SE

print(model_12.summary())

# --------------------------------------------- (For printing)

from statsmodels.iolib.summary2 import summary_col

# ==========================================
# MODEL 1: Baseline FE
# ==========================================

model1 = smf.ols(
    formula="""
    abs_congestion ~ temperature
                    + C(region)
                    + C(hour_ending_type)
    """,
    data=regression_dataset_2
).fit(cov_type="HC3")

# ==========================================
# MODEL 2: + Capacity Tightness
# ==========================================

model2 = smf.ols(
    formula="""
    abs_congestion ~ capacity_tightness
                    + temperature
                    + C(region)
                    + C(hour_ending_type)
    """,
    data=regression_dataset_2
).fit(cov_type="HC3")

# ==========================================
# MODEL 3: Full Network Model
# ==========================================

model3 = smf.ols(
    formula="""
    abs_congestion ~ EC_load_weighted
                    + BC_load_weighted
                    + capacity_tightness
                    + temperature
                    + C(region)
                    + C(hour_ending_type)
    """,
    data=regression_dataset_2
).fit(cov_type="HC3")

# ==========================================
# MODEL 4: Log Absolute Congestion
# ==========================================

model4 = smf.ols(
    formula="""
    abs_log_congestion ~ EC_load_weighted
                        + BC_load_weighted
                        + capacity_tightness
                        + temperature
                        + C(region)
                        + C(hour_ending_type)
    """,
    data=regression_dataset_2
).fit(cov_type="HC3")

# ==========================================
# CREATE TABLE
# ==========================================

results_table = summary_col(
    [model1, model2, model3, model4],
    stars=True,
    float_format='%0.4f',
    model_names=[
        'Baseline',
        '+ Capacity',
        '+ Network',
        'Log Abs'
    ],
    info_dict={
        'Observations': lambda x: f"{int(x.nobs)}",
        'R-squared': lambda x: f"{x.rsquared:.3f}"
    }
)

# ==========================================
# PRINT TABLE
# ==========================================

print(results_table)

# ==========================================
# EXPORT TO LATEX
# ==========================================

with open("../Results/regression_results.tex", "w") as f:
    f.write(results_table.as_latex())

with open("../Results/full_regression_table.html", "w") as f:
    f.write(results_table.as_html())

print("LaTeX table exported successfully.")

