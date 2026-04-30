from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Download dataset from Congestion_lmp folder
ROOT = Path(__file__).resolve().parents[1] # Avoid conflicts
congestion_q_1 = pd.read_csv(ROOT/ "Dataset" / "Congestion_lmp"/ "grid_status_data_1.csv")
congestion_q_2 = pd.read_csv(ROOT/ "Dataset" / "Congestion_lmp"/ "grid_status_data_2.csv")
congestion_q_3 = pd.read_csv(ROOT/ "Dataset" / "Congestion_lmp"/ "grid_status_data_3.csv")
congestion_q_4 = pd.read_csv(ROOT/ "Dataset" / "Congestion_lmp"/ "grid_status_data_4.csv")

# Merge all dataset into one
congestion_2025 = pd.concat([congestion_q_1, 
                             congestion_q_2, 
                             congestion_q_3, 
                             congestion_q_4],
                             ignore_index= True)

# Sanity Check 
print(congestion_2025.head())
print(congestion_2025.shape)
print(congestion_2025.columns)

# Now we have to find hourly average (as now the timestamp is every 5 minutes)
congestion_2025["Timestamp"] = pd.to_datetime(congestion_2025["Timestamp"])

# Set timestamp as indexes
congestion_2025 = congestion_2025.set_index("Timestamp")

# Find hourly data
congestion_2025_hourly = congestion_2025.resample("H").mean(numeric_only=True)

print(congestion_2025_hourly)

# Save to CSV
out_dir = ROOT / "data" / "processed"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "congestion_hourly.csv"
congestion_2025_hourly.to_csv(out_path, index=False)

