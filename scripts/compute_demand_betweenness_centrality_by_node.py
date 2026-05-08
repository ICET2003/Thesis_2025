from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx 
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

# project root = Thesis_2025/
ROOT = Path(__file__).resolve().parents[1]

candidates_path = ROOT / "Dataset" /  "Texas2k_series2025" / "Expansion_Planning_Problem_Data" / "Candidates.csv"
bus_load_path = ROOT/ "data/processed" / "base_bus_load.csv"

print("Reading from:")
print(candidates_path, "\n")

candidates = pd.read_csv(candidates_path)
bus_load = pd.read_csv(bus_load_path)

# Sanity check
print(candidates.head(10))
print(bus_load.head(10))

# Build Master Bus List
from_buses = candidates["from_bus_number"]
to_buses   = candidates["to_bus_number"]

all_buses = pd.Index(from_buses).union(pd.Index(to_buses))
all_buses = np.sort(all_buses.unique())

# Sanity check
print(all_buses)

# Aggregate all demand
bus_demand = (
    bus_load
    .groupby("BusNum")["PD_base"]
    .sum()
)

demand = bus_demand.reindex(all_buses).fillna(0).values

# Compute Betweenness Centrality

G = nx.Graph()

G.add_nodes_from(all_buses)

for _, row in candidates.iterrows():
    G.add_edge(row["from_bus_number"], row["to_bus_number"])

# Define Origin-Distance weights
demand_dict = dict(zip(all_buses, demand))

# Compute through loops 
betweenness = {node: 0.0 for node in G.nodes()}

for s in G.nodes():
    for t in G.nodes():
        if s >= t:
            continue
        
        weight = demand_dict[s] * demand_dict[t]
        if weight == 0:
            continue
        
        if not nx.has_path(G, s, t):
            continue

        paths = list(nx.all_shortest_paths(G, s, t))
        num_paths = len(paths)
        
        for path in paths:
            for v in path[1:-1]:
                betweenness[v] += weight / num_paths

print(betweenness)

# Normalize 
## Get the sum of all weight
total_weight = sum(demand_dict[s] * demand_dict[t]
                   for s in G.nodes()
                   for t in G.nodes()
                   if s < t)

for v in betweenness:
    betweenness[v] /= total_weight

# Convert to dataframe
bc_df = pd.DataFrame({
    "BusNum": list(betweenness.keys()),
    "BC_demand": list(betweenness.values())
})

print(bc_df)
# Save to CSV
out_dir = ROOT / "data" / "processed"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "betweenness_centrality.csv"
bc_df.to_csv(out_path, index=False)
