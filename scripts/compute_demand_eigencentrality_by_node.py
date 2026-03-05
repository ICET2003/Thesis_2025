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

print(demand)

# Build Undirected Adjacency Matrices

## Map bus numbers to index
bus_index = {bus: i for i, bus in enumerate(all_buses)}

print(bus_index)

## Build edge lists
rows = []
cols = []

for _, row in candidates.iterrows():
    i = bus_index[row["from_bus_number"]]
    j = bus_index[row["to_bus_number"]]
    
    rows.extend([i, j])   # symmetric
    cols.extend([j, i])

# Construct adjacency
n = len(all_buses)
data = np.ones(len(rows)) # Rows of 1

A = csr_matrix((data, (rows, cols)), shape=(n, n))

# Constructed demand-weighted matrix

D = csr_matrix(np.diag(demand))
W = A @ D

print(W)

# Compute Eigenvector Centrality
vals, vecs = eigs(W, k=1, which='LR')  # largest real part

eigencentrality = np.abs(vecs[:, 0])
eigencentrality = eigencentrality / eigencentrality.sum()

print(eigencentrality)
print(len(eigencentrality))

eig = eigencentrality   # shape (nb,)
bus_ids = bus_index    # list of bus numbers

ec_df = pd.DataFrame(zip(bus_ids, eig), columns=["BusNum", "EigenvectorCentrality"])

print(ec_df.head())

# Save it to data/processed
## output path
output_path = ROOT / "data" / "processed" / "eigenvector_centrality.csv"

## save
ec_df.to_csv(output_path, index=False)
