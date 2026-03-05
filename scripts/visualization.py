from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------
# Project root
# ------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "processed"
FIGS = ROOT / "results" / "figures"

FIGS.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------
# Load datasets
# ------------------------------------------------

ec = pd.read_csv(DATA / "eigenvector_centrality.csv")
bc = pd.read_csv(DATA / "betweenness_centrality.csv")
ptdf = pd.read_csv(DATA / "ptdf_exposure.csv")


# ------------------------------------------------
# Rename columns (if needed)
# ------------------------------------------------

ec = ec.rename(columns={"EigenvectorCentrality": "EC"})
bc = bc.rename(columns={"BC_demand": "BC"})
ptdf = ptdf.rename(columns={"PTDF_exposure": "PTDF"})


# ------------------------------------------------
# Plot distributions
# ------------------------------------------------

plt.figure()
sns.histplot(ec["EC"], bins=40)
plt.title("Eigenvector Centrality Distribution")
plt.savefig(FIGS / "ec_distribution.png")
plt.close()


plt.figure()
sns.histplot(bc["BC"], bins=40)
plt.title("Betweenness Centrality Distribution")
plt.savefig(FIGS / "bc_distribution.png")
plt.close()


plt.figure()
sns.histplot(ptdf["PTDF"], bins=40)
plt.title("PTDF Exposure Distribution")
plt.savefig(FIGS / "ptdf_distribution.png")
plt.close()


# ------------------------------------------------
# Top 20 buses by centrality
# ------------------------------------------------

top_ec = ec.nlargest(20, "EC")

plt.figure(figsize=(8,5))
sns.barplot(x="EC", y="BusNum", data=top_ec)
plt.title("Top 20 Buses by Eigenvector Centrality")
plt.savefig(FIGS / "top_ec_buses.png")
plt.close()


top_bc = bc.nlargest(20, "BC")

plt.figure(figsize=(8,5))
sns.barplot(x="BC", y="BusNum", data=top_bc)
plt.title("Top 20 Buses by Betweenness Centrality")
plt.savefig(FIGS / "top_bc_buses.png")
plt.close()


# ------------------------------------------------
# Merge datasets for scatter plots
# ------------------------------------------------

df = ec.merge(bc, on="BusNum").merge(ptdf, on="BusNum")


plt.figure()
sns.scatterplot(x="EC", y="PTDF", data=df)
plt.title("EC vs PTDF Exposure")
plt.savefig(FIGS / "ec_vs_ptdf.png")
plt.close()


plt.figure()
sns.scatterplot(x="BC", y="PTDF", data=df)
plt.title("BC vs PTDF Exposure")
plt.savefig(FIGS / "bc_vs_ptdf.png")
plt.close()

plt.figure()
sns.scatterplot(x="BC", y="EC", data=df)
plt.title("BC vs  EC")
plt.savefig(FIGS / "bc_vs_ec.png")
plt.close()

print("All figures saved to:", FIGS)