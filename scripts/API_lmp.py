import requests
import pandas as pd

API_KEY = "f477488059bc4c12b88763f3ff212aaf"

url = "https://api.gridstatus.io/v1/datasets/ercot_lmp"

params = {
    "api_key": API_KEY,
    "start": "2024-01-01",
    "end": "2024-01-02",
    "limit": 10000
}

res = requests.get(url, params=params)
data = res.json()

df = pd.DataFrame(data["data"])
print(df.head())