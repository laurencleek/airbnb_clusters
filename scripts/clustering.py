from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT_DIR / "outputs"

# Load lockbox listings
df = pd.read_parquet(OUTPUTS_DIR / "london_airbnb.parquet")
lock = df[df.self_checkin_flag == True].copy()

# Coordinates in radians for haversine
coords = np.radians(lock[["latitude","longitude"]].to_numpy())

# ~200-250 meters radius = 0.20-0.25 km 
# haversine DBSCAN uses radians → km / Earth's radius (6371 km)
radius_km = 0.25
eps = radius_km / 6371.0

db = DBSCAN(
    eps=eps,
    min_samples=25,         # cluster threshold
    metric='haversine',
    algorithm='ball_tree'   # efficient for geo
).fit(coords)

lock["cluster_id"] = db.labels_

# Save clustered results
lock.to_parquet(OUTPUTS_DIR / "london_airbnb_clusters_fixed.parquet", index=False)

print(lock.cluster_id.value_counts().head(15))
print("\nNumber of clusters found:", len([c for c in set(lock.cluster_id) if c != -1]))
