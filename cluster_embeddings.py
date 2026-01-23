import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

PCA_CSV = "E:/projects/near-duplicate-detection/clip_embeddings_pca.csv"
OUT_CSV = "data/clip_embeddings_pca_clusters.csv"

EPS = 0.25
MIN_SAMPLES = 2


df = pd.read_csv(PCA_CSV)

pca_cols = [c for c in df.columns if c.startswith("pca_")]
X_pca = df[pca_cols].to_numpy(dtype=np.float32)

# DBSCAN CLUSTERING

db = DBSCAN(
    eps=EPS,
    min_samples=MIN_SAMPLES,
    metric="cosine"
)

labels = db.fit_predict(X_pca)
df["cluster_id"] = labels


df.to_csv(OUT_CSV, index=False)

print("Clustering complete")
print("Total clusters:", len(set(labels)) - (1 if -1 in labels else 0))
print("Noise images:", (labels == -1).sum())
