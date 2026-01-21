from vector import get_image_embedding
import pandas as pd
import numpy as np
from display import show_images
import os
import joblib

# =========================
# CONFIG
# =========================
PCA_MODEL = "E:/models/pca.joblib"
CLUSTERED_CSV = "E:/projects/near-duplicate-detection/clip_embeddings_pca_clusters.csv"

TARGET_IMAGE = "E:/IMAGES/0/0/0/0000059611c7d079.jpg"
TOP_K = 5

IMAGE_BASE = "E:/IMAGES/0/0"

# =========================
# LOAD PCA
# =========================
pca = joblib.load(PCA_MODEL)

# =========================
# TARGET EMBEDDING
# =========================
target = get_image_embedding(TARGET_IMAGE)
target = np.asarray(target, dtype=np.float32)

t_norm = np.linalg.norm(target)
if t_norm == 0:
    raise ValueError("Target embedding norm is ZERO")

target /= t_norm

# PCA projection
target_pca = pca.transform(target.reshape(1, -1))[0]

# =========================
# LOAD PCA + CLUSTER DATASET
# =========================
df = pd.read_csv(CLUSTERED_CSV)

pca_cols = [c for c in df.columns if c.startswith("pca_")]
X_pca = df[pca_cols].to_numpy(dtype=np.float32)

# =========================
# FIND NEAREST CLUSTER
# =========================
# cosine similarity in PCA space
scores_all = X_pca @ target_pca
best_idx = np.argmax(scores_all)

target_cluster = df.iloc[best_idx]["cluster_id"]
print("Assigned cluster:", target_cluster)

# =========================
# FILTER SEARCH SPACE
# =========================
if target_cluster != -1:
    mask = df["cluster_id"] == target_cluster
    df_search = df[mask].reset_index(drop=True)
    X_search = X_pca[mask]
    print(f"Searching inside cluster ({len(df_search)} images)")
else:
    df_search = df
    X_search = X_pca
    print("Target is noise → searching entire dataset")

# =========================
# COSINE SIMILARITY (FINAL)
# =========================
scores = X_search @ target_pca

top_idx = np.argsort(scores)[-TOP_K:][::-1]
results = df_search.iloc[top_idx].copy()
results["similarity"] = scores[top_idx]

print("\nTop similar images:")
print("=" * 100)
print(results[["folder", "file", "similarity", "cluster_id"]])


#------------------FOR DISPLAYING IMAGES USING MATPLOTLIb-----------------------


image_paths = []
similarities = []

for _, row in results.iterrows():
    img_path = os.path.join(
        IMAGE_BASE,
        str(row["folder"]),
        str(row["file"])
    )

    if os.path.exists(img_path):
        image_paths.append(img_path)
        similarities.append(float(row["similarity"]))
    else:
        print("Missing file:", img_path)

show_images(image_paths, similarities, cols=3)
