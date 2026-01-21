import pandas as pd
import numpy as np
import joblib
import os

# =========================
# CONFIG
# =========================
RAW_CSV = "E:/projects/near-duplicate-detection/clip_embeddings.csv"          # input (512-D)
PCA_MODEL = "E:/models/pca.joblib"               # trained PCA
OUT_CSV = "data/clip_embeddings_pca.csv"      # output (128-D)

# =========================
# LOAD DATA
# =========================
print("Loading raw embeddings CSV...")
df = pd.read_csv(RAW_CSV)

embed_cols = [c for c in df.columns if c.startswith("dim_")]
if not embed_cols:
    raise ValueError("No embedding columns found (dim_*)")

X = df[embed_cols].to_numpy(dtype=np.float32)

# =========================
# NORMALIZE (CRITICAL)
# =========================
norms = np.linalg.norm(X, axis=1, keepdims=True)
mask = norms.squeeze() > 0

df = df.iloc[mask].reset_index(drop=True)
X = X[mask]
X /= norms[mask]

# =========================
# LOAD PCA
# =========================
print("Loading PCA model...")
pca = joblib.load(PCA_MODEL)

# =========================
# APPLY PCA
# =========================
print("Applying PCA...")
X_pca = pca.transform(X)

# =========================
# BUILD OUTPUT CSV
# =========================
df_pca = df[["folder", "file"]].copy()

for i in range(X_pca.shape[1]):
    df_pca[f"pca_{i}"] = X_pca[:, i]

# =========================
# SAVE
# =========================
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df_pca.to_csv(OUT_CSV, index=False)

print("Done.")
print(f"Saved PCA embeddings to: {OUT_CSV}")
print("Shape:", X_pca.shape)
