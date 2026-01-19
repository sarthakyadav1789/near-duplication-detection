from vector import get_image_embedding
import pandas as pd
import numpy as np
from readcsv import read_csv_file
from display import show_images
import os

# =========================
# CONFIG
# =========================
IMAGE_ROOT = "E:/IMAGES"
CSV_PATH = "E:/projects/near-duplicate-detection/clip_embeddings.csv"
TARGET_IMAGE = "E:/IMAGES/0/0/0/0000059611c7d079.jpg"
TOP_K = 5

# =========================
# TARGET EMBEDDING
# =========================
target = get_image_embedding(TARGET_IMAGE)

# Force CPU + NumPy
if hasattr(target, "detach"):
    target = target.detach().cpu().numpy()

target = np.asarray(target, dtype=np.float32).reshape(-1)

t_norm = np.linalg.norm(target)
if t_norm == 0:
    raise ValueError("Target embedding norm is ZERO")
target /= t_norm

# =========================
# LOAD DATASET EMBEDDINGS
# =========================
dataset = read_csv_file(CSV_PATH)

embed_cols = [c for c in dataset.columns if c.startswith("dim_")]

embeddings = (
    dataset[embed_cols]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0.0)
    .to_numpy(dtype=np.float32)
)

# =========================
# NORMALIZE DATASET (SAFE)
# =========================
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
mask = norms.squeeze() > 0

embeddings = embeddings[mask]
dataset = dataset.iloc[mask]
embeddings /= norms[mask]

# =========================
# COSINE SIMILARITY
# =========================
scores = embeddings @ target

top_idx = np.argsort(scores)[-TOP_K:][::-1]
results = dataset.iloc[top_idx].copy()
results["similarity"] = scores[top_idx]

print("\nTop similar images:")
print("=" * 100)
print(results[["folder", "file", "similarity"]])

# =========================
# DISPLAY IMAGES (SAFE)
# =========================
image_paths = []
similarities = []
basepth = "E:/IMAGES/0/0"
for _, row in results.iterrows():
    img_path = os.path.join(
        basepth,
        str(row["folder"]),
        str(row["file"])
    )

    if os.path.exists(img_path):
        image_paths.append(img_path)
        similarities.append(float(row["similarity"]))
    else:
        print("Missing file:", img_path)

try:
    show_images(image_paths, similarities, cols=3)
except Exception as e:
    print("Display error:", e)
    print("Paths:", image_paths)
