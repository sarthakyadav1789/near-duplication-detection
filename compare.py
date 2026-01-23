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
IMAGE_BASE = "E:/IMAGES/0/0"

TOP_K = 5


def run(target_image_path, candidate_filenames):
    """
    Run CLIP + PCA + cluster-based comparison on candidate images.

    Parameters:
    - target_image_path (str)
    - candidate_filenames (list[str])

    Returns:
    - results DataFrame
    """

    if not candidate_filenames:
        print("❌ No candidates passed to AI.")
        return None

    # =========================
    # LOAD PCA
    # =========================
    pca = joblib.load(PCA_MODEL)

    # =========================
    # TARGET EMBEDDING
    # =========================
    target = get_image_embedding(target_image_path)
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

    # 🔹 filter using pHash-passed filenames
    df = df[df["file"].isin(candidate_filenames)].reset_index(drop=True)

    if df.empty:
        print("❌ No matching rows found after pHash filtering.")
        return None

    pca_cols = [c for c in df.columns if c.startswith("pca_")]
    X_pca = df[pca_cols].to_numpy(dtype=np.float32)

    # =========================
    # COSINE SIMILARITY
    # =========================
    scores = X_pca @ target_pca

    top_idx = np.argsort(scores)[-TOP_K:][::-1]
    results = df.iloc[top_idx].copy()
    results["similarity"] = scores[top_idx]

    print("\n🤖 Top similar images:")
    print("=" * 100)
    print(results[["folder", "file", "similarity", "cluster_id"]])

    # =========================
    # DISPLAY
    # =========================
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

    return results
