import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA

df = pd.read_csv("E:/projects/near-duplicate-detection/clip_embeddings.csv")
embed_cols = [c for c in df.columns if c.startswith("dim_")]

X = df[embed_cols].to_numpy(dtype=np.float32)

# normalize
X /= np.linalg.norm(X, axis=1, keepdims=True)

pca = PCA(n_components=128, random_state=42)
X_pca = pca.fit_transform(X)

joblib.dump(pca, "E:/models/pca.joblib")
