#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 18:39:07 2026

@author: ines
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clustering SHAP Contrastive Embeddings
- Load SHAP embeddings
- Standardize embeddings
- Compute UMAP for visualization
- Run KMeans on original embeddings
- Save plot and cluster assignments
"""

################# IMPORTS ################

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

################# PATHS ################

WORK = os.environ["WORK"]

EMBED_PATH = os.path.join(
    WORK,
    "ines/results/contrastive_learning_addecode_shap/shap_embeddings.csv"
)

OUT_DIR = os.path.join(
    WORK,
    "ines/results/shap_embedding_clustering"
)
os.makedirs(OUT_DIR, exist_ok=True)

CLUSTER_CSV_OUT = os.path.join(OUT_DIR, "shap_embedding_clusters.csv")
PLOT_OUT = os.path.join(OUT_DIR, "umap_kmeans_clusters_k4.png")

print("Embeddings path:", EMBED_PATH)
print("Output dir:", OUT_DIR)

################# 1. LOAD SHAP EMBEDDINGS ################

df_embed = pd.read_csv(EMBED_PATH)

if "Subject_ID" not in df_embed.columns:
    raise KeyError("shap_embeddings.csv must contain a 'Subject_ID' column.")

df_embed["Subject_ID"] = df_embed["Subject_ID"].astype(str).str.zfill(5)

################# 2. EXTRACT EMBEDDING MATRIX ################

embed_cols = [col for col in df_embed.columns if col.startswith("embed_")]

if len(embed_cols) == 0:
    raise RuntimeError("No embedding columns found. Expected columns starting with 'embed_'.")

X = df_embed[embed_cols].values
print("Embedding matrix shape:", X.shape)

################# 3. SCALE BEFORE UMAP / KMEANS ################

X_scaled = StandardScaler().fit_transform(X)

################# 4. UMAP FOR 2D VISUALIZATION ################

reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X_scaled)

df_embed["UMAP1"] = X_umap[:, 0]
df_embed["UMAP2"] = X_umap[:, 1]

################# 5. KMEANS CLUSTERING ################

k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_embed["Cluster"] = kmeans.fit_predict(X_scaled)

################# 6. PLOT ################

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_embed,
    x="UMAP1",
    y="UMAP2",
    hue="Cluster",
    palette="Set2",
    s=60,
    edgecolor="k",
    linewidth=0.3
)
plt.title(f"UMAP of SHAP Embeddings with K-Means Clusters (k={k})")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", PLOT_OUT)

################# 7. SAVE CLUSTER ASSIGNMENTS ################

df_embed[["Subject_ID", "UMAP1", "UMAP2", "Cluster"]].to_csv(CLUSTER_CSV_OUT, index=False)
print("Saved:", CLUSTER_CSV_OUT)

print("\nDONE.")