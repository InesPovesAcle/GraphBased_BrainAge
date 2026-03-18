#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SHAP Embedding Heatmap with Hierarchical Clustering

- Loads SHAP contrastive embeddings + risk labels
- Keeps only embedding dimensions in the heatmap
- Uses risk_for_ad only as row color annotation
- Scales embedding dimensions across subjects
- Saves clustered heatmap and ordered subject table
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler

################ PATHS ################

WORK = os.environ["WORK"]

INPUT_PATH = os.path.join(
    WORK,
    "ines/results/contrastive_learning_addecode_shap/shap_embeddings_with_riskprob.csv"
)

OUT_DIR = os.path.join(
    WORK,
    "ines/results/shap_embedding_heatmap"
)
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_FIG = os.path.join(OUT_DIR, "shap_embedding_heatmap_improved.png")
OUTPUT_CSV = os.path.join(OUT_DIR, "shap_embedding_heatmap_ordered_subjects.csv")

print("Input:", INPUT_PATH)
print("Output fig:", OUTPUT_FIG)

################ LOAD DATA ################

df = pd.read_csv(INPUT_PATH)

if "Subject_ID" not in df.columns:
    raise KeyError("Missing 'Subject_ID' column.")

if "risk_for_ad" not in df.columns:
    raise KeyError("Missing 'risk_for_ad' column.")

df["Subject_ID"] = df["Subject_ID"].astype(str).str.zfill(5)

################ EXTRACT ONLY EMBEDDINGS ################

# Keep only columns that really are embeddings
embedding_matrix = df.loc[:, df.columns.str.startswith("embed_")].copy()

if embedding_matrix.shape[1] == 0:
    raise RuntimeError("No embedding columns found. Expected columns starting with 'embed_'.")

# Extra safety in case risk slipped in somehow
embedding_matrix = embedding_matrix.drop(columns=["risk_for_ad"], errors="ignore")

# Set subject IDs as row index
embedding_matrix.index = df["Subject_ID"]

# Remove any residual names that seaborn may display
embedding_matrix.columns.name = None
embedding_matrix.index.name = None

print("Columns used in heatmap:")
print(list(embedding_matrix.columns))

################ STANDARDIZE EMBEDDINGS ################

scaler = StandardScaler()
embedding_scaled = pd.DataFrame(
    scaler.fit_transform(embedding_matrix),
    index=embedding_matrix.index,
    columns=embedding_matrix.columns
)

# Remove names again after reconstruction
embedding_scaled.columns.name = None
embedding_scaled.index.name = None

################ DEFINE ROW COLORS ################

group_colors = {
    0: "green",
    1: "blue",
    2: "orange",
    3: "red"
}

row_colors = df["risk_for_ad"].map(group_colors)
row_colors.index = df["Subject_ID"]
row_colors.name = None

################ CLUSTER HEATMAP ################

sns.set(style="white", font_scale=1.0)

g = sns.clustermap(
    embedding_scaled,
    cmap="vlag",
    center=0,
    row_cluster=True,
    col_cluster=True,
    yticklabels=False,
    xticklabels=True,
    figsize=(14, 10),
    dendrogram_ratio=(0.16, 0.18),
    cbar_kws={"label": "Scaled Embedding Value"},
    linewidths=0
)

################ CLEAN LABELS ################

# Force only embedding column labels to appear
g.ax_heatmap.set_xticklabels(g.data2d.columns, rotation=90)

# Explicitly remove any possible axis names
g.ax_heatmap.set_xlabel("Embedding Dimension", fontsize=13)
g.ax_heatmap.set_ylabel("Subjects", fontsize=13)
g.ax_heatmap.xaxis.labelpad = 10

# These lines prevent names like 'risk_for_ad' from appearing as headers
g.ax_heatmap.set_title("")
g.ax_row_dendrogram.set_title("")
g.ax_col_dendrogram.set_title("")
g.data2d.columns.name = None
g.data2d.index.name = None

################ LEGEND ################



################ TITLE ################

g.fig.suptitle(
    "SHAP Embedding Heatmap (Hierarchical Clustering)",
    fontsize=16,
    y=1.02
)

################ SAVE ORDERED SUBJECTS ################

ordered_subjects = embedding_scaled.index[g.dendrogram_row.reordered_ind]

df_ordered = df.set_index("Subject_ID").loc[ordered_subjects].reset_index()
df_ordered.to_csv(OUTPUT_CSV, index=False)

################ SAVE FIGURE ################

plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
plt.close()

print("Saved figure:", OUTPUT_FIG)
print("Saved ordered subjects:", OUTPUT_CSV)
print("DONE.")