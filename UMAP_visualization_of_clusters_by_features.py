#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 18:31:57 2026

@author: ines
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UMAP Visualization of SHAP-based Embeddings

Steps:
1. Load SHAP-based embeddings (from contrastive learning)
2. Load subject-level metadata
3. Merge embeddings and metadata using Subject_ID / MRI_Exam
4. Standardize embeddings
5. Apply UMAP to project embeddings into 2D
6. Visualize clustering by clinical/genetic variables
"""

################# IMPORTS ################

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

from sklearn.preprocessing import StandardScaler

################# PATHS ################

WORK = os.environ["WORK"]

EMBED_PATH = os.path.join(
    WORK,
    "ines/results/contrastive_learning_addecode_shap/shap_embeddings.csv"
)

META_PATH = os.path.join(
    WORK,
    "ines/data/AD_DECODE_data4.xlsx"
)

OUT_DIR = os.path.join(
    WORK,
    "ines/results/umap_shap_embeddings"
)
os.makedirs(OUT_DIR, exist_ok=True)

print("Embeddings path:", EMBED_PATH)
print("Metadata path:", META_PATH)
print("Output dir:", OUT_DIR)

################# 1. LOAD EMBEDDINGS ################

df_embed = pd.read_csv(EMBED_PATH)

if "Subject_ID" not in df_embed.columns:
    raise KeyError("shap_embeddings.csv must contain a 'Subject_ID' column.")

df_embed["Subject_ID"] = df_embed["Subject_ID"].astype(str).str.zfill(5)

print(f"Embeddings loaded: {df_embed.shape[0]} subjects")

################# 2. LOAD METADATA ################

df_meta = pd.read_excel(META_PATH)

if "MRI_Exam" not in df_meta.columns:
    raise KeyError("Metadata file must contain an 'MRI_Exam' column.")

df_meta["MRI_Exam_fixed"] = (
    df_meta["MRI_Exam"]
    .fillna(0)
    .astype(int)
    .astype(str)
    .str.zfill(5)
)

print(f"Metadata loaded: {df_meta.shape[0]} rows")

################# 3. MERGE EMBEDDINGS + METADATA ################

df = df_embed.merge(
    df_meta,
    left_on="Subject_ID",
    right_on="MRI_Exam_fixed",
    how="inner"
)

print(f"Merged subjects: {df.shape[0]}")

if df.shape[0] == 0:
    raise RuntimeError("Merge returned 0 rows. Check Subject_ID / MRI_Exam formatting.")

################# 4. EXTRACT + SCALE EMBEDDINGS ################

embed_cols = [col for col in df.columns if col.startswith("embed_")]

if len(embed_cols) == 0:
    raise RuntimeError("No embedding columns found. Expected columns starting with 'embed_'.")

X = df[embed_cols].values
X_scaled = StandardScaler().fit_transform(X)

print("Embedding matrix shape:", X.shape)

################# 5. APPLY UMAP ################

reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X_scaled)

df["UMAP1"] = X_umap[:, 0]
df["UMAP2"] = X_umap[:, 1]

# Save merged table with UMAP coordinates
df.to_csv(os.path.join(OUT_DIR, "umap_embeddings_with_metadata.csv"), index=False)
print("Saved:", os.path.join(OUT_DIR, "umap_embeddings_with_metadata.csv"))

################# 6. PLOTTING FUNCTION ################

def plot_umap_by(column, palette="Set2", save=True):
    """
    Visualize UMAP embeddings colored by a metadata column.
    """

    if column not in df.columns:
        print(f"Skipping {column}: column not found.")
        return

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="UMAP1",
        y="UMAP2",
        hue=column,
        palette=palette,
        s=60,
        alpha=0.9,
        edgecolor="k",
        linewidth=0.3
    )

    plt.title(f"UMAP of SHAP Embeddings colored by: {column}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if save:
        filename = os.path.join(OUT_DIR, f"umap_by_{column}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved: {filename}")

    plt.close()

################# 7. EXAMPLE PLOTS ################

plot_umap_by("age", palette="viridis", save=True)
plot_umap_by("APOE", palette="coolwarm", save=True)
plot_umap_by("genotype", palette="coolwarm", save=True)
plot_umap_by("sex", palette="Set1", save=True)

################# 8. RISK GROUP PLOT ################

# Try to make a readable risk label depending on available columns
if "risk_for_ad" in df.columns:
    risk_labels = {
        0: "No risk",
        1: "Familial",
        2: "MCI",
        3: "AD"
    }
    df["Risk_Label"] = df["risk_for_ad"].map(risk_labels)

elif "Risk" in df.columns:
    df["Risk_Label"] = df["Risk"].fillna("NoRisk").replace(r"^\s*$", "NoRisk", regex=True)

else:
    df["Risk_Label"] = "Unknown"

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="UMAP1",
    y="UMAP2",
    hue="Risk_Label",
    palette="Set2",
    s=60,
    alpha=0.9,
    edgecolor="k",
    linewidth=0.3
)
plt.title("UMAP of SHAP Embeddings colored by Risk Group")
plt.legend(title="Risk Group", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

risk_plot_path = os.path.join(OUT_DIR, "umap_by_risk_group.png")
plt.savefig(risk_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", risk_plot_path)

print("\nDONE.")