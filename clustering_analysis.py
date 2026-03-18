#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 2026

Hierarchical clustering + subject stratification + stacked bars + UMAP
for SHAP contrastive embeddings in AD-DECODE

This script:
1) Loads SHAP embeddings
2) Loads metadata Excel
3) Merges by Subject_ID / MRI_Exam
4) Recreates hierarchical clustering from embeddings
5) Assigns subjects to clusters (Cluster_HC)
6) Performs subject stratification statistics
7) Creates boxplots
8) Creates stacked barplots for categorical cluster composition
9) Computes UMAP for visualization
10) Saves tables and figures
11) Shows plots on screen and prints tables to console
"""

# =========================================================
# IMPORTS
# =========================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import kruskal, chi2_contingency

# =========================================================
# OPTIONAL: BETTER WINDOWS FOR PLOTS IN SPYDER
# Uncomment if figures do not pop up in separate windows
# =========================================================
# import matplotlib
# matplotlib.use("Qt5Agg")

# =========================================================
# PATHS
# =========================================================
WORK = os.environ["WORK"]

EMBED_PATH = os.path.join(
    WORK,
    "ines/results/contrastive_learning_addecode_shap/shap_embeddings_with_riskprob.csv"
)

META_PATH = os.path.join(
    WORK,
    "ines/results/merged_data/AD_DECODE_data6_with_cBAGs_and_PCA.xlsx"
)

OUT_DIR = os.path.join(
    WORK,
    "ines/results/shap_hc_stratification_cbag_withPCA"
)
os.makedirs(OUT_DIR, exist_ok=True)

print("Embeddings path:", EMBED_PATH)
print("Metadata path:", META_PATH)
print("Output dir:", OUT_DIR)

# =========================================================
# CONFIG
# =========================================================
cluster_col = "Cluster_HC"
cbag_col = "cBAG_withPCA"
n_clusters = 4   # change to 3 or 4 as needed

continuous_vars_preferred = [
    "cBAG_withPCA",
    "age",
    "BMI",
    "MOCA_TOTAL",
    "Memory_Composite",
    "Executive_Function_Composite",
    "Global_Cognition_Composite",
    "Memory_Composite_resid",
    "Executive_Function_Composite_resid",
    "Global_Cognition_Composite_resid"
]

categorical_vars_preferred = [
    "Risk_y",
    "risk_for_ad_y",
    "APOE_y",
    "genotype_y",
    "sex_y"
]


# only the boxplots you want
boxplot_vars_preferred = [
    "cBAG_withPCA",
    "age",
    "BMI",
    "MOCA_TOTAL",
    "Global_Cognition_Composite"
]

# stacked bars you want
stacked_bar_vars_preferred = [
    "Risk_y",
    "APOE_y",
    "genotype_y",
    "sex_y"
]

# category orders for prettier plots
category_orders = {
    "Risk_y": ["NoRisk", "Familial", "MCI", "AD"],
    "risk_for_ad_y": ["NoRisk", "Familial", "MCI", "AD"],
    "APOE_y": ["E4-", "E4+"],
    "genotype_y": ["APOE23", "APOE33", "APOE34", "APOE44"],
    "sex_y": ["F", "M"]
}

# =========================================================
# PLOT STYLE
# =========================================================
sns.set(style="whitegrid", font_scale=1.1)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def epsilon_squared_kruskal(H, n, k):
    if n <= k:
        return np.nan
    return max(0, (H - k + 1) / (n - k))

def cramers_v(tab):
    chi2, p, dof, expected = chi2_contingency(tab)
    n = tab.to_numpy().sum()
    r, c = tab.shape
    denom = n * min(r - 1, c - 1)
    v = np.sqrt(chi2 / denom) if denom > 0 else np.nan
    return chi2, p, dof, v

def p_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"

def nice_risk_label(series):
    mapper = {
        0: "NoRisk",
        1: "Familial",
        2: "MCI",
        3: "AD",
        "0": "NoRisk",
        "1": "Familial",
        "2": "MCI",
        "3": "AD"
    }
    return series.map(mapper).fillna(series.astype(str))

def clean_categorical(series, varname):
    s = series.copy()

    if varname in ["risk_for_ad", "risk_for_ad_x", "risk_for_ad_y"]:
        s = nice_risk_label(s)

    s = s.astype(str).str.strip()

    # normalize common missing strings
    s = s.replace({
        "": np.nan,
        "nan": np.nan,
        "NaN": np.nan,
        "None": np.nan,
        "none": np.nan,
        "NA": np.nan,
        "N/A": np.nan
    })

    # small normalizations
    if varname == "sex":
        s = s.replace({
            "Female": "F", "female": "F", "f": "F",
            "Male": "M", "male": "M", "m": "M"
        })

    return s

def show_and_save(fig_path):
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show(block=True)
    plt.close()

# =========================================================
# 1. LOAD EMBEDDINGS
# =========================================================
df_embed = pd.read_csv(EMBED_PATH)

if "Subject_ID" not in df_embed.columns:
    raise KeyError("Embeddings file must contain 'Subject_ID'.")

df_embed["Subject_ID"] = df_embed["Subject_ID"].astype(str).str.zfill(5)

embed_cols = [c for c in df_embed.columns if c.startswith("embed_")]
if len(embed_cols) == 0:
    raise RuntimeError("No embedding columns found. Expected columns starting with 'embed_'.")

print(f"\nEmbeddings loaded: {df_embed.shape[0]} subjects")
print(f"Number of embedding dimensions: {len(embed_cols)}")

# =========================================================
# 2. LOAD METADATA
# =========================================================
df_meta = pd.read_excel(META_PATH)

if "MRI_Exam" not in df_meta.columns:
    raise KeyError("Metadata file must contain 'MRI_Exam'.")

df_meta["MRI_Exam_fixed"] = (
    df_meta["MRI_Exam"]
    .fillna(0)
    .astype(int)
    .astype(str)
    .str.zfill(5)
)

print(f"Metadata loaded: {df_meta.shape[0]} rows")

# =========================================================
# 3. MERGE
# =========================================================
df = df_embed.merge(
    df_meta,
    left_on="Subject_ID",
    right_on="MRI_Exam_fixed",
    how="inner"
)
print("\n=== CHECK CATEGORICAL COLUMNS AFTER MERGE ===")
print("Columns in merged df:")
print(df.columns.tolist())

for col in ["Risk", "risk_for_ad", "APOE", "genotype", "sex"]:
    print(f"\n--- {col} ---")
    if col in df.columns:
        print("Non-null count:", df[col].notna().sum())
        print("Unique raw values:")
        print(df[col].dropna().astype(str).str.strip().value_counts(dropna=False).to_string())
    else:
        print("Column NOT found")
print(f"Merged subjects: {df.shape[0]}")

if df.shape[0] == 0:
    raise RuntimeError("Merge returned 0 rows. Check Subject_ID / MRI_Exam formatting.")

if cbag_col not in df.columns:
    raise KeyError(f"Column '{cbag_col}' not found in merged dataframe.")

print("\nMerged columns:")
print(df.columns.tolist())

# =========================================================
# 4. RECREATE HIERARCHICAL CLUSTERING FROM EMBEDDINGS
# =========================================================
X = df[embed_cols].copy()
X_scaled = StandardScaler().fit_transform(X)

Z = linkage(X_scaled, method="ward", metric="euclidean")

df[cluster_col] = fcluster(Z, t=n_clusters, criterion="maxclust")
df[cluster_col] = df[cluster_col].astype(int).astype(str)

cluster_order = sorted(df[cluster_col].unique(), key=lambda x: int(x))

print("\n=== CLUSTER SIZES ===")
print(df[cluster_col].value_counts().sort_index().to_string())

df.to_csv(os.path.join(OUT_DIR, "merged_embeddings_metadata_with_hc_clusters.csv"), index=False)

# =========================================================
# 5. SELECT VARIABLES
# =========================================================
continuous_vars = [v for v in continuous_vars_preferred if v in df.columns]
categorical_vars = [v for v in categorical_vars_preferred if v in df.columns]
boxplot_vars = [v for v in boxplot_vars_preferred if v in df.columns]
stacked_bar_vars = [v for v in stacked_bar_vars_preferred if v in df.columns]

print("\nContinuous vars used:")
print(continuous_vars)

print("\nCategorical vars used:")
print(categorical_vars)

print("\nBoxplot vars used:")
print(boxplot_vars)

print("\nStacked bar vars used:")
print(stacked_bar_vars)

# =========================================================
# 6. SUBJECT STRATIFICATION: SUMMARY TABLES
# =========================================================
summary_rows = []

for cl in cluster_order:
    sub = df[df[cluster_col] == cl]
    row = {"cluster": cl, "N": len(sub)}

    for v in continuous_vars:
        vals = pd.to_numeric(sub[v], errors="coerce")
        row[f"{v}_mean"] = vals.mean()
        row[f"{v}_std"] = vals.std()
        row[f"{v}_median"] = vals.median()

    for v in categorical_vars:
        vals = clean_categorical(sub[v], v)
        counts = vals.value_counts(dropna=False)

        for cat, count in counts.items():
            row[f"{v}_{cat}_n"] = count
            row[f"{v}_{cat}_pct"] = 100 * count / len(sub) if len(sub) > 0 else np.nan

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT_DIR, "cluster_summary_full.csv"), index=False)

paper_rows = []
for cl in cluster_order:
    sub = df[df[cluster_col] == cl]
    row = {"cluster": cl, "N": len(sub)}

    for v in continuous_vars:
        vals = pd.to_numeric(sub[v], errors="coerce")
        if vals.notna().sum() > 0:
            row[v] = f"{vals.mean():.2f} ± {vals.std():.2f}"
        else:
            row[v] = np.nan

    paper_rows.append(row)

paper_summary_df = pd.DataFrame(paper_rows)
paper_summary_df.to_csv(os.path.join(OUT_DIR, "cluster_summary_paper_format.csv"), index=False)

print("\n=== FULL CLUSTER SUMMARY ===")
print(summary_df.to_string(index=False))

print("\n=== PAPER-STYLE CLUSTER SUMMARY ===")
print(paper_summary_df.to_string(index=False))

# =========================================================
# 7. STATISTICAL TESTS
# =========================================================
continuous_stats = []

for v in continuous_vars:
    tmp = df[[cluster_col, v]].copy()
    tmp[v] = pd.to_numeric(tmp[v], errors="coerce")
    tmp = tmp.dropna()

    groups = [tmp.loc[tmp[cluster_col] == cl, v].values for cl in cluster_order]
    groups = [g for g in groups if len(g) > 0]

    if len(groups) >= 2:
        H, p = kruskal(*groups)
        n = len(tmp)
        k = len(groups)
        eps2 = epsilon_squared_kruskal(H, n, k)

        continuous_stats.append({
            "variable": v,
            "test": "Kruskal-Wallis",
            "H": H,
            "p": p,
            "p_stars": p_to_stars(p),
            "epsilon_squared": eps2
        })

continuous_stats_df = pd.DataFrame(continuous_stats)
continuous_stats_df.to_csv(os.path.join(OUT_DIR, "continuous_stats_by_cluster.csv"), index=False)

categorical_stats = []

for v in categorical_vars:
    tmp = df[[cluster_col, v]].copy()
    tmp[v] = clean_categorical(tmp[v], v)
    tmp = tmp.dropna()

    if tmp.empty:
        continue

    tab = pd.crosstab(tmp[cluster_col], tmp[v])

    if tab.shape[0] >= 2 and tab.shape[1] >= 2:
        chi2, p, dof, cv = cramers_v(tab)

        categorical_stats.append({
            "variable": v,
            "test": "Chi-square",
            "chi2": chi2,
            "p": p,
            "p_stars": p_to_stars(p),
            "dof": dof,
            "cramers_v": cv
        })

        tab.to_csv(os.path.join(OUT_DIR, f"crosstab_{v}_by_cluster.csv"))

        print(f"\n=== CROSSTAB: {v} by {cluster_col} ===")
        print(tab.to_string())

categorical_stats_df = pd.DataFrame(categorical_stats)
categorical_stats_df.to_csv(os.path.join(OUT_DIR, "categorical_stats_by_cluster.csv"), index=False)

print("\n=== CONTINUOUS STATS BY CLUSTER ===")
if not continuous_stats_df.empty:
    print(continuous_stats_df.to_string(index=False))
else:
    print("No continuous stats available.")

print("\n=== CATEGORICAL STATS BY CLUSTER ===")
if not categorical_stats_df.empty:
    print(categorical_stats_df.to_string(index=False))
else:
    print("No categorical stats available.")

# =========================================================
# 8. BOXPLOTS
# =========================================================
for v in boxplot_vars:
    plt.figure(figsize=(6.5, 5.5))
    ax = sns.boxplot(
        data=df,
        x=cluster_col,
        y=v,
        order=cluster_order
    )
    sns.stripplot(
        data=df,
        x=cluster_col,
        y=v,
        order=cluster_order,
        color="black",
        alpha=0.6,
        size=4
    )

    counts = df[cluster_col].value_counts().sort_index()
    ymin, ymax = plt.ylim()
    y_text = ymin - 0.08 * (ymax - ymin)

    for i, cl in enumerate(cluster_order):
        n = counts.get(cl, 0)
        ax.text(i, y_text, f"N={n}", ha="center", va="top", fontsize=10)

    p_row = continuous_stats_df.loc[continuous_stats_df["variable"] == v]
    if not p_row.empty:
        p_val = p_row["p"].values[0]
        stars = p_row["p_stars"].values[0]
        plt.title(f"{v} by hierarchical cluster\nKruskal-Wallis p={p_val:.4f} ({stars})")
    else:
        plt.title(f"{v} by hierarchical cluster")

    plt.xlabel("Hierarchical cluster")
    plt.ylabel(v)

    show_and_save(os.path.join(OUT_DIR, f"boxplot_{v}_by_cluster.png"))

print("\nSaved boxplots.")

# =========================================================
# 9. STACKED BARPLOTS OF CLUSTER COMPOSITION
# =========================================================
print("\n================ STACKED BARPLOTS DEBUG ================")
print("Categorical variables for stacked bars:", stacked_bar_vars)
print("Cluster order:", cluster_order)

for v in stacked_bar_vars:
    print(f"\n----- Variable: {v} -----")

    tmp = df[[cluster_col, v]].copy()
    tmp[cluster_col] = tmp[cluster_col].astype(str).str.strip()
    tmp[v] = clean_categorical(tmp[v], v)
    tmp = tmp.dropna(subset=[cluster_col, v])

    print("Unique cleaned values:", sorted(tmp[v].unique().tolist()) if not tmp.empty else [])
    print("N rows after cleaning:", len(tmp))

    if tmp.empty:
        print(f"Skipping {v}: no valid rows after cleaning.")
        continue

    tab_counts = pd.crosstab(tmp[cluster_col], tmp[v])
    tab_counts = tab_counts.reindex(cluster_order)

    if v in category_orders:
        desired_cols = [c for c in category_orders[v] if c in tab_counts.columns]
        other_cols = [c for c in tab_counts.columns if c not in desired_cols]
        tab_counts = tab_counts[desired_cols + other_cols]

    row_sums = tab_counts.sum(axis=1)
    valid_rows = row_sums[row_sums > 0].index.tolist()
    tab_counts = tab_counts.loc[valid_rows]

    print("\nCounts table:")
    print(tab_counts.to_string())

    if tab_counts.empty:
        print(f"Skipping {v}: empty table.")
        continue

    tab_pct = tab_counts.div(tab_counts.sum(axis=1), axis=0) * 100

    print("\nPercent table:")
    print(tab_pct.round(2).to_string())

    if tab_counts.shape[0] >= 2 and tab_counts.shape[1] >= 2:
        chi2, p, dof, expected = chi2_contingency(tab_counts)
        title = f"{v} composition by hierarchical cluster\nChi-square p={p:.4f}"
        print(f"\nChi-square = {chi2:.3f}, dof = {dof}, p = {p:.4f}")
    else:
        p = np.nan
        title = f"{v} composition by hierarchical cluster"
        print("\nChi-square not computed.")

    ax = tab_pct.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5.5),
        width=0.8
    )

    cluster_counts = tab_counts.sum(axis=1)
    for i, cl in enumerate(tab_pct.index):
        n = cluster_counts.loc[cl]
        ax.text(i, 102, f"N={n}", ha="center", va="bottom", fontsize=10)

    ax.set_title(title)
    ax.set_xlabel("Hierarchical cluster")
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 110)
    ax.legend(title=v, bbox_to_anchor=(1.02, 1), loc="upper left")

    fig = ax.get_figure()
    out_file = os.path.join(OUT_DIR, f"stackedbar_{v}_by_cluster.png")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    print("Saved file:", out_file)

    plt.show(block=True)
    plt.close(fig)

    tab_counts.to_csv(os.path.join(OUT_DIR, f"stackedbar_counts_{v}_by_cluster.csv"))
    tab_pct.to_csv(os.path.join(OUT_DIR, f"stackedbar_percent_{v}_by_cluster.csv"))

print("\nSaved stacked barplots.")
# =========================================================
# 10. UMAP FROM EMBEDDINGS
# =========================================================
reducer = umap.UMAP(
    n_neighbors=10,
    min_dist=0.3,
    n_components=2,
    random_state=42
)

X_umap = reducer.fit_transform(X_scaled)
df["UMAP1"] = X_umap[:, 0]
df["UMAP2"] = X_umap[:, 1]

df.to_csv(os.path.join(OUT_DIR, "merged_embeddings_metadata_with_hc_clusters_umap.csv"), index=False)

# =========================================================
# 11. UMAP PLOTS
# =========================================================
plt.figure(figsize=(7, 6))
sns.scatterplot(
    data=df,
    x="UMAP1",
    y="UMAP2",
    hue=cluster_col,
    hue_order=cluster_order,
    palette="Set2",
    s=70,
    alpha=0.9,
    edgecolor="k",
    linewidth=0.3
)
plt.title("UMAP of SHAP embeddings colored by hierarchical cluster")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
show_and_save(os.path.join(OUT_DIR, "umap_by_hierarchical_cluster.png"))

plt.figure(figsize=(7, 6))
sc = plt.scatter(
    df["UMAP1"],
    df["UMAP2"],
    c=pd.to_numeric(df[cbag_col], errors="coerce"),
    s=70,
    edgecolors="k",
    linewidths=0.3
)
plt.colorbar(sc, label=cbag_col)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title(f"UMAP of SHAP embeddings colored by {cbag_col}")
show_and_save(os.path.join(OUT_DIR, f"umap_by_{cbag_col}.png"))

if "risk_for_ad" in df.columns:
    df["risk_for_ad_label"] = clean_categorical(df["risk_for_ad"], "risk_for_ad")

    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=df,
        x="UMAP1",
        y="UMAP2",
        hue="risk_for_ad_label",
        palette="Set2",
        s=70,
        alpha=0.9,
        edgecolor="k",
        linewidth=0.3
    )
    plt.title("UMAP of SHAP embeddings colored by risk_for_ad")
    plt.legend(title="Risk", bbox_to_anchor=(1.05, 1), loc="upper left")
    show_and_save(os.path.join(OUT_DIR, "umap_by_risk_for_ad.png"))

elif "Risk" in df.columns:
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=df,
        x="UMAP1",
        y="UMAP2",
        hue="Risk",
        palette="Set2",
        s=70,
        alpha=0.9,
        edgecolor="k",
        linewidth=0.3
    )
    plt.title("UMAP of SHAP embeddings colored by Risk")
    plt.legend(title="Risk", bbox_to_anchor=(1.05, 1), loc="upper left")
    show_and_save(os.path.join(OUT_DIR, "umap_by_Risk.png"))

print("\nSaved UMAP plots.")

# =========================================================
# 12. EXTRA CONSOLE SUMMARY
# =========================================================
cluster_counts = df[cluster_col].value_counts().sort_index()
cluster_counts.to_csv(os.path.join(OUT_DIR, "cluster_sizes.csv"), header=["N"])

print("\n=== CLUSTER SIZES ===")
print(cluster_counts.to_string())

key_vars = [v for v in [cbag_col, "age", "BMI", "MOCA_TOTAL", "Global_Cognition_Composite"] if v in df.columns]

if key_vars:
    print("\n=== MEAN / SD / MEDIAN / COUNT OF KEY VARIABLES BY CLUSTER ===")
    for v in key_vars:
        print(f"\n{v}")
        tmp = df.groupby(cluster_col)[v].agg(["mean", "std", "median", "count"])
        print(tmp.to_string())

print("\n===================================")
print("DONE")
print("Main outputs saved in:", OUT_DIR)
print("Files created:")
print("- merged_embeddings_metadata_with_hc_clusters.csv")
print("- merged_embeddings_metadata_with_hc_clusters_umap.csv")
print("- cluster_summary_full.csv")
print("- cluster_summary_paper_format.csv")
print("- continuous_stats_by_cluster.csv")
print("- categorical_stats_by_cluster.csv")
print("- boxplots / stacked bars / UMAP pngs")
print("===================================")