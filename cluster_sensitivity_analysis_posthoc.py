#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sensitivity analysis for hierarchical clustering on SHAP embeddings
Runs the full downstream analysis for k = 2, 3, 4

Outputs for each k:
- merged dataframe with Cluster_HC
- summary tables
- continuous stats
- categorical stats
- pairwise post hoc tests for continuous variables
- pairwise post hoc tests for categorical variables
- boxplots
- stacked barplots
- UMAP by cluster
- UMAP by cBAG
- optional UMAP by risk
"""

# =========================================================
# IMPORTS
# =========================================================
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import kruskal, chi2_contingency, mannwhitneyu, fisher_exact
from statsmodels.stats.multitest import multipletests

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

OUT_BASE = os.path.join(
    WORK,
    "ines/results/shap_hc_sensitivity_analysis"
)
os.makedirs(OUT_BASE, exist_ok=True)

print("Embeddings path:", EMBED_PATH)
print("Metadata path:", META_PATH)
print("Output base dir:", OUT_BASE)

# =========================================================
# CONFIG
# =========================================================
cluster_col = "Cluster_HC"
cbag_col = "cBAG_withPCA"
k_list = [2, 3, 4]

# Post hoc settings
run_continuous_posthoc_only_if_global_sig = True
run_categorical_posthoc_only_if_global_sig = True
alpha_global = 0.05
alpha_posthoc = 0.05
posthoc_correction_method = "fdr_bh"   # "bonferroni" or "fdr_bh"

continuous_vars_preferred = [
    "cBAG_withPCA",
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

boxplot_vars_preferred = [
    "cBAG_withPCA",
    "BMI",
    "MOCA_TOTAL",
    "Global_Cognition_Composite",
    "Global_Cognition_Composite_resid"
]

stacked_bar_vars_preferred = [
    "Risk_y",
    "risk_for_ad_y",
    "APOE_y",
    "genotype_y",
    "sex_y"
]

category_orders = {
    "Risk_y": ["NoRisk", "Familial", "MCI", "AD"],
    "risk_for_ad_y": ["NoRisk", "Familial", "MCI", "AD"],
    "APOE_y": ["E4-", "E4+"],
    "genotype_y": ["APOE23", "APOE33", "APOE34", "APOE44"],
    "sex_y": ["F", "M"]
}

pretty_labels = {
    "Risk_y": "Risk",
    "risk_for_ad_y": "AD risk",
    "APOE_y": "APOE E4 carriage",
    "genotype_y": "APOE genotype",
    "sex_y": "Sex",
    "cBAG_withPCA": "cBAG_withPCA",
    "BMI": "BMI",
    "MOCA_TOTAL": "MoCA total",
    "Memory_Composite": "Memory composite",
    "Executive_Function_Composite": "Executive function composite",
    "Global_Cognition_Composite": "Global cognition composite",
    "Memory_Composite_resid": "Memory composite residual",
    "Executive_Function_Composite_resid": "Executive function composite residual",
    "Global_Cognition_Composite_resid": "Global cognition composite residual",
}

sns.set(style="whitegrid", font_scale=1.1)

# =========================================================
# HELPERS
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
    if pd.isna(p):
        return "na"
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

    s = s.replace({
        "": np.nan,
        "nan": np.nan,
        "NaN": np.nan,
        "None": np.nan,
        "none": np.nan,
        "NA": np.nan,
        "N/A": np.nan
    })

    if varname in ["sex", "sex_x", "sex_y"]:
        s = s.replace({
            "Female": "F", "female": "F", "f": "F",
            "Male": "M", "male": "M", "m": "M"
        })

    return s

def save_and_show(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def rank_biserial_from_u(x, y, u_stat):
    n1 = len(x)
    n2 = len(y)
    if n1 == 0 or n2 == 0:
        return np.nan
    return 1 - (2 * u_stat) / (n1 * n2)

def pairwise_mannwhitney_posthoc(df_in, cluster_col, var, cluster_order,
                                 correction_method="fdr_bh", alpha=0.05):
    rows = []
    tmp = df_in[[cluster_col, var]].copy()
    tmp[var] = pd.to_numeric(tmp[var], errors="coerce")
    tmp = tmp.dropna()

    pairs = list(itertools.combinations(cluster_order, 2))
    raw_pvals = []

    for c1, c2 in pairs:
        x = tmp.loc[tmp[cluster_col] == c1, var].values
        y = tmp.loc[tmp[cluster_col] == c2, var].values

        if len(x) == 0 or len(y) == 0:
            rows.append({
                "variable": var,
                "cluster_1": c1,
                "cluster_2": c2,
                "n1": len(x),
                "n2": len(y),
                "median_1": np.nan,
                "median_2": np.nan,
                "u_stat": np.nan,
                "p_raw": np.nan,
                "rank_biserial": np.nan
            })
            raw_pvals.append(np.nan)
            continue

        u_stat, p_raw = mannwhitneyu(x, y, alternative="two-sided")
        rbc = rank_biserial_from_u(x, y, u_stat)

        rows.append({
            "variable": var,
            "cluster_1": c1,
            "cluster_2": c2,
            "n1": len(x),
            "n2": len(y),
            "median_1": np.median(x),
            "median_2": np.median(y),
            "u_stat": u_stat,
            "p_raw": p_raw,
            "rank_biserial": rbc
        })
        raw_pvals.append(p_raw)

    valid_mask = np.array([not pd.isna(p) for p in raw_pvals])
    p_adj = np.full(len(raw_pvals), np.nan, dtype=float)
    reject = np.full(len(raw_pvals), False, dtype=bool)

    if valid_mask.sum() > 0:
        rej, adj, _, _ = multipletests(
            np.array(raw_pvals)[valid_mask],
            alpha=alpha,
            method=correction_method
        )
        p_adj[valid_mask] = adj
        reject[valid_mask] = rej

    out = pd.DataFrame(rows)
    out["p_adj"] = p_adj
    out["significant"] = reject
    out["p_adj_stars"] = out["p_adj"].apply(p_to_stars)
    return out

def pairwise_categorical_posthoc(df_in, cluster_col, var, cluster_order,
                                 correction_method="fdr_bh", alpha=0.05):
    rows = []
    tmp = df_in[[cluster_col, var]].copy()
    tmp[var] = clean_categorical(tmp[var], var)
    tmp = tmp.dropna()

    pairs = list(itertools.combinations(cluster_order, 2))
    raw_pvals = []

    for c1, c2 in pairs:
        sub = tmp[tmp[cluster_col].isin([c1, c2])].copy()
        tab = pd.crosstab(sub[cluster_col], sub[var])

        if tab.shape[0] < 2 or tab.shape[1] < 2:
            rows.append({
                "variable": var,
                "cluster_1": c1,
                "cluster_2": c2,
                "test": "not_computable",
                "p_raw": np.nan,
                "chi2_or_oddsratio": np.nan,
                "dof": np.nan,
                "cramers_v": np.nan,
                "table_shape": f"{tab.shape[0]}x{tab.shape[1]}"
            })
            raw_pvals.append(np.nan)
            continue

        if tab.shape == (2, 2):
            chi2, p_chi, dof, expected = chi2_contingency(tab)
            use_fisher = (expected < 5).any()

            if use_fisher:
                oddsratio, p_raw = fisher_exact(tab.values)
                rows.append({
                    "variable": var,
                    "cluster_1": c1,
                    "cluster_2": c2,
                    "test": "fisher_exact",
                    "p_raw": p_raw,
                    "chi2_or_oddsratio": oddsratio,
                    "dof": np.nan,
                    "cramers_v": np.nan,
                    "table_shape": "2x2"
                })
                raw_pvals.append(p_raw)
            else:
                chi2, p_raw, dof, cv = cramers_v(tab)
                rows.append({
                    "variable": var,
                    "cluster_1": c1,
                    "cluster_2": c2,
                    "test": "chi_square",
                    "p_raw": p_raw,
                    "chi2_or_oddsratio": chi2,
                    "dof": dof,
                    "cramers_v": cv,
                    "table_shape": "2x2"
                })
                raw_pvals.append(p_raw)
        else:
            chi2, p_raw, dof, cv = cramers_v(tab)
            rows.append({
                "variable": var,
                "cluster_1": c1,
                "cluster_2": c2,
                "test": "chi_square",
                "p_raw": p_raw,
                "chi2_or_oddsratio": chi2,
                "dof": dof,
                "cramers_v": cv,
                "table_shape": f"{tab.shape[0]}x{tab.shape[1]}"
            })
            raw_pvals.append(p_raw)

    valid_mask = np.array([not pd.isna(p) for p in raw_pvals])
    p_adj = np.full(len(raw_pvals), np.nan, dtype=float)
    reject = np.full(len(raw_pvals), False, dtype=bool)

    if valid_mask.sum() > 0:
        rej, adj, _, _ = multipletests(
            np.array(raw_pvals)[valid_mask],
            alpha=alpha,
            method=correction_method
        )
        p_adj[valid_mask] = adj
        reject[valid_mask] = rej

    out = pd.DataFrame(rows)
    out["p_adj"] = p_adj
    out["significant"] = reject
    out["p_adj_stars"] = out["p_adj"].apply(p_to_stars)
    return out

# =========================================================
# LOAD DATA ONCE
# =========================================================
df_embed = pd.read_csv(EMBED_PATH)
df_embed["Subject_ID"] = df_embed["Subject_ID"].astype(str).str.zfill(5)

embed_cols = [c for c in df_embed.columns if c.startswith("embed_")]
if not embed_cols:
    raise RuntimeError("No embed_* columns found.")

df_meta = pd.read_excel(META_PATH)
df_meta["MRI_Exam_fixed"] = (
    df_meta["MRI_Exam"]
    .fillna(0)
    .astype(int)
    .astype(str)
    .str.zfill(5)
)

df_merged = df_embed.merge(
    df_meta,
    left_on="Subject_ID",
    right_on="MRI_Exam_fixed",
    how="inner"
)

if cbag_col not in df_merged.columns:
    raise KeyError(f"{cbag_col} not found in merged dataframe.")

print("\nMerged subjects:", df_merged.shape[0])

# =========================================================
# PREPARE EMBEDDINGS ONCE
# =========================================================
X = df_merged[embed_cols].copy()
X_scaled = StandardScaler().fit_transform(X)
Z = linkage(X_scaled, method="ward", metric="euclidean")

# UMAP once, reused for all k
reducer = umap.UMAP(
    n_neighbors=10,
    min_dist=0.3,
    n_components=2,
    random_state=42
)
X_umap = reducer.fit_transform(X_scaled)

# =========================================================
# VALIDITY METRICS TABLE
# =========================================================
validity_rows = []

for k in range(2, 7):
    labels = fcluster(Z, t=k, criterion="maxclust")
    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    sizes = dict(zip(*np.unique(labels, return_counts=True)))

    validity_rows.append({
        "k": k,
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "cluster_sizes": str(sizes)
    })

validity_df = pd.DataFrame(validity_rows)
validity_df.to_csv(os.path.join(OUT_BASE, "cluster_validity_metrics.csv"), index=False)

print("\n=== CLUSTER VALIDITY METRICS ===")
print(validity_df.to_string(index=False))

# =========================================================
# LOOP OVER K
# =========================================================
for k in k_list:
    print("\n" + "=" * 70)
    print(f"RUNNING SENSITIVITY ANALYSIS FOR k = {k}")
    print("=" * 70)

    OUT_DIR = os.path.join(OUT_BASE, f"k_{k}")
    os.makedirs(OUT_DIR, exist_ok=True)

    df = df_merged.copy()
    df[cluster_col] = fcluster(Z, t=k, criterion="maxclust").astype(int).astype(str)
    df["UMAP1"] = X_umap[:, 0]
    df["UMAP2"] = X_umap[:, 1]

    cluster_order = sorted(df[cluster_col].unique(), key=lambda x: int(x))

    print("\nCluster sizes:")
    print(df[cluster_col].value_counts().sort_index().to_string())

    continuous_vars = [v for v in continuous_vars_preferred if v in df.columns]
    categorical_vars = [v for v in categorical_vars_preferred if v in df.columns]
    boxplot_vars = [v for v in boxplot_vars_preferred if v in df.columns]
    stacked_bar_vars = [v for v in stacked_bar_vars_preferred if v in df.columns]

    # -----------------------------------------------------
    # SAVE MERGED DF
    # -----------------------------------------------------
    df.to_csv(os.path.join(OUT_DIR, f"merged_embeddings_metadata_with_hc_clusters_k{k}.csv"), index=False)

    # -----------------------------------------------------
    # SUMMARY TABLES
    # -----------------------------------------------------
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
    summary_df.to_csv(os.path.join(OUT_DIR, f"cluster_summary_full_k{k}.csv"), index=False)

    paper_rows = []
    for cl in cluster_order:
        sub = df[df[cluster_col] == cl]
        row = {"cluster": cl, "N": len(sub)}

        for v in continuous_vars:
            vals = pd.to_numeric(sub[v], errors="coerce")
            row[v] = f"{vals.mean():.2f} ± {vals.std():.2f}" if vals.notna().sum() > 0 else np.nan

        paper_rows.append(row)

    paper_summary_df = pd.DataFrame(paper_rows)
    paper_summary_df.to_csv(os.path.join(OUT_DIR, f"cluster_summary_paper_format_k{k}.csv"), index=False)

    print("\nPaper summary:")
    print(paper_summary_df.to_string(index=False))

    # -----------------------------------------------------
    # CONTINUOUS TESTS
    # -----------------------------------------------------
    continuous_stats = []

    for v in continuous_vars:
        tmp = df[[cluster_col, v]].copy()
        tmp[v] = pd.to_numeric(tmp[v], errors="coerce")
        tmp = tmp.dropna()

        groups = [tmp.loc[tmp[cluster_col] == cl, v].values for cl in cluster_order]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) >= 2:
            H, p = kruskal(*groups)
            eps2 = epsilon_squared_kruskal(H, len(tmp), len(groups))

            continuous_stats.append({
                "variable": v,
                "test": "Kruskal-Wallis",
                "H": H,
                "p": p,
                "p_stars": p_to_stars(p),
                "epsilon_squared": eps2
            })

    continuous_stats_df = pd.DataFrame(continuous_stats)
    continuous_stats_df.to_csv(os.path.join(OUT_DIR, f"continuous_stats_by_cluster_k{k}.csv"), index=False)

    print("\nContinuous stats:")
    print(continuous_stats_df.to_string(index=False))

    # -----------------------------------------------------
    # CONTINUOUS POST HOC
    # -----------------------------------------------------
    continuous_posthoc_frames = []

    for v in continuous_vars:
        global_row = continuous_stats_df.loc[continuous_stats_df["variable"] == v]
        if global_row.empty:
            continue

        global_p = float(global_row["p"].iloc[0])

        if run_continuous_posthoc_only_if_global_sig and global_p >= alpha_global:
            continue

        ph = pairwise_mannwhitney_posthoc(
            df_in=df,
            cluster_col=cluster_col,
            var=v,
            cluster_order=cluster_order,
            correction_method=posthoc_correction_method,
            alpha=alpha_posthoc
        )
        continuous_posthoc_frames.append(ph)
        ph.to_csv(os.path.join(OUT_DIR, f"posthoc_continuous_{v}_k{k}.csv"), index=False)

        print(f"\nPost hoc continuous: {v}")
        print(ph.to_string(index=False))

    if continuous_posthoc_frames:
        continuous_posthoc_df = pd.concat(continuous_posthoc_frames, ignore_index=True)
    else:
        continuous_posthoc_df = pd.DataFrame(columns=[
            "variable", "cluster_1", "cluster_2", "n1", "n2",
            "median_1", "median_2", "u_stat", "p_raw",
            "rank_biserial", "p_adj", "significant", "p_adj_stars"
        ])

    continuous_posthoc_df.to_csv(
        os.path.join(OUT_DIR, f"continuous_posthoc_all_k{k}.csv"),
        index=False
    )

    # -----------------------------------------------------
    # CATEGORICAL TESTS
    # -----------------------------------------------------
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

            tab.to_csv(os.path.join(OUT_DIR, f"crosstab_{v}_by_cluster_k{k}.csv"))

    categorical_stats_df = pd.DataFrame(categorical_stats)
    categorical_stats_df.to_csv(os.path.join(OUT_DIR, f"categorical_stats_by_cluster_k{k}.csv"), index=False)

    print("\nCategorical stats:")
    if categorical_stats_df.empty:
        print("No categorical stats available.")
    else:
        print(categorical_stats_df.to_string(index=False))

    # -----------------------------------------------------
    # CATEGORICAL POST HOC
    # -----------------------------------------------------
    categorical_posthoc_frames = []

    for v in categorical_vars:
        global_row = categorical_stats_df.loc[categorical_stats_df["variable"] == v]
        if global_row.empty:
            continue

        global_p = float(global_row["p"].iloc[0])

        if run_categorical_posthoc_only_if_global_sig and global_p >= alpha_global:
            continue

        ph = pairwise_categorical_posthoc(
            df_in=df,
            cluster_col=cluster_col,
            var=v,
            cluster_order=cluster_order,
            correction_method=posthoc_correction_method,
            alpha=alpha_posthoc
        )
        categorical_posthoc_frames.append(ph)
        ph.to_csv(os.path.join(OUT_DIR, f"posthoc_categorical_{v}_k{k}.csv"), index=False)

        print(f"\nPost hoc categorical: {v}")
        print(ph.to_string(index=False))

    if categorical_posthoc_frames:
        categorical_posthoc_df = pd.concat(categorical_posthoc_frames, ignore_index=True)
    else:
        categorical_posthoc_df = pd.DataFrame(columns=[
            "variable", "cluster_1", "cluster_2", "test",
            "p_raw", "chi2_or_oddsratio", "dof", "cramers_v",
            "table_shape", "p_adj", "significant", "p_adj_stars"
        ])

    categorical_posthoc_df.to_csv(
        os.path.join(OUT_DIR, f"categorical_posthoc_all_k{k}.csv"),
        index=False
    )

    # -----------------------------------------------------
    # BOXPLOTS
    # -----------------------------------------------------
    for v in boxplot_vars:
        plt.figure(figsize=(6.5, 5.5))
        ax = sns.boxplot(data=df, x=cluster_col, y=v, order=cluster_order)
        sns.stripplot(data=df, x=cluster_col, y=v, order=cluster_order, color="black", alpha=0.6, size=4)

        counts = df[cluster_col].value_counts().sort_index()
        ymin, ymax = plt.ylim()
        y_text = ymin - 0.08 * (ymax - ymin)

        for i, cl in enumerate(cluster_order):
            ax.text(i, y_text, f"N={counts.get(cl, 0)}", ha="center", va="top", fontsize=10)

        p_row = continuous_stats_df.loc[continuous_stats_df["variable"] == v]
        if not p_row.empty:
            p_val = p_row["p"].values[0]
            stars = p_row["p_stars"].values[0]
            plt.title(f"{pretty_labels.get(v, v)} by hierarchical cluster (k={k})\nKruskal-Wallis p={p_val:.4f} ({stars})")
        else:
            plt.title(f"{pretty_labels.get(v, v)} by hierarchical cluster (k={k})")

        plt.xlabel("Hierarchical cluster")
        plt.ylabel(pretty_labels.get(v, v))

        save_and_show(os.path.join(OUT_DIR, f"boxplot_{v}_by_cluster_k{k}.png"))

    # -----------------------------------------------------
    # STACKED BARPLOTS
    # -----------------------------------------------------
    for v in stacked_bar_vars:
        tmp = df[[cluster_col, v]].copy()
        tmp[cluster_col] = tmp[cluster_col].astype(str).str.strip()
        tmp[v] = clean_categorical(tmp[v], v)
        tmp = tmp.dropna(subset=[cluster_col, v])

        if tmp.empty:
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

        if tab_counts.empty:
            continue

        tab_pct = tab_counts.div(tab_counts.sum(axis=1), axis=0) * 100

        if tab_counts.shape[0] >= 2 and tab_counts.shape[1] >= 2:
            chi2, p, dof, expected = chi2_contingency(tab_counts)
            title = f"{pretty_labels.get(v, v)} composition by hierarchical cluster (k={k})\nChi-square p={p:.4f}"
        else:
            p = np.nan
            title = f"{pretty_labels.get(v, v)} composition by hierarchical cluster (k={k})"

        ax = tab_pct.plot(kind="bar", stacked=True, figsize=(8, 5.5), width=0.8)

        cluster_counts = tab_counts.sum(axis=1)
        for i, cl in enumerate(tab_pct.index):
            ax.text(i, 102, f"N={cluster_counts.loc[cl]}", ha="center", va="bottom", fontsize=10)

        ax.set_title(title)
        ax.set_xlabel("Hierarchical cluster")
        ax.set_ylabel("Percentage")
        ax.set_ylim(0, 110)
        ax.legend(title=pretty_labels.get(v, v), bbox_to_anchor=(1.02, 1), loc="upper left")

        save_and_show(os.path.join(OUT_DIR, f"stackedbar_{v}_by_cluster_k{k}.png"))

        tab_counts.to_csv(os.path.join(OUT_DIR, f"stackedbar_counts_{v}_by_cluster_k{k}.csv"))
        tab_pct.to_csv(os.path.join(OUT_DIR, f"stackedbar_percent_{v}_by_cluster_k{k}.csv"))

    # -----------------------------------------------------
    # UMAP PLOTS
    # -----------------------------------------------------
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
    plt.title(f"UMAP of SHAP embeddings colored by hierarchical cluster (k={k})")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    save_and_show(os.path.join(OUT_DIR, f"umap_by_hierarchical_cluster_k{k}.png"))

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
    plt.title(f"UMAP of SHAP embeddings colored by {cbag_col} (k={k})")
    save_and_show(os.path.join(OUT_DIR, f"umap_by_{cbag_col}_k{k}.png"))

    if "risk_for_ad_y" in df.columns:
        df["risk_for_ad_y_label"] = clean_categorical(df["risk_for_ad_y"], "risk_for_ad_y")

        plt.figure(figsize=(7, 6))
        sns.scatterplot(
            data=df,
            x="UMAP1",
            y="UMAP2",
            hue="risk_for_ad_y_label",
            palette="Set2",
            s=70,
            alpha=0.9,
            edgecolor="k",
            linewidth=0.3
        )
        plt.title(f"UMAP of SHAP embeddings colored by risk_for_ad_y (k={k})")
        plt.legend(title="Risk", bbox_to_anchor=(1.05, 1), loc="upper left")
        save_and_show(os.path.join(OUT_DIR, f"umap_by_risk_for_ad_y_k{k}.png"))

    # -----------------------------------------------------
    # FINAL CLUSTER COUNTS
    # -----------------------------------------------------
    cluster_counts = df[cluster_col].value_counts().sort_index()
    cluster_counts.to_csv(os.path.join(OUT_DIR, f"cluster_sizes_k{k}.csv"), header=["N"])

print("\n" + "=" * 70)
print("SENSITIVITY ANALYSIS DONE")
print("Main outputs saved in:", OUT_BASE)
print("=" * 70)