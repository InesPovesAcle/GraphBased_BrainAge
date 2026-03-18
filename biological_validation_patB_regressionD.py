#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:36:58 2026

@author: ines
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

WORK = os.environ["WORK"]

# =========================================================
# PATHS
# =========================================================
out_path = os.path.join(
    WORK, "ines/results/AD_DECODE_data6_merged_with_cBAG_PCA_HCmetrics.xlsx"
)

results_dir = os.path.join(
    WORK, "ines/results/cBAG_and_age_associations_reducedFDR"
)
os.makedirs(results_dir, exist_ok=True)

plots_dir = os.path.join(results_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# =========================================================
# SETTINGS
# =========================================================
cbag_cols = [
    "cBAG_withPCA",
    "cBAG_withoutPCA",
]

age_col = "age"

cognition_vars = [
    "Memory_Composite",
    "Executive_Function_Composite",
    "Processing_Speed_Composite",
    "Language_Composite",
    "Visuospatial_Composite",
    "Global_Cognition_Composite",
]

imaging_vars = [
    "HC_volume_meanLR",
    "HC_volume_norm_meanLR",
    "HC_FA_meanLR",
    "HC_MD_meanLR",
    "HC_QSM_meanLR",
    "TotalBrain_mL",
]

target_vars = cognition_vars + imaging_vars

winsorize = False
winsor_limits = (0.01, 0.99)

# =========================================================
# HELPERS
# =========================================================
def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def maybe_winsorize(series, q_low=0.01, q_high=0.99):
    lo = series.quantile(q_low)
    hi = series.quantile(q_high)
    return series.clip(lower=lo, upper=hi)

def fit_linear_model(x, y):
    X = sm.add_constant(x)
    return sm.OLS(y, X).fit()

def extract_regression_stats(df_sub, x_col, y_col):
    x = safe_numeric(df_sub[x_col])
    y = safe_numeric(df_sub[y_col])

    if winsorize:
        x = maybe_winsorize(x, winsor_limits[0], winsor_limits[1])
        y = maybe_winsorize(y, winsor_limits[0], winsor_limits[1])

    valid = ~(x.isna() | y.isna())
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return None

    model = fit_linear_model(x, y)

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)

    slope = model.params[x_col]
    intercept = model.params["const"]

    slope_ci_low, slope_ci_high = model.conf_int().loc[x_col]
    intercept_ci_low, intercept_ci_high = model.conf_int().loc["const"]

    stats_row = {
        "Predictor": x_col,
        "Outcome": y_col,
        "N": int(len(x)),
        "Pearson_r": pearson_r,
        "Pearson_p": pearson_p,
        "Spearman_rho": spearman_rho,
        "Spearman_p": spearman_p,
        "R2": model.rsquared,
        "Adj_R2": model.rsquared_adj,
        "Beta": slope,
        "Beta_p": model.pvalues[x_col],
        "Beta_CI_low": slope_ci_low,
        "Beta_CI_high": slope_ci_high,
        "Intercept": intercept,
        "Intercept_CI_low": intercept_ci_low,
        "Intercept_CI_high": intercept_ci_high,
    }

    df_used = pd.DataFrame({
        x_col: x.values,
        y_col: y.values
    })

    return stats_row, df_used, model

def make_scatterplot(df_used, x_col, y_col, stats_row, out_png):
    plt.figure(figsize=(8, 6))

    sns.regplot(
        data=df_used,
        x=x_col,
        y=y_col,
        ci=95,
        scatter_kws={"s": 90, "alpha": 0.75, "edgecolor": "black"},
        line_kws={"linewidth": 3}
    )

    if "bag" in y_col.lower():
        plt.axhline(0, linestyle="--", color="gray", linewidth=1.5)

    textstr = (
        f"n = {stats_row['N']}\n"
        f"r = {stats_row['Pearson_r']:.3f}\n"
        f"p = {stats_row['Pearson_p']:.3g}\n"
        f"R² = {stats_row['R2']:.3f}\n"
        f"β = {stats_row['Beta']:.3f}"
    )

    plt.text(
        0.98, 0.02,
        textstr,
        transform=plt.gca().transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9)
    )

    plt.title(f"{y_col} vs {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def add_domain_labels(stats_df):
    stats_df = stats_df.copy()
    stats_df["Domain"] = np.where(
        stats_df["Outcome"].isin(cognition_vars),
        "Cognition",
        np.where(stats_df["Outcome"].isin(imaging_vars), "Imaging", "Other")
    )
    return stats_df

def apply_fdr_by_domain(stats_df, p_col, predictor_col="Predictor", domain_col="Domain"):
    stats_df = stats_df.copy()
    stats_df[f"{p_col}_FDR"] = np.nan
    stats_df[f"{p_col}_FDR_reject"] = False

    for predictor in stats_df[predictor_col].dropna().unique():
        for domain in ["Cognition", "Imaging"]:
            mask = (stats_df[predictor_col] == predictor) & (stats_df[domain_col] == domain)
            sub = stats_df.loc[mask].copy()

            if sub.empty:
                continue

            pvals = sub[p_col].astype(float).values
            reject, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

            stats_df.loc[mask, f"{p_col}_FDR"] = pvals_fdr
            stats_df.loc[mask, f"{p_col}_FDR_reject"] = reject

    return stats_df

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_excel(out_path)

print("Loaded:", out_path)
print("Shape:", df.shape)

available_cbags = [c for c in cbag_cols if c in df.columns]
if len(available_cbags) == 0:
    raise ValueError(f"None of the requested cBAG columns were found: {cbag_cols}")

if age_col not in df.columns:
    raise ValueError(f"Age column '{age_col}' not found in file.")

available_targets = [v for v in target_vars if v in df.columns]
missing_targets = [v for v in target_vars if v not in df.columns]

print("\nAvailable cBAG columns:")
for c in available_cbags:
    print(" -", c)

print("\nAvailable targets:")
for v in available_targets:
    print(" -", v)

if missing_targets:
    print("\nMissing targets:")
    for v in missing_targets:
        print(" -", v)

# =========================================================
# 1) cBAG ASSOCIATIONS
# =========================================================
cbag_results_dir = os.path.join(results_dir, "cBAG_associations")
os.makedirs(cbag_results_dir, exist_ok=True)

all_cbag_stats = []

for cbag_col in available_cbags:
    sub_results_dir = os.path.join(cbag_results_dir, cbag_col)
    os.makedirs(sub_results_dir, exist_ok=True)

    sub_plots_dir = os.path.join(sub_results_dir, "plots")
    os.makedirs(sub_plots_dir, exist_ok=True)

    for x_col in available_targets:
        df_sub = df[[x_col, cbag_col]].copy()
        result = extract_regression_stats(df_sub, x_col=x_col, y_col=cbag_col)

        if result is None:
            print(f"Skipping {cbag_col} ~ {x_col}: too few usable rows")
            continue

        stats_row, df_used, model = result
        all_cbag_stats.append(stats_row)

        used_csv = os.path.join(sub_results_dir, f"used_data_{cbag_col}_vs_{x_col}.csv")
        df_used.to_csv(used_csv, index=False)

        plot_path = os.path.join(sub_plots_dir, f"{cbag_col}_vs_{x_col}.png")
        make_scatterplot(df_used, x_col, cbag_col, stats_row, plot_path)

        summary_path = os.path.join(sub_results_dir, f"model_summary_{cbag_col}_vs_{x_col}.txt")
        with open(summary_path, "w") as f:
            f.write(model.summary().as_text())

        print(f"Saved: {cbag_col} ~ {x_col}")

cbag_stats_df = pd.DataFrame(all_cbag_stats)
cbag_stats_df = add_domain_labels(cbag_stats_df)

# Keep original columns:
# Predictor = metric
# Outcome = cBAG model
cbag_stats_df = cbag_stats_df.rename(columns={
    "Predictor": "Metric",
    "Outcome": "cBAG_Model"
})

# Create helper column only for FDR grouping
cbag_stats_df["FDR_Group"] = cbag_stats_df["cBAG_Model"]

# FDR separately within each cBAG model and domain
cbag_stats_df = apply_fdr_by_domain(
    cbag_stats_df,
    "Pearson_p",
    predictor_col="FDR_Group",
    domain_col="Domain"
)

cbag_stats_df = apply_fdr_by_domain(
    cbag_stats_df,
    "Beta_p",
    predictor_col="FDR_Group",
    domain_col="Domain"
)

cbag_stats_df = cbag_stats_df.drop(columns=["FDR_Group"])
cbag_stats_df = cbag_stats_df.sort_values(["cBAG_Model", "Domain", "Pearson_p"])

cbag_stats_csv = os.path.join(cbag_results_dir, "cBAG_association_stats_all.csv")
cbag_stats_xlsx = os.path.join(cbag_results_dir, "cBAG_association_stats_all.xlsx")
cbag_stats_df.to_csv(cbag_stats_csv, index=False)
cbag_stats_df.to_excel(cbag_stats_xlsx, index=False)

# =========================================================
# 1b) SAVE FDR STATS SEPARATELY FOR withPCA / withoutPCA
# =========================================================
for cbag_model in cbag_stats_df["cBAG_Model"].unique():
    sub = cbag_stats_df[cbag_stats_df["cBAG_Model"] == cbag_model].copy()

    if sub.empty:
        continue

    sub_csv = os.path.join(
        cbag_results_dir,
        f"FDR_stats_{cbag_model}.csv"
    )
    sub_xlsx = os.path.join(
        cbag_results_dir,
        f"FDR_stats_{cbag_model}.xlsx"
    )

    sub.to_csv(sub_csv, index=False)
    sub.to_excel(sub_xlsx, index=False)

    print("Saved:", sub_csv)
    print("Saved:", sub_xlsx)

# Domain-specific FDR tables too
for cbag_model in cbag_stats_df["cBAG_Model"].unique():
    for domain in ["Cognition", "Imaging"]:
        sub = cbag_stats_df[
            (cbag_stats_df["cBAG_Model"] == cbag_model) &
            (cbag_stats_df["Domain"] == domain)
        ].copy()

        if sub.empty:
            continue

        sub_csv = os.path.join(
            cbag_results_dir,
            f"FDR_stats_{cbag_model}_{domain.lower()}.csv"
        )
        sub_xlsx = os.path.join(
            cbag_results_dir,
            f"FDR_stats_{cbag_model}_{domain.lower()}.xlsx"
        )

        sub.to_csv(sub_csv, index=False)
        sub.to_excel(sub_xlsx, index=False)

        print("Saved:", sub_csv)
        print("Saved:", sub_xlsx)

# =========================================================
# 2) AGE ASSOCIATIONS
# =========================================================
age_results_dir = os.path.join(results_dir, "Age_associations")
os.makedirs(age_results_dir, exist_ok=True)

age_plots_dir = os.path.join(age_results_dir, "plots")
os.makedirs(age_plots_dir, exist_ok=True)

all_age_stats = []

for y_col in available_targets:
    df_sub = df[[age_col, y_col]].copy()
    result = extract_regression_stats(df_sub, x_col=age_col, y_col=y_col)

    if result is None:
        print(f"Skipping {y_col} ~ {age_col}: too few usable rows")
        continue

    stats_row, df_used, model = result
    all_age_stats.append(stats_row)

    used_csv = os.path.join(age_results_dir, f"used_data_{y_col}_vs_{age_col}.csv")
    df_used.to_csv(used_csv, index=False)

    plot_path = os.path.join(age_plots_dir, f"{y_col}_vs_{age_col}.png")
    make_scatterplot(df_used, age_col, y_col, stats_row, plot_path)

    summary_path = os.path.join(age_results_dir, f"model_summary_{y_col}_vs_{age_col}.txt")
    with open(summary_path, "w") as f:
        f.write(model.summary().as_text())

    print(f"Saved: {y_col} ~ {age_col}")

age_stats_df = pd.DataFrame(all_age_stats)
age_stats_df = add_domain_labels(age_stats_df)

# For age analyses, FDR separately by domain with Age as common predictor
age_stats_df["PredictorLabel"] = "Age"
age_stats_df = apply_fdr_by_domain(age_stats_df, "Pearson_p", predictor_col="PredictorLabel", domain_col="Domain")
age_stats_df = apply_fdr_by_domain(age_stats_df, "Beta_p", predictor_col="PredictorLabel", domain_col="Domain")
age_stats_df = age_stats_df.drop(columns=["PredictorLabel"])

age_stats_df = age_stats_df.sort_values(["Domain", "Pearson_p"])

age_stats_csv = os.path.join(age_results_dir, "Age_association_stats_all.csv")
age_stats_xlsx = os.path.join(age_results_dir, "Age_association_stats_all.xlsx")
age_stats_df.to_csv(age_stats_csv, index=False)
age_stats_df.to_excel(age_stats_xlsx, index=False)

# =========================================================
# 3) DOMAIN-SPECIFIC OUTPUTS
# =========================================================
for domain in ["Cognition", "Imaging"]:
    sub = age_stats_df[age_stats_df["Domain"] == domain].copy()
    if not sub.empty:
        sub.to_csv(os.path.join(age_results_dir, f"Age_association_stats_{domain.lower()}.csv"), index=False)
        sub.to_excel(os.path.join(age_results_dir, f"Age_association_stats_{domain.lower()}.xlsx"), index=False)

for cbag_model in cbag_stats_df["cBAG_Model"].unique():
    for domain in ["Cognition", "Imaging"]:
        sub = cbag_stats_df[
            (cbag_stats_df["cBAG_Model"] == cbag_model) &
            (cbag_stats_df["Domain"] == domain)
        ].copy()
        if not sub.empty:
            sub.to_csv(
                os.path.join(cbag_results_dir, f"cBAG_association_stats_{cbag_model}_{domain.lower()}.csv"),
                index=False
            )
            sub.to_excel(
                os.path.join(cbag_results_dir, f"cBAG_association_stats_{cbag_model}_{domain.lower()}.xlsx"),
                index=False
            )

# =========================================================
# 4) COMPACT TABLES
# =========================================================
cbag_compact_cols = [
    "cBAG_Model", "Domain", "Metric", "N",
    "Pearson_r", "Pearson_p", "Pearson_p_FDR", "Pearson_p_FDR_reject",
    "R2", "Beta", "Beta_p", "Beta_p_FDR", "Beta_p_FDR_reject",
    "Beta_CI_low", "Beta_CI_high"
]
cbag_compact_df = cbag_stats_df[cbag_compact_cols].copy()
cbag_compact_df.to_csv(os.path.join(cbag_results_dir, "cBAG_association_stats_compact.csv"), index=False)
cbag_compact_df.to_excel(os.path.join(cbag_results_dir, "cBAG_association_stats_compact.xlsx"), index=False)

age_compact_cols = [
    "Predictor", "Domain", "Outcome", "N",
    "Pearson_r", "Pearson_p", "Pearson_p_FDR", "Pearson_p_FDR_reject",
    "R2", "Beta", "Beta_p", "Beta_p_FDR", "Beta_p_FDR_reject",
    "Beta_CI_low", "Beta_CI_high"
]
age_compact_df = age_stats_df[age_compact_cols].copy()
age_compact_df.to_csv(os.path.join(age_results_dir, "Age_association_stats_compact.csv"), index=False)
age_compact_df.to_excel(os.path.join(age_results_dir, "Age_association_stats_compact.xlsx"), index=False)

print("\nSaved cBAG stats to:")
print(" -", cbag_stats_csv)
print(" -", cbag_stats_xlsx)

print("\nSaved Age stats to:")
print(" -", age_stats_csv)
print(" -", age_stats_xlsx)

print("\nDone.")