#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

WORK = os.environ["WORK"]

# =========================
# Paths
# =========================
base_path = os.path.join(
    WORK, "ines/data/AD_DECODE_data6.xlsx"
)

extra_path = os.path.join(
    WORK, "ines/results/merged_data/AD_DECODE_data6_with_cBAGs_and_PCA.xlsx"
)

out_path = os.path.join(
    WORK, "ines/results/AD_DECODE_data6_merged_with_cBAG_PCA_HCmetrics.xlsx"
)

volume_norm = os.path.join(
    WORK, "ines/data/Regional_stats/ADDecode/AD_Decode_studywide_stats_for_volume_norm.csv"
)

volume = os.path.join(
    WORK, "ines/data/Regional_stats/ADDecode/AD_Decode_studywide_stats_for_volume.csv"
)

fa = os.path.join(
    WORK, "ines/data/Regional_stats/ADDecode/AD_Decode_studywide_stats_for_mrtrixfa.csv"
)

md = os.path.join(
    WORK, "ines/data/Regional_stats/ADDecode/AD_Decode_studywide_stats_for_mrtrixmd.csv"
)

# Optional QSM file: adjust if needed
qsm = os.path.join(
    WORK, "ines/data/Regional_stats/ADDecode/AD_Decode_studywide_stats_for_QSM.csv"
)

# =========================
# Cognitive columns to force-keep
# =========================
cog_cols_to_keep = [
    "Language_Composite",
    "Visuospatial_Composite",
    "Global_Cognition_Composite",
    "sex_numeric",
    "Memory_Composite_resid",
    "Executive_Function_Composite_resid",
    "Processing_Speed_Composite_resid",
    "Language_Composite_resid",
    "Visuospatial_Composite_resid",
    "Global_Cognition_Composite_resid",
]

# =========================
# Load Excel files
# =========================
df_base = pd.read_excel(base_path)
df_extra = pd.read_excel(extra_path)

for df in (df_base, df_extra):
    df["MRI_Exam_fixed"] = (
        pd.to_numeric(df["MRI_Exam"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.zfill(5)
    )

df_base = df_base.drop_duplicates(subset="MRI_Exam_fixed", keep="first")
df_extra = df_extra.drop_duplicates(subset="MRI_Exam_fixed", keep="first")

# Ensure cognitive columns are present from base
base_keep_cols = ["MRI_Exam_fixed", "MRI_Exam"] + [c for c in cog_cols_to_keep if c in df_base.columns]
df_base_keep = df_base[base_keep_cols].copy()

# Keep only genuinely new columns from extra
extra_only_cols = ["MRI_Exam_fixed"] + [c for c in df_extra.columns if c not in df_base.columns]
df_extra_subset = df_extra[extra_only_cols].copy()

# Merge base + extra
df_merged = df_base.merge(df_extra_subset, on="MRI_Exam_fixed", how="left")

# Re-append cognitive columns explicitly to be safe
for c in cog_cols_to_keep:
    if c in df_base_keep.columns and c not in df_merged.columns:
        df_merged = df_merged.merge(
            df_base_keep[["MRI_Exam_fixed", c]],
            on="MRI_Exam_fixed",
            how="left"
        )

# =========================
# Helper: extract HC metrics
# Index2 = 17 (Left Hippocampus), 53 (Right Hippocampus)
# =========================
def extract_hc_metrics(file_path, prefix, left_idx2=17, right_idx2=53):
    df = pd.read_csv(file_path)

    if "Index2" not in df.columns:
        raise ValueError(f"'Index2' column not found in {file_path}")

    subject_cols = [c for c in df.columns if str(c).startswith("S")]

    left_row = df.loc[df["Index2"] == left_idx2]
    right_row = df.loc[df["Index2"] == right_idx2]

    if left_row.empty:
        raise ValueError(f"Index2={left_idx2} not found in {file_path}")
    if right_row.empty:
        raise ValueError(f"Index2={right_idx2} not found in {file_path}")

    left_row = left_row.iloc[0]
    right_row = right_row.iloc[0]

    out = pd.DataFrame({
        "MRI_Exam_fixed": [c.replace("S", "").zfill(5) for c in subject_cols],
        f"{prefix}_L": pd.to_numeric([left_row[c] for c in subject_cols], errors="coerce"),
        f"{prefix}_R": pd.to_numeric([right_row[c] for c in subject_cols], errors="coerce"),
    })

    out[f"{prefix}_meanLR"] = out[[f"{prefix}_L", f"{prefix}_R"]].mean(axis=1)
    return out

# =========================
# Extract HC metrics
# =========================
hc_volume = extract_hc_metrics(volume, "HC_volume")
hc_volume_norm = extract_hc_metrics(volume_norm, "HC_volume_norm")
hc_fa = extract_hc_metrics(fa, "HC_FA")
hc_md = extract_hc_metrics(md, "HC_MD")

for df_metric in (hc_volume, hc_volume_norm, hc_fa, hc_md):
    df_merged = df_merged.merge(df_metric, on="MRI_Exam_fixed", how="left")

# Total brain volume from Index2 = 0
vol_df = pd.read_csv(volume)
if "Index2" in vol_df.columns:
    subject_cols = [c for c in vol_df.columns if str(c).startswith("S")]
    brain_row = vol_df.loc[vol_df["Index2"] == 0]
    if not brain_row.empty:
        brain_row = brain_row.iloc[0]
        total_brain = pd.DataFrame({
            "MRI_Exam_fixed": [c.replace("S", "").zfill(5) for c in subject_cols],
            "TotalBrain_mL": pd.to_numeric([brain_row[c] for c in subject_cols], errors="coerce")
        })
        df_merged = df_merged.merge(total_brain, on="MRI_Exam_fixed", how="left")

# Summary columns
if {"HC_volume_L", "HC_volume_R"}.issubset(df_merged.columns):
    df_merged["Hippocampus_mL"] = df_merged[["HC_volume_L", "HC_volume_R"]].mean(axis=1)

if {"HC_volume_norm_L", "HC_volume_norm_R"}.issubset(df_merged.columns):
    df_merged["Hippocampus_pct"] = df_merged[["HC_volume_norm_L", "HC_volume_norm_R"]].mean(axis=1)

if {"Hippocampus_mL", "TotalBrain_mL"}.issubset(df_merged.columns):
    df_merged["Hippocampus_pct_of_total"] = 100 * df_merged["Hippocampus_mL"] / df_merged["TotalBrain_mL"]

# =========================
# Optional QSM
# =========================
if os.path.exists(qsm):
    hc_qsm = extract_hc_metrics(qsm, "HC_QSM")
    df_merged = df_merged.merge(hc_qsm, on="MRI_Exam_fixed", how="left")
else:
    print(f"QSM file not found, skipping: {qsm}")

# =========================
# Save merged output
# =========================
df_merged.to_excel(out_path, index=False)
print("Saved merged file to:", out_path)
print("Merged shape:", df_merged.shape)

print("\nCognitive columns present in output:")
for c in cog_cols_to_keep:
    print(f" - {c}: {'YES' if c in df_merged.columns else 'NO'}")

# =========================
# Read output and check cBAG associations
# =========================
df = pd.read_excel(out_path)

# Pick cBAG column automatically
candidate_cbag_cols = [
    "cBAG_withPCA",
    "cBAG_withoutPCA",
    "Brain_Age_Gap_BiasCorrected",
    "cBAG"
]
cbag_col = None
for c in candidate_cbag_cols:
    if c in df.columns:
        cbag_col = c
        break

if cbag_col is None:
    raise ValueError("Could not find a cBAG column in merged output.")

# Variables to test against cBAG
candidate_targets = [
    "Hippocampus_mL",
    "HC_volume_meanLR",
    "Hippocampus_pct",
    "HC_volume_norm_meanLR",
    "HC_FA_meanLR",
    "HC_MD_meanLR",
    "HC_QSM_meanLR",
    "Global_Cognition_Composite",
    "Global_Cognition_Composite_resid",
]

results = []
for target in candidate_targets:
    if target not in df.columns:
        continue

    sub = df[[cbag_col, target]].dropna().copy()
    if len(sub) < 3:
        continue

    x = pd.to_numeric(sub[cbag_col], errors="coerce")
    y = pd.to_numeric(sub[target], errors="coerce")
    valid = ~(x.isna() | y.isna())
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        continue

    r_p, p_p = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)

    results.append({
        "cBAG_col": cbag_col,
        "Target": target,
        "N": len(x),
        "Pearson_r": r_p,
        "Pearson_p": p_p,
        "Spearman_rho": r_s,
        "Spearman_p": p_s
    })

results_df = pd.DataFrame(results).sort_values("Pearson_p")
results_path = os.path.join(
    WORK, "ines/results/cBAG_vs_HC_QSM_Global_associations.csv"
)
results_df.to_csv(results_path, index=False)

print("\nAssociation results saved to:", results_path)
print(results_df)