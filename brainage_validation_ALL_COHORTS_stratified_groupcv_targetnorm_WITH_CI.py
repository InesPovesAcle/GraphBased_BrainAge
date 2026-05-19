#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brain-age validation / figure generation for ALL cohorts and ALL ablation feature sets.

Designed for the new training output structure:

$WORK/ines/results/BrainAgePredictionADNI/ablation_imaging_only/
$WORK/ines/results/BrainAgePredictionADNI/ablation_imaging_demographics/
$WORK/ines/results/BrainAgePredictionADNI/ablation_imaging_biomarkers/
$WORK/ines/results/BrainAgePredictionADNI/ablation_full/
$WORK/ines/results/BrainAgePredictionADNI/ablation_full_no_cardiovascular/

and the same pattern for ADRC, HABS, and AD_DECODE.

What it generates
-----------------
Per cohort + feature_set:
  - histogram of BAG / cBAG
  - predicted age vs chronological age
  - BAG / cBAG vs chronological age
  - correlations with available cognition, imaging, vascular, metabolic, and biomarker variables
  - boxplots by diagnostic / cognitive group when available
  - ROC curves for APOE4 carriage and cognitive status when available
  - subject-level validation table
  - logs and correlation stats

Across cohorts + feature sets:
  - combined CV summary table
  - barplots comparing MAE / RMSE / R2 / r by cohort and feature_set
  - combined validation summary table

Run:
  python brainage_validation_all_cohorts_ablation_figures.py

Edit only USER CONFIG below if needed.
"""

import os
import re
import glob
import json
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
matplotlib.set_loglevel("error")
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr, linregress, ttest_ind, f_oneway, mannwhitneyu, kruskal
from sklearn.metrics import roc_auc_score, roc_curve, mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# =========================================================
# USER CONFIG
# =========================================================
WORK = os.environ.get("WORK", "/mnt/newStor/paros/paros_WORK")
RESULTS_ROOT = os.path.join(WORK, "ines/results")

COHORTS_TO_RUN = ["ADNI", "ADRC", "HABS", "AD_DECODE"]
FEATURE_SETS_TO_RUN = [
    "imaging_only",
    "imaging_demographics",
    "imaging_biomarkers",
    "full",
    "full_no_cardiovascular",
]

# Which prediction residual to validate.
# The script will use the first available column in this order.
PREFERRED_BRAIN_METRICS = ["cBAG_global", "cBAG", "BAG", "BAG_raw"]
CORR_METHOD = "pearson"  # "pearson" or "spearman"
CLEAR_OLD_FIGURES = True

# New-training evaluation settings.
# Use OOF predictions for validation figures so predicted-age plots match CV metrics.
PREFER_OOF_FOR_VALIDATION = True

# Save this evaluation in a separate folder so baseline outputs are not overwritten.
COMBINED_VALIDATION_DIR_NAME = "BrainAgeValidation_AllCohorts_stratified_groupcv_targetnorm"

# HABS optional clinical file.
HABS_CLINICAL_PATH = os.path.join(
    WORK,
    "ines/data/harmonization/HABS/metadata/RP_HD_7_Clinical.xlsx"
)

RESULTS_DIR_MAP = {
    "ADNI": "BrainAgePredictionADNI_stratified_groupcv_targetnorm",
    "ADRC": "BrainAgePredictionADRC_stratified_groupcv_targetnorm",
    "HABS": "BrainAgePredictionHABS_stratified_groupcv_targetnorm",
    "AD_DECODE": "BrainAgePredictionADDECODE_stratified_groupcv_targetnorm",
}

PREFIX_MAP = {
    "ADNI": "adni",
    "ADRC": "adrc",
    "HABS": "habs",
    "AD_DECODE": "addecode",
}

GRAPH_BUILDER_PREFIX_MAP = {
    "ADNI": "adni",
    "ADRC": "adrc",
    "HABS": "habs",
    "AD_DECODE": "ad_decode",
}

SENTINEL_VALUES = {
    -999999, -888888, -777777,
    -99999, -88888, -77777,
    -9999, -8888, -7777,
    -999, -888, -777,
    999, 888, 777,
    9999, 8888, 7777,
    99999, 88888, 77777,
    999999, 888888, 777777,
}


# =========================================================
# BASIC HELPERS
# =========================================================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def find_existing_file(candidates: List[str]) -> Optional[str]:
    for fp in candidates:
        if fp and os.path.exists(fp):
            return fp
    return None


def load_table_auto(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


def save_table_both(df: pd.DataFrame, csv_path: str, xlsx_path: Optional[str] = None):
    df.to_csv(csv_path, index=False)
    if xlsx_path is not None:
        try:
            df.to_excel(xlsx_path, index=False)
        except Exception as e:
            print(f"Could not save Excel {xlsx_path}: {e}")


def sanitize_filename(name) -> str:
    safe = str(name)
    replacements = {
        " ": "_", "/": "_", "\\": "_", "(": "", ")": "",
        "[": "", "]": "", ":": "_", ";": "_", ",": "_",
        "<": "_", ">": "_", "=": "eq", "*": "x",
    }
    for old, new in replacements.items():
        safe = safe.replace(old, new)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")[:180]


def normalize_id_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s


def clean_numeric_with_sentinels(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").copy()
    for val in SENTINEL_VALUES:
        s = s.mask(s == val, np.nan)
    return s


def zscore_series(series: pd.Series) -> pd.Series:
    s = clean_numeric_with_sentinels(series)
    valid = s.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=s.index, dtype=float)
    std = valid.std(ddof=1)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=s.index, dtype=float)
    return (s - valid.mean()) / std


def first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def unique_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def clear_image_files(folder: str):
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.svg", "*.pdf"]
    for pat in patterns:
        for fp in glob.glob(os.path.join(folder, pat)):
            try:
                os.remove(fp)
            except Exception:
                pass


def format_p_value(p):
    if pd.isna(p):
        return "nan"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


# =========================================================
# PATH HELPERS FOR NEW OUTPUT STRUCTURE
# =========================================================
def get_paths(cohort: str, feature_set: str) -> Dict[str, str]:
    if cohort not in RESULTS_DIR_MAP:
        raise ValueError(f"Unsupported cohort: {cohort}")
    base_results_dir = os.path.join(RESULTS_ROOT, RESULTS_DIR_MAP[cohort])
    ablation_dir = os.path.join(base_results_dir, f"ablation_{feature_set}")

    train_prefix = f"{PREFIX_MAP[cohort]}_{feature_set}"
    gb_prefix = GRAPH_BUILDER_PREFIX_MAP[cohort]

    harmonized_graph_dir = os.path.join(
        RESULTS_ROOT,
        "harmonized",
        cohort,
        "graphs",
        feature_set,
    )

    val_outdir = os.path.join(ablation_dir, "validation_figures")
    ensure_dir(val_outdir)

    return {
        "base_results_dir": base_results_dir,
        "ablation_dir": ablation_dir,
        "train_prefix": train_prefix,
        "gb_prefix": gb_prefix,
        "harmonized_graph_dir": harmonized_graph_dir,
        "val_outdir": val_outdir,

        "oof_csv": os.path.join(ablation_dir, f"{train_prefix}_cv_oof_predictions.csv"),
        "oof_xlsx": os.path.join(ablation_dir, f"{train_prefix}_cv_oof_predictions.xlsx"),
        "full_pred_csv": os.path.join(ablation_dir, f"{train_prefix}_full_cohort_predictions.csv"),
        "full_pred_xlsx": os.path.join(ablation_dir, f"{train_prefix}_full_cohort_predictions.xlsx"),
        "metadata_cv_csv": os.path.join(ablation_dir, f"{train_prefix}_metadata_with_cv_predictions.csv"),
        "metadata_cv_xlsx": os.path.join(ablation_dir, f"{train_prefix}_metadata_with_cv_predictions.xlsx"),
        "metadata_all_pred_csv": os.path.join(ablation_dir, f"{train_prefix}_metadata_all_with_predictions.csv"),
        "metadata_all_pred_xlsx": os.path.join(ablation_dir, f"{train_prefix}_metadata_all_with_predictions.xlsx"),
        "cv_summary_csv": os.path.join(ablation_dir, f"{train_prefix}_cv_summary_metrics.csv"),
        "cv_summary_xlsx": os.path.join(ablation_dir, f"{train_prefix}_cv_summary_metrics.xlsx"),
        "master_xlsx": os.path.join(ablation_dir, f"{train_prefix}_master_results.xlsx"),

        "cv_fold_raw_csv": os.path.join(ablation_dir, f"{train_prefix}_cv_fold_metrics_raw.csv"),
        "cv_fold_raw_xlsx": os.path.join(ablation_dir, f"{train_prefix}_cv_fold_metrics_raw.xlsx"),
        "cv_fold_bc_csv": os.path.join(ablation_dir, f"{train_prefix}_cv_fold_metrics_bias_corrected.csv"),
        "cv_fold_bc_xlsx": os.path.join(ablation_dir, f"{train_prefix}_cv_fold_metrics_bias_corrected.xlsx"),

        "bootstrap_summary_csv": os.path.join(ablation_dir, f"{train_prefix}_bootstrap_metric_summary.csv"),
        "bootstrap_summary_xlsx": os.path.join(ablation_dir, f"{train_prefix}_bootstrap_metric_summary.xlsx"),

        "metadata_aligned_csv": os.path.join(harmonized_graph_dir, f"{gb_prefix}_metadata_aligned.csv"),
        "metadata_aligned_raw_csv": os.path.join(harmonized_graph_dir, f"{gb_prefix}_metadata_aligned_raw.csv"),
        "metadata_all_aligned_csv": os.path.join(harmonized_graph_dir, f"{gb_prefix}_metadata_all_aligned.csv"),
        "metadata_all_aligned_raw_csv": os.path.join(harmonized_graph_dir, f"{gb_prefix}_metadata_all_aligned_raw.csv"),
    }


def discover_inputs(cohort: str, feature_set: str) -> Dict[str, Optional[str]]:
    p = get_paths(cohort, feature_set)

    oof_path = find_existing_file([p["oof_csv"], p["oof_xlsx"]])
    full_pred_path = find_existing_file([p["full_pred_csv"], p["full_pred_xlsx"]])

    metadata_path = find_existing_file([
        p["metadata_cv_csv"],
        p["metadata_cv_xlsx"],
        p["metadata_aligned_raw_csv"],
        p["metadata_aligned_csv"],
    ])

    metadata_all_path = find_existing_file([
        p["metadata_all_pred_csv"],
        p["metadata_all_pred_xlsx"],
        p["metadata_all_aligned_raw_csv"],
        p["metadata_all_aligned_csv"],
    ])

    cv_summary_path = find_existing_file([p["cv_summary_csv"], p["cv_summary_xlsx"]])
    bootstrap_summary_path = find_existing_file([p["bootstrap_summary_csv"], p["bootstrap_summary_xlsx"]])

    return {
        **p,
        "oof_path": oof_path,
        "full_pred_path": full_pred_path,
        "metadata_path": metadata_path,
        "metadata_all_path": metadata_all_path,
        "cv_summary_path": cv_summary_path,
        "bootstrap_summary_path": bootstrap_summary_path,
    }


# =========================================================
# DATA NORMALIZATION / MERGE HELPERS
# =========================================================
def normalize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}
    aliases = {
        "Real_Age": ["Real_Age", "Age", "age", "VISIT_AGE", "AGE", "age_true"],
        "Predicted_Age_RAW": ["Predicted_Age_RAW", "Predicted_Age_raw", "Pred_raw", "PredictedAgeRaw", "y_pred_raw", "pred_raw"],
        "Predicted_Age_BiasCorrected": ["Predicted_Age_BiasCorrected", "Predicted_Age_corrected", "Pred_corr", "Pred_corr_foldwise", "y_pred_corrected", "pred_bias_corrected"],
        "Predicted_Age_GlobalCorrected": ["Predicted_Age_GlobalCorrected", "Pred_corr_global", "Predicted_Age_corrected_global", "y_pred_global_corrected", "pred_global_corrected"],
        "BAG": ["BAG", "bag", "BAG_raw", "bag_raw"],
        "cBAG": ["cBAG", "cbag"],
        "cBAG_global": ["cBAG_global", "cbag_global"],
    }
    for target, possible_names in aliases.items():
        if target in df.columns:
            continue
        for old in possible_names:
            if old in df.columns:
                rename_map[old] = target
                break
    df = df.rename(columns=rename_map)

    # Derive common residual names if possible.
    if "BAG" not in df.columns and {"Predicted_Age_RAW", "Real_Age"}.issubset(df.columns):
        df["BAG"] = clean_numeric_with_sentinels(df["Predicted_Age_RAW"]) - clean_numeric_with_sentinels(df["Real_Age"])
    if "cBAG" not in df.columns and {"Predicted_Age_BiasCorrected", "Real_Age"}.issubset(df.columns):
        df["cBAG"] = clean_numeric_with_sentinels(df["Predicted_Age_BiasCorrected"]) - clean_numeric_with_sentinels(df["Real_Age"])
    if "cBAG_global" not in df.columns and {"Predicted_Age_GlobalCorrected", "Real_Age"}.issubset(df.columns):
        df["cBAG_global"] = clean_numeric_with_sentinels(df["Predicted_Age_GlobalCorrected"]) - clean_numeric_with_sentinels(df["Real_Age"])

    return df


def ensure_subject_id_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Subject_ID" not in df.columns:
        candidates = [
            "graph_id", "connectome_key", "connectome_id", "subject_id", "match_id",
            "PTID", "ptid", "RID", "ID", "MRI_Exam", "regional_id", "runno", "Subject",
        ]
        for c in candidates:
            if c in df.columns:
                df["Subject_ID"] = df[c].astype(str)
                break
    if "Subject_ID" not in df.columns:
        raise KeyError("No usable subject identifier found.")
    df["Subject_ID"] = normalize_id_series(df["Subject_ID"])
    return df


def find_best_metadata_merge_key(metadata_df: pd.DataFrame, graph_ids: List[str]) -> Tuple[Optional[str], int]:
    candidate_cols = [
        "connectome_key", "graph_id", "Subject_ID", "match_id", "subject_id", "PTID",
        "ptid", "regional_id", "RID", "MRI_Exam", "runno", "Subject", "ID",
    ]
    graph_id_set = set(normalize_id_series(pd.Series(graph_ids)).tolist())
    best_col, best_matches = None, -1
    for col in candidate_cols:
        if col in metadata_df.columns:
            meta_vals = set(normalize_id_series(metadata_df[col]).tolist())
            overlap = len(graph_id_set.intersection(meta_vals))
            if overlap > best_matches:
                best_col = col
                best_matches = overlap
    return best_col, best_matches


def coalesce_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    meta_cols = [c for c in df.columns if c.endswith("_meta")]
    for meta_col in meta_cols:
        base_col = meta_col[:-5]
        if base_col in df.columns:
            df[base_col] = df[base_col].combine_first(df[meta_col])
        else:
            df[base_col] = df[meta_col]
    return df


def merge_predictions_and_metadata(pred_df: pd.DataFrame, metadata_df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Optional[str], int]:
    pred_df = normalize_prediction_columns(ensure_subject_id_col(pred_df))
    if metadata_df is None:
        return pred_df, None, 0

    metadata_df = normalize_prediction_columns(ensure_subject_id_col(metadata_df))
    for col in metadata_df.columns:
        if col in ["connectome_key", "graph_id", "Subject_ID", "match_id", "subject_id", "PTID", "ptid", "regional_id", "RID", "MRI_Exam", "runno", "Subject", "ID"]:
            metadata_df[col] = normalize_id_series(metadata_df[col])

    merge_key, overlap = find_best_metadata_merge_key(metadata_df, pred_df["Subject_ID"].astype(str).tolist())
    if merge_key is None or overlap <= 0:
        return pred_df, merge_key, overlap

    meta_small = metadata_df.drop_duplicates(subset=[merge_key]).copy()
    keep_cols = [merge_key] + [c for c in meta_small.columns if c != merge_key and c not in pred_df.columns]
    meta_small = meta_small[keep_cols]

    merged = pred_df.merge(meta_small, left_on="Subject_ID", right_on=merge_key, how="left", suffixes=("", "_meta"))
    merged = coalesce_meta_columns(merged)
    return merged, merge_key, overlap


# =========================================================
# VARIABLE SELECTION
# =========================================================
def choose_brain_metric(df: pd.DataFrame) -> str:
    for col in PREFERRED_BRAIN_METRICS:
        if col in df.columns and clean_numeric_with_sentinels(df[col]).notna().sum() >= 3:
            return col
    raise KeyError(f"No usable brain metric found. Tried: {PREFERRED_BRAIN_METRICS}")


def find_existing_columns_by_patterns(df: pd.DataFrame, patterns: List[str]) -> List[str]:
    cols_found = []
    for col in df.columns:
        low = str(col).strip().lower()
        for pat in patterns:
            if pat.lower() in low:
                cols_found.append(col)
                break
    return cols_found


def get_candidate_validation_vars(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    candidate_cognition_vars = [
        "MMSE_total", "MOCA_total_corrected", "MOCA_total", "ADAS_total", "CDGLOBAL", "CDRSB",
        "cognition_composite", "Memory_Composite", "Executive_Function_Composite",
        "Processing_Speed_Composite", "Language_Composite", "Visuospatial_Composite",
        "Global_Cognition_Composite", "Memory_Composite_resid", "Executive_Function_Composite_resid",
        "Processing_Speed_Composite_resid", "Language_Composite_resid", "Visuospatial_Composite_resid",
        "Global_Cognition_Composite_resid",
    ]
    candidate_imaging_vars = [
        "Clustering_Coeff", "Path_Length", "Global_Efficiency", "Local_Efficiency",
        "FA_mean", "FA_median", "Volume_mean", "Volume_median",
        "Hippocampus_Total_pct", "Left_Hippocampus_pct", "Right_Hippocampus_pct",
        "Hippocampus_FA_Mean", "Hippocampus_FA_Total", "Left_Hippocampus_FA", "Right_Hippocampus_FA",
        "Total_Brain_volume", "ABETA42", "ABETA40", "TAU", "PTAU", "PLASMA_PTAU217",
        "GFAP", "NfL", "amyloid_42", "amyloid_40", "tau_total", "ptau", "ptau217", "gfap", "nfl",
        "BMI", "bmi", "OM_BMI", "VSBPSYS", "VSBPDIA", "VSPULSE", "BPSYS_AVG", "BPDIA_AVG",
        "bp_sys", "bp_dia", "pulse", "pulse_pressure", "MAP", "Systolic", "Diastolic", "Pulse",
        "BW_Glucose_y", "Glucose", "glucose", "fasting_glucose", "BW_HBA1c_y", "HbA1c", "hba1c",
        "BW_CholTotal_y", "BW_HDLChol_y", "BW_LDLchol_y", "BW_Triglycerides_y",
        "CholTotal", "HDL", "LDL", "Triglycerides", "chol_total", "hdl", "ldl", "triglycerides",
        "ATN_composite", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10",
    ]
    auto_patterns = [
        "clustering", "path_length", "path length", "efficiency", "hippocampus", "fa", "volume",
        "amyloid", "abeta", "tau", "ptau", "gfap", "nfl", "glucose", "hba1c", "chol", "ldl",
        "hdl", "triglycer", "systolic", "diastolic", "pulse", "map", "blood_pressure", "bp_", "bmi",
    ]
    candidate_imaging_vars = unique_preserve_order(candidate_imaging_vars + find_existing_columns_by_patterns(df, auto_patterns))

    cognition_vars = [c for c in candidate_cognition_vars if c in df.columns and clean_numeric_with_sentinels(df[c]).notna().sum() >= 3]
    imaging_vars = [c for c in candidate_imaging_vars if c in df.columns and clean_numeric_with_sentinels(df[c]).notna().sum() >= 3]
    return cognition_vars, imaging_vars


def add_clean_and_z_versions(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        df[f"{c}_raw_clean"] = clean_numeric_with_sentinels(df[c])
        df[f"{c}_z_clean"] = zscore_series(df[c])
    return df


def find_group_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "Research Group", "Diagnosis", "DX", "DX_bl", "Group", "group_status",
        "DEMENTED", "Diagnostic_Group", "NORMCOG", "Risk", "Risk_y", "risk_for_ad",
    ]
    for gc in candidates:
        if gc in df.columns:
            s = df[gc].dropna()
            if s.nunique() >= 2:
                return gc
    return None


# =========================================================
# METRICS / PLOTS
# =========================================================
def compute_scatter_metrics(x, y, corr_method="pearson", use_identity_r2=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"n": len(x), "r": np.nan, "p": np.nan, "r2": np.nan, "mae": np.nan, "rmse": np.nan, "slope": np.nan, "intercept": np.nan}
    if corr_method.lower() == "spearman":
        r, p = spearmanr(x, y)
    else:
        r, p = pearsonr(x, y)
    lr = linregress(x, y)
    return {
        "n": int(len(x)), "r": float(r), "p": float(p),
        "r2": float(r2_score(x, y) if use_identity_r2 else r ** 2),
        "mae": float(mean_absolute_error(x, y)),
        "rmse": float(np.sqrt(mean_squared_error(x, y))),
        "slope": float(lr.slope), "intercept": float(lr.intercept),
    }


def valid_xy(df: pd.DataFrame, x_col: str, y_col: str):
    if x_col not in df.columns or y_col not in df.columns:
        return False, f"missing column: {x_col if x_col not in df.columns else y_col}"
    tmp = df[[x_col, y_col]].copy()
    tmp[x_col] = clean_numeric_with_sentinels(tmp[x_col])
    tmp[y_col] = clean_numeric_with_sentinels(tmp[y_col])
    tmp = tmp.dropna()
    if len(tmp) < 3:
        return False, "fewer than 3 complete rows"
    if tmp[x_col].nunique(dropna=True) < 2:
        return False, f"{x_col} is constant"
    if tmp[y_col].nunique(dropna=True) < 2:
        return False, f"{y_col} is constant"
    return True, tmp


def add_metrics_box(ax, metrics, include_error_metrics=False):
    if include_error_metrics:
        text = (
            f"n = {metrics['n']}\nR = {metrics['r']:.3f}\nR² = {metrics['r2']:.3f}\n"
            f"p = {format_p_value(metrics['p'])}\nMAE = {metrics['mae']:.3f}\nRMSE = {metrics['rmse']:.3f}"
        )
    else:
        text = f"n = {metrics['n']}\nR = {metrics['r']:.3f}\nR² = {metrics['r2']:.3f}\np = {format_p_value(metrics['p'])}"
    ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))


def save_histogram(values, out_png, title, xlabel):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return False, "no finite values"
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.hist(values, bins=20, alpha=0.85)
    ax.axvline(np.mean(values), linestyle="--", label=f"mean={np.mean(values):.2f}")
    ax.axvline(np.median(values), linestyle=":", label=f"median={np.median(values):.2f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    return True, None


def save_scatter(df, x_col, y_col, out_png, title, method=CORR_METHOD, identity=False):
    ok, tmp = valid_xy(df, x_col, y_col)
    if not ok:
        return False, tmp, None

    x, y = tmp[x_col].values, tmp[y_col].values

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    ax.scatter(x, y, alpha=0.72, edgecolors="k")

    try:
        lr = linregress(x, y)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xx, lr.slope * xx + lr.intercept, linestyle="--")
    except Exception:
        pass

    if identity:
        lo = min(np.nanmin(x), np.nanmin(y))
        hi = max(np.nanmax(x), np.nanmax(y))
        ax.plot([lo, hi], [lo, hi], linestyle=":")

    metrics = compute_scatter_metrics(x, y, method, use_identity_r2=identity)
    add_metrics_box(ax, metrics, include_error_metrics=identity)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    return True, None, metrics

def compute_group_pvalue(groups):
    groups = [np.asarray(g, dtype=float) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups if len(g) > 0]
    if len(groups) < 2:
        return {"test": None, "p": np.nan}
    if len(groups) == 2:
        try:
            _, p = ttest_ind(groups[0], groups[1], equal_var=False, nan_policy="omit")
            return {"test": "Welch t-test", "p": float(p)}
        except Exception:
            _, p = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            return {"test": "Mann-Whitney U", "p": float(p)}
    try:
        _, p = f_oneway(*groups)
        return {"test": "One-way ANOVA", "p": float(p)}
    except Exception:
        _, p = kruskal(*groups)
        return {"test": "Kruskal-Wallis", "p": float(p)}


def save_boxplot(df, group_col, value_col, out_png, title):
    if group_col not in df.columns or value_col not in df.columns:
        return False, "missing grouping/value column", None
    tmp = df[[group_col, value_col]].copy()
    tmp[value_col] = clean_numeric_with_sentinels(tmp[value_col])
    tmp = tmp.dropna()
    if len(tmp) == 0:
        return False, "no complete rows", None
    groups, labels = [], []
    for grp, g in tmp.groupby(group_col):
        vals = pd.to_numeric(g[value_col], errors="coerce").dropna().values
        if len(vals) > 0:
            groups.append(vals)
            labels.append(str(grp))
    if len(groups) < 2:
        return False, "need at least 2 non-empty groups", None
    pinfo = compute_group_pvalue(groups)
    plt.figure(figsize=(max(8, len(groups) * 1.4), 6))
    ax = plt.gca()
    ax.boxplot(groups, labels=labels, vert=True)
    ax.set_title(f"{title}\n{pinfo['test']}: p={format_p_value(pinfo['p'])}")
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    return True, None, pinfo


# =========================================================
# ROC HELPERS
# =========================================================
def derive_binary_0_1(series):
    s_num = pd.to_numeric(series, errors="coerce")
    uniq = set(s_num.dropna().unique().tolist())
    if len(uniq) == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    if uniq.issubset({0, 1}):
        return s_num.astype(float)
    if uniq.issubset({1, 2}):
        out = pd.Series(np.nan, index=series.index, dtype=float)
        out[s_num == 1] = 0
        out[s_num == 2] = 1
        return out
    s_str = series.astype(str).str.strip().str.upper()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[s_str.isin(["0", "NO", "FALSE", "NEGATIVE", "N", "NON-CARRIER", "NONCARRIER"])] = 0
    out[s_str.isin(["1", "YES", "TRUE", "POSITIVE", "Y", "CARRIER"])] = 1
    return out


def derive_apoe4_carrier(series):
    s_num = clean_numeric_with_sentinels(series)
    uniq = set(s_num.dropna().unique().tolist())
    if len(uniq) > 0 and uniq.issubset({0, 1}):
        return s_num.astype(float)
    s_str = series.astype(str).str.strip().str.upper().str.replace(" ", "", regex=False)
    out = derive_binary_0_1(series)
    out[s_str.isin(["E4-", "APOE4-", "NONE4", "NON-E4", "NONCARRIER"])] = 0
    out[s_str.isin(["E4+", "APOE4+", "E4CARRIER", "CARRIER"])] = 1
    explicit_carrier = s_str.str.contains(r"E?4[/_]E?[234]|E?[234][/_]E?4|^4[/_]4$|^3[/_]4$|^2[/_]4$", regex=True, na=False)
    explicit_noncarrier = s_str.str.contains(r"^2[/_]2$|^2[/_]3$|^3[/_]3$|^E2[/_]E2$|^E2[/_]E3$|^E3[/_]E3$", regex=True, na=False)
    has_4 = s_str.str.contains(r"4", regex=True, na=False)
    genotype_like = s_str.str.contains(r"[234][/_][234]|E[234][/_]E[234]", regex=True, na=False)
    out[explicit_carrier | (genotype_like & has_4)] = 1
    out[explicit_noncarrier | (genotype_like & ~has_4)] = 0
    return out


def find_apoe_col(df):
    return first_existing_column(df, [
        "APOE4_Positivity_y", "APOE4_Positivity", "APOE4_carrier", "APOE4",
        "apoe4_carrier", "apoe4", "genotype", "APOE_genotype", "APOE", "APOE_y",
    ])


def find_cognition_status_col(df):
    return first_existing_column(df, [
        "group_status", "NORMCOG", "DEMENTED", "Research Group", "Diagnosis", "DX", "DX_bl",
        "Diagnostic_Group", "Group", "cognitive_status", "Cognitive_Status", "Risk", "Risk_y",
    ])


def derive_binary_cognitive_status(df, preferred_col=None):
    col = preferred_col or find_cognition_status_col(df)
    if col is None:
        return None, None, None
    s = df[col]
    s_num = pd.to_numeric(s, errors="coerce")
    if col.upper() == "NORMCOG" and s_num.notna().sum() > 0:
        y = pd.Series(np.nan, index=s.index, dtype=float)
        y[s_num == 1] = 0
        y[s_num == 0] = 1
        return y, col, "Normal vs impaired"
    if col.upper() == "DEMENTED" and s_num.notna().sum() > 0:
        y = pd.Series(np.nan, index=s.index, dtype=float)
        y[s_num == 0] = 0
        y[s_num == 1] = 1
        return y, col, "Non-demented vs demented"
    s_str = s.astype(str).str.strip().str.upper()
    y = pd.Series(np.nan, index=s.index, dtype=float)
    control = {"CN", "HC", "CONTROL", "CONTROLS", "HEALTHY", "NORMAL", "NORMCOG", "CU", "NONDEMENTED", "NON-DEMENTED"}
    impaired = {"MCI", "LMCI", "EMCI", "AD", "DEMENTIA", "DEMENTED", "ALZHEIMER", "IMPAIRED", "CASE", "PATIENT", "ATRISK", "AT_RISK", "AT RISK"}
    for idx, val in s_str.items():
        if val in control or any(tok in val for tok in ["CONTROL", "HEALTHY", "NORMAL", "NORMCOG", "CU"]):
            y.loc[idx] = 0
        elif val in impaired or any(tok in val for tok in ["MCI", "DEMENT", "ALZ", "IMPAIRED", "RISK", "PATIENT"]):
            y.loc[idx] = 1
    return y, col, "Control/CN vs impaired/risk"


def maybe_flip_auc_direction(y_true, y_score):
    y_true = pd.to_numeric(pd.Series(y_true), errors="coerce")
    y_score = pd.to_numeric(pd.Series(y_score), errors="coerce")
    mask = y_true.notna() & y_score.notna()
    y_true = y_true[mask].astype(int)
    y_score = y_score[mask].astype(float)
    if len(y_true) == 0 or sorted(y_true.unique().tolist()) != [0, 1]:
        return y_score, False
    auc = roc_auc_score(y_true, y_score)
    if auc < 0.5:
        return -y_score, True
    return y_score, False


def save_roc_curve(y_true, y_score, out_png, title, score_name):
    y_true = pd.to_numeric(pd.Series(y_true), errors="coerce")
    y_score = pd.to_numeric(pd.Series(y_score), errors="coerce")
    mask = y_true.notna() & y_score.notna()
    y_true = y_true[mask].astype(int)
    y_score = y_score[mask].astype(float)
    if len(y_true) < 2:
        return False, "fewer than 2 complete rows", None
    if sorted(y_true.unique().tolist()) != [0, 1]:
        return False, f"target is not binary: {sorted(y_true.unique().tolist())}", None
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    n0, n1 = int((y_true == 0).sum()), int((y_true == 1).sum())
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.plot(fpr, tpr, lw=2, label=f"{score_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.text(0.03, 0.20, f"n={len(y_true)}\ncontrols={n0}\ncases={n1}\nAUC={auc:.3f}",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    return True, None, {"auc": float(auc), "n": int(len(y_true)), "n0": n0, "n1": n1}


# =========================================================
# HABS OPTIONAL CLINICAL MERGE
# =========================================================
def extract_habs_med_id(x):
    s = str(x).strip()
    m = re.search(r"H(\d{4})_y\d+", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    groups = re.findall(r"\d+", s)
    if groups:
        return groups[-1][-4:].zfill(4)
    return s


def merge_habs_clinical_columns(df_validation, clinical_path):
    if not os.path.exists(clinical_path):
        return df_validation
    try:
        clinical_df = pd.read_excel(clinical_path)
    except Exception as e:
        print(f"Could not read HABS clinical file: {e}")
        return df_validation
    needed = ["Med_ID", "CDX_Diabetes", "CDX_Hypertension", "IMH_HighBP", "OM_BMI"]
    keep = [c for c in needed if c in clinical_df.columns]
    if "Med_ID" not in keep:
        return df_validation
    clinical_small = clinical_df[keep].copy()
    clinical_small["Med_ID"] = clinical_small["Med_ID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(4)
    out = df_validation.copy()
    merge_source_col = first_existing_column(out, ["runno", "connectome_key", "Subject_ID", "graph_id"])
    if merge_source_col is None:
        return out
    out["_med_id_tmp"] = out[merge_source_col].astype(str).map(extract_habs_med_id).astype(str).str.zfill(4)
    merged = out.merge(clinical_small, left_on="_med_id_tmp", right_on="Med_ID", how="left", suffixes=("", "_clinical"))
    return merged.drop(columns=["_med_id_tmp", "Med_ID"], errors="ignore")


# =========================================================
# MAIN VALIDATION FOR ONE COHORT + FEATURE SET
# =========================================================
def run_validation_for(cohort: str, feature_set: str) -> Optional[Dict]:
    paths = discover_inputs(cohort, feature_set)
    val_outdir = paths["val_outdir"]
    ensure_dir(val_outdir)
    if CLEAR_OLD_FIGURES:
        clear_image_files(val_outdir)

    print("\n" + "=" * 90)
    print(f"VALIDATION: cohort={cohort} | feature_set={feature_set}")
    print("=" * 90)
    print("Ablation dir:", paths["ablation_dir"])

    if not os.path.isdir(paths["ablation_dir"]):
        print("Skipping: ablation directory not found")
        return None

    # For the new training comparison, OOF predictions are preferred because
    # they are the validation predictions used in CV metrics. Set
    # PREFER_OOF_FOR_VALIDATION=False if you specifically want final-model
    # full-cohort prediction figures instead.
    if PREFER_OOF_FOR_VALIDATION and paths["oof_path"] is not None:
        print("Using OOF predictions:", paths["oof_path"])
        pred_df = load_table_auto(paths["oof_path"])
        metadata_df = load_table_auto(paths["metadata_path"])
        prediction_source = "oof"
    elif paths["full_pred_path"] is not None:
        print("Using full-cohort predictions:", paths["full_pred_path"])
        pred_df = load_table_auto(paths["full_pred_path"])
        metadata_df = load_table_auto(paths["metadata_all_path"])
        prediction_source = "full_cohort"
    elif paths["oof_path"] is not None:
        print("Using OOF predictions:", paths["oof_path"])
        pred_df = load_table_auto(paths["oof_path"])
        metadata_df = load_table_auto(paths["metadata_path"])
        prediction_source = "oof"
    else:
        print("Skipping: no prediction file found")
        return None

    if pred_df is None or len(pred_df) == 0:
        print("Skipping: empty prediction file")
        return None

    df, merge_key, overlap = merge_predictions_and_metadata(pred_df, metadata_df)
    if cohort == "HABS":
        df = merge_habs_clinical_columns(df, HABS_CLINICAL_PATH)

    # Clean numeric sentinel values globally for numeric columns.
    for c in df.columns:
        if c != "Subject_ID" and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = clean_numeric_with_sentinels(df[c])

    brain_metric = choose_brain_metric(df)
    cognition_vars, imaging_vars = get_candidate_validation_vars(df)
    group_col = find_group_col(df)

    base_vars = unique_preserve_order([brain_metric, "Real_Age", "Predicted_Age_RAW", "Predicted_Age_BiasCorrected"] + cognition_vars + imaging_vars)
    base_vars = [c for c in base_vars if c in df.columns]
    df = add_clean_and_z_versions(df, base_vars)

    brain_metric_raw = f"{brain_metric}_raw_clean"
    brain_metric_z = f"{brain_metric}_z_clean"

    subject_csv = os.path.join(val_outdir, "subject_level_validation_input.csv")
    subject_xlsx = os.path.join(val_outdir, "subject_level_validation_input.xlsx")
    save_table_both(df, subject_csv, subject_xlsx)

    plot_log = []
    corr_rows = []
    roc_rows = []

    # Basic prediction plots.
    for metric_col, scale_tag in [(brain_metric_raw, "raw_clean"), (brain_metric_z, "z_clean")]:
        if metric_col not in df.columns:
            continue
        hist_name = f"{sanitize_filename(metric_col)}_histogram.png"
        ok, reason = save_histogram(df[metric_col].values, os.path.join(val_outdir, hist_name),
                                    f"{cohort} {feature_set}: {metric_col}", metric_col)
        plot_log.append({"plot": hist_name, "status": "saved" if ok else "skipped", "reason": reason, "type": "histogram", "scale": scale_tag})

    age_col = "Real_Age" if "Real_Age" in df.columns else first_existing_column(df, ["age", "Age", "AGE", "VISIT_AGE"])
    pred_cols = [c for c in ["Predicted_Age_RAW", "Predicted_Age_BiasCorrected", "Predicted_Age_GlobalCorrected"] if c in df.columns]
    if age_col is not None:
        for pred_col in pred_cols:
            fname = f"{sanitize_filename(pred_col)}_vs_{sanitize_filename(age_col)}.png"
            ok, reason, metrics = save_scatter(
                df, age_col, pred_col, os.path.join(val_outdir, fname),
                f"{cohort} {feature_set}: {pred_col} vs {age_col}", identity=True
            )
            plot_log.append({"plot": fname, "status": "saved" if ok else "skipped", "reason": reason, "type": "pred_vs_age"})
            if metrics:
                corr_rows.append({"cohort": cohort, "feature_set": feature_set, "brain_metric": pred_col, "variable": age_col, "scale": "raw", "status": "ok", **metrics})

        for metric_col, scale_tag in [(brain_metric_raw, "raw_clean"), (brain_metric_z, "z_clean")]:
            if metric_col in df.columns:
                age_clean = f"{age_col}_raw_clean"
                if age_clean not in df.columns:
                    df[age_clean] = clean_numeric_with_sentinels(df[age_col])
                fname = f"{sanitize_filename(metric_col)}_vs_{sanitize_filename(age_col)}.png"
                ok, reason, metrics = save_scatter(
                    df, age_clean, metric_col, os.path.join(val_outdir, fname),
                    f"{cohort} {feature_set}: {metric_col} vs {age_col}"
                )
                plot_log.append({"plot": fname, "status": "saved" if ok else "skipped", "reason": reason, "type": "bag_vs_age", "scale": scale_tag})
                if metrics:
                    corr_rows.append({"cohort": cohort, "feature_set": feature_set, "brain_metric": metric_col, "variable": age_clean, "scale": scale_tag, "status": "ok", **metrics})

    # Correlation plots with available variables.
    validation_vars = unique_preserve_order(cognition_vars + imaging_vars)
    for var in validation_vars:
        for suffix, scale_tag, y_col in [("raw_clean", "raw_clean", brain_metric_raw), ("z_clean", "z_clean", brain_metric_z)]:
            x_col = f"{var}_{suffix}"
            if x_col not in df.columns or y_col not in df.columns or x_col == y_col:
                continue
            fname = f"{sanitize_filename(y_col)}_vs_{sanitize_filename(x_col)}.png"
            ok, reason, metrics = save_scatter(
                df, x_col, y_col, os.path.join(val_outdir, fname),
                f"{cohort} {feature_set}: {y_col} vs {x_col}"
            )
            plot_log.append({"plot": fname, "status": "saved" if ok else "skipped", "reason": reason, "type": "correlation", "scale": scale_tag})
            if metrics:
                corr_rows.append({"cohort": cohort, "feature_set": feature_set, "brain_metric": y_col, "variable": x_col, "scale": scale_tag, "status": "ok", **metrics})
            else:
                corr_rows.append({"cohort": cohort, "feature_set": feature_set, "brain_metric": y_col, "variable": x_col, "scale": scale_tag, "status": "skipped", "reason": reason})

    # Boxplots by group.
    if group_col is not None:
        for y_col, scale_tag in [(brain_metric_raw, "raw_clean"), (brain_metric_z, "z_clean")]:
            if y_col not in df.columns:
                continue
            fname = f"{sanitize_filename(y_col)}_by_{sanitize_filename(group_col)}.png"
            ok, reason, pinfo = save_boxplot(
                df, group_col, y_col, os.path.join(val_outdir, fname),
                f"{cohort} {feature_set}: {y_col} by {group_col}"
            )
            plot_log.append({"plot": fname, "status": "saved" if ok else "skipped", "reason": reason, "type": "boxplot", "scale": scale_tag})

    # ROC: APOE and cognitive status.
    for score_col, scale_tag in [(brain_metric_raw, "raw_clean"), (brain_metric_z, "z_clean")]:
        if score_col not in df.columns:
            continue
        score = clean_numeric_with_sentinels(df[score_col])

        apoe_col = find_apoe_col(df)
        if apoe_col is not None:
            y = derive_apoe4_carrier(df[apoe_col])
            score_used, flipped = maybe_flip_auc_direction(y, score)
            fname = f"roc_{sanitize_filename(score_col)}_vs_apoe4_carriage.png"
            ok, reason, info = save_roc_curve(
                y, score_used, os.path.join(val_outdir, fname),
                f"{cohort} {feature_set}: ROC APOE4 carriage ({scale_tag})",
                score_col + (" flipped" if flipped else "")
            )
            roc_rows.append({"cohort": cohort, "feature_set": feature_set, "target": "APOE4_carriage", "source_column": apoe_col, "score_column": score_col, "scale": scale_tag, "flipped": flipped, "status": "saved" if ok else "skipped", "reason": reason, **(info or {})})

        y_cog, cog_col, cog_desc = derive_binary_cognitive_status(df)
        if y_cog is not None and cog_col is not None:
            score_used, flipped = maybe_flip_auc_direction(y_cog, score)
            fname = f"roc_{sanitize_filename(score_col)}_vs_cognitive_status.png"
            ok, reason, info = save_roc_curve(
                y_cog, score_used, os.path.join(val_outdir, fname),
                f"{cohort} {feature_set}: ROC {cog_desc} ({scale_tag})",
                score_col + (" flipped" if flipped else "")
            )
            roc_rows.append({"cohort": cohort, "feature_set": feature_set, "target": "cognitive_status", "source_column": cog_col, "score_column": score_col, "scale": scale_tag, "flipped": flipped, "status": "saved" if ok else "skipped", "reason": reason, **(info or {})})

    # Save logs.
    plot_log_df = pd.DataFrame(plot_log)
    corr_df = pd.DataFrame(corr_rows)
    roc_df = pd.DataFrame(roc_rows)

    save_table_both(plot_log_df, os.path.join(val_outdir, "image_generation_log.csv"), os.path.join(val_outdir, "image_generation_log.xlsx"))
    save_table_both(corr_df, os.path.join(val_outdir, "correlation_stats.csv"), os.path.join(val_outdir, "correlation_stats.xlsx"))
    save_table_both(roc_df, os.path.join(val_outdir, "roc_stats.csv"), os.path.join(val_outdir, "roc_stats.xlsx"))

    summary = {
        "cohort": cohort,
        "feature_set": feature_set,
        "prediction_source": prediction_source,
        "ablation_dir": paths["ablation_dir"],
        "val_outdir": val_outdir,
        "oof_path": paths["oof_path"],
        "full_pred_path": paths["full_pred_path"],
        "metadata_path": paths["metadata_path"],
        "metadata_all_path": paths["metadata_all_path"],
        "merge_key": merge_key,
        "merge_overlap": overlap,
        "brain_metric": brain_metric,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "n_cognition_vars": len(cognition_vars),
        "n_imaging_vars": len(imaging_vars),
        "group_col": group_col,
        "n_plots_saved": int((plot_log_df.get("status", pd.Series(dtype=str)) == "saved").sum()) if len(plot_log_df) else 0,
        "n_plots_skipped": int((plot_log_df.get("status", pd.Series(dtype=str)) != "saved").sum()) if len(plot_log_df) else 0,
        "n_roc_saved": int((roc_df.get("status", pd.Series(dtype=str)) == "saved").sum()) if len(roc_df) else 0,
    }
    summary_df = pd.DataFrame([summary])
    save_table_both(summary_df, os.path.join(val_outdir, "validation_summary.csv"), os.path.join(val_outdir, "validation_summary.xlsx"))

    print("Saved validation outputs to:", val_outdir)
    return summary


# =========================================================
# COMBINED SUMMARY AND COMPARISON FIGURES
# =========================================================
def load_cv_summary_for(cohort: str, feature_set: str) -> Optional[pd.DataFrame]:
    paths = discover_inputs(cohort, feature_set)
    path = paths["cv_summary_path"]
    if path is None:
        return None
    df = load_table_auto(path)
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    df["cohort"] = cohort
    df["feature_set"] = feature_set
    df["source_path"] = path
    return df


def load_bootstrap_summary_for(cohort: str, feature_set: str) -> Optional[pd.DataFrame]:
    """
    Load bootstrap confidence intervals generated by the improved training script.

    Expected format:
        feature_set, evaluation, metric, point_estimate, ci_low, ci_high,
        n_bootstrap_valid, n_bootstrap_requested, bootstrap_unit

    One row per metric/evaluation.
    """
    paths = discover_inputs(cohort, feature_set)
    path = paths.get("bootstrap_summary_path")
    if path is None:
        return None

    df = load_table_auto(path)
    if df is None or len(df) == 0:
        return None

    df = df.copy()
    df["cohort"] = cohort
    df["feature_set"] = feature_set
    df["source_path"] = path

    for c in ["point_estimate", "ci_low", "ci_high", "n_bootstrap_valid", "n_bootstrap_requested"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def load_fold_metrics_for(cohort: str, feature_set: str, evaluation: str) -> Optional[pd.DataFrame]:
    """
    Load per-fold CV metrics for one cohort and one feature set.

    evaluation:
        "OOF_RAW" -> uses cv_fold_metrics_raw
        "OOF_BIAS_CORRECTED" -> uses cv_fold_metrics_bias_corrected
    """
    paths = discover_inputs(cohort, feature_set)

    if evaluation == "OOF_RAW":
        path = find_existing_file([paths["cv_fold_raw_csv"], paths["cv_fold_raw_xlsx"]])
    elif evaluation == "OOF_BIAS_CORRECTED":
        path = find_existing_file([paths["cv_fold_bc_csv"], paths["cv_fold_bc_xlsx"]])
    else:
        return None

    if path is None:
        return None

    df = load_table_auto(path)
    if df is None or len(df) == 0:
        return None

    df = df.copy()
    df["cohort"] = cohort
    df["feature_set"] = feature_set
    df["evaluation"] = evaluation
    df["source_path"] = path

    for c in ["MAE", "RMSE", "R2", "r"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def save_metric_comparison_plot(
    df: pd.DataFrame,
    metric: str,
    out_png: str,
    evaluation_filter="OOF_BIAS_CORRECTED",
    bootstrap_df: Optional[pd.DataFrame] = None,
    fold_df: Optional[pd.DataFrame] = None,
    use_bootstrap_ci: bool = True,
    fallback_to_fold_ci: bool = True,
):
    if metric not in df.columns:
        return False

    tmp = df.copy()
    if "evaluation" in tmp.columns and evaluation_filter is not None:
        tmp = tmp[tmp["evaluation"].astype(str) == evaluation_filter].copy()
    if len(tmp) == 0:
        return False

    cohorts = [c for c in COHORTS_TO_RUN if c in tmp["cohort"].unique()]
    feature_sets = [fs for fs in FEATURE_SETS_TO_RUN if fs in tmp["feature_set"].unique()]
    if not cohorts or not feature_sets:
        return False

    # -------------------------------------------------
    # 1) bootstrap CI lookup
    # -------------------------------------------------
    bootstrap_lookup = {}
    if use_bootstrap_ci and bootstrap_df is not None and len(bootstrap_df):
        bs = bootstrap_df.copy()
        required = {"cohort", "feature_set", "evaluation", "metric", "ci_low", "ci_high"}
        if required.issubset(set(bs.columns)):
            bs = bs[
                (bs["evaluation"].astype(str) == str(evaluation_filter)) &
                (bs["metric"].astype(str) == str(metric))
            ].copy()

            for _, row in bs.iterrows():
                key = (str(row["cohort"]), str(row["feature_set"]))
                bootstrap_lookup[key] = {
                    "ci_low": float(row["ci_low"]) if pd.notna(row["ci_low"]) else np.nan,
                    "ci_high": float(row["ci_high"]) if pd.notna(row["ci_high"]) else np.nan,
                }

    # -------------------------------------------------
    # 2) fold-based CI lookup
    # -------------------------------------------------
    fold_lookup = {}
    if fallback_to_fold_ci and fold_df is not None and len(fold_df):
        fd = fold_df.copy()
        needed = {"cohort", "feature_set", "evaluation", metric}
        if needed.issubset(set(fd.columns)):
            fd = fd[fd["evaluation"].astype(str) == str(evaluation_filter)].copy()

            for (cohort, fs), g in fd.groupby(["cohort", "feature_set"]):
                vals = pd.to_numeric(g[metric], errors="coerce").dropna().values
                if len(vals) == 0:
                    continue

                mean_val = float(np.mean(vals))

                if len(vals) >= 2:
                    sd = float(np.std(vals, ddof=1))
                    sem = sd / np.sqrt(len(vals))
                    ci95 = 1.96 * sem
                    ci_low = mean_val - ci95
                    ci_high = mean_val + ci95
                else:
                    ci_low = np.nan
                    ci_high = np.nan

                fold_lookup[(str(cohort), str(fs))] = {
                    "mean": mean_val,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_folds": len(vals),
                }

    x = np.arange(len(cohorts))
    width = 0.8 / max(len(feature_sets), 1)

    plt.figure(figsize=(max(9, len(cohorts) * 2.5), 6))
    ax = plt.gca()

    any_ci_used = False
    ci_source_used = None

    for i, fs in enumerate(feature_sets):
        vals = []
        err_low = []
        err_high = []

        for cohort in cohorts:
            key = (cohort, fs)

            rows = tmp[(tmp["cohort"] == cohort) & (tmp["feature_set"] == fs)]
            summary_val = float(rows[metric].iloc[0]) if len(rows) else np.nan

            ci_low = np.nan
            ci_high = np.nan
            bar_val = summary_val

            if key in bootstrap_lookup:
                ci_low = bootstrap_lookup[key]["ci_low"]
                ci_high = bootstrap_lookup[key]["ci_high"]
                ci_source_used = "bootstrap"
            elif key in fold_lookup:
                # If no bootstrap, use fold mean as bar height and fold 95% CI as error
                bar_val = fold_lookup[key]["mean"]
                ci_low = fold_lookup[key]["ci_low"]
                ci_high = fold_lookup[key]["ci_high"]
                ci_source_used = "fold_95CI"

            vals.append(bar_val)

            if np.isfinite(bar_val) and np.isfinite(ci_low) and np.isfinite(ci_high):
                err_low.append(max(0.0, bar_val - ci_low))
                err_high.append(max(0.0, ci_high - bar_val))
                any_ci_used = True
            else:
                err_low.append(0.0)
                err_high.append(0.0)

        xpos = x - 0.4 + width / 2 + i * width

        if any(np.array(err_low) > 0) or any(np.array(err_high) > 0):
            ax.bar(
                xpos,
                vals,
                width,
                label=fs,
                yerr=np.vstack([err_low, err_high]),
                capsize=4,
                error_kw={"elinewidth": 1.2, "capthick": 1.2},
            )
        else:
            ax.bar(xpos, vals, width, label=fs)

    ax.set_xticks(x)
    ax.set_xticklabels(cohorts)
    ax.set_ylabel(metric)

    if any_ci_used:
        title_suffix = f"with error bars ({ci_source_used})"
    else:
        title_suffix = "no CI available"

    ax.set_title(f"{metric} comparison across cohorts and feature sets ({evaluation_filter}; {title_suffix})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    if metric.upper() == "R2":
        ax.axhline(0, linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    return True

def save_combined_outputs(validation_summaries: List[Dict]):
    combined_dir = ensure_dir(os.path.join(RESULTS_ROOT, COMBINED_VALIDATION_DIR_NAME))

    # Validation summaries.
    val_summary_df = pd.DataFrame(validation_summaries)
    if len(val_summary_df):
        save_table_both(
            val_summary_df,
            os.path.join(combined_dir, "combined_validation_summaries.csv"),
            os.path.join(combined_dir, "combined_validation_summaries.xlsx"),
        )

    # CV summaries and bootstrap CIs from training outputs.
    cv_frames = []
    bootstrap_frames = []
    fold_frames = []
        
    for cohort in COHORTS_TO_RUN:
        for fs in FEATURE_SETS_TO_RUN:
            cv = load_cv_summary_for(cohort, fs)
            if cv is not None:
                cv_frames.append(cv)
    
            bs = load_bootstrap_summary_for(cohort, fs)
            if bs is not None:
                bootstrap_frames.append(bs)
    
            fd_raw = load_fold_metrics_for(cohort, fs, "OOF_RAW")
            if fd_raw is not None:
                fold_frames.append(fd_raw)
    
            fd_bc = load_fold_metrics_for(cohort, fs, "OOF_BIAS_CORRECTED")
            if fd_bc is not None:
                fold_frames.append(fd_bc)

    bootstrap_all = pd.DataFrame()
    if bootstrap_frames:
        bootstrap_all = pd.concat(bootstrap_frames, ignore_index=True)
        save_table_both(
            bootstrap_all,
            os.path.join(combined_dir, "combined_bootstrap_metric_summary.csv"),
            os.path.join(combined_dir, "combined_bootstrap_metric_summary.xlsx"),
        )
        
    fold_all = pd.DataFrame()
    if fold_frames:
        fold_all = pd.concat(fold_frames, ignore_index=True)
        save_table_both(
            fold_all,
            os.path.join(combined_dir, "combined_fold_metrics.csv"),
            os.path.join(combined_dir, "combined_fold_metrics.xlsx"),
        )
    
    if cv_frames:
        cv_all = pd.concat(cv_frames, ignore_index=True)
        save_table_both(
            cv_all,
            os.path.join(combined_dir, "combined_cv_summaries.csv"),
            os.path.join(combined_dir, "combined_cv_summaries.xlsx"),
        )

        for metric in ["MAE", "RMSE", "R2", "r"]:
            # Main filenames now include bootstrap error bars when CI files exist.
            save_metric_comparison_plot(
                cv_all,
                metric,
                os.path.join(combined_dir, f"comparison_{metric}_OOF_BIAS_CORRECTED.png"),
                evaluation_filter="OOF_BIAS_CORRECTED",
                bootstrap_df=bootstrap_all,
                fold_df=fold_all,
                use_bootstrap_ci=True,
                fallback_to_fold_ci=True,
            )
                            
            save_metric_comparison_plot(
                cv_all,
                metric,
                os.path.join(combined_dir, f"comparison_{metric}_OOF_RAW.png"),
                evaluation_filter="OOF_RAW",
                bootstrap_df=bootstrap_all,
                fold_df=fold_all,
                use_bootstrap_ci=True,
                fallback_to_fold_ci=True,
           )
        

            # Also keep no-CI copies for debugging/comparison.
            save_metric_comparison_plot(
                cv_all,
                metric,
                os.path.join(combined_dir, f"comparison_{metric}_OOF_BIAS_CORRECTED_no_CI.png"),
                evaluation_filter="OOF_BIAS_CORRECTED",
                bootstrap_df=None,
                use_bootstrap_ci=False,
            )
            save_metric_comparison_plot(
                cv_all,
                metric,
                os.path.join(combined_dir, f"comparison_{metric}_OOF_RAW_no_CI.png"),
                evaluation_filter="OOF_RAW",
                bootstrap_df=None,
                use_bootstrap_ci=False,
            )

    print("\nCombined outputs saved to:", combined_dir)
    if len(bootstrap_all):
        print("Combined bootstrap CIs saved to:", os.path.join(combined_dir, "combined_bootstrap_metric_summary.csv"))
    else:
        print("Warning: no bootstrap CI files were found. Comparison plots were saved without error bars.")



# =========================================================
# RUN
# =========================================================
def main():
    validation_summaries = []
    for cohort in COHORTS_TO_RUN:
        for feature_set in FEATURE_SETS_TO_RUN:
            try:
                summary = run_validation_for(cohort, feature_set)
                if summary is not None:
                    validation_summaries.append(summary)
            except Exception as e:
                print(f"ERROR in cohort={cohort}, feature_set={feature_set}: {e}")
                validation_summaries.append({
                    "cohort": cohort,
                    "feature_set": feature_set,
                    "status": "error",
                    "error": str(e),
                })

    save_combined_outputs(validation_summaries)


if __name__ == "__main__":
    main()
