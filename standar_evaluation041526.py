#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone brain-age validation script
--------------------------------------
What this script does:
1. Loads full-cohort predictions if available; otherwise falls back to subject/global or OOF predictions
2. Loads aligned metadata and merges it into the validation dataframe
3. Chooses validation brain metric automatically:
      cBAG_global -> cBAG -> BAG
4. Cleans sentinel missing values (e.g. -7777, -8888, -9999, 7777, 8888, 9999)
5. Generates:
      - histogram of brain metric in raw-clean and z-clean versions
      - correlation scatter plots vs cognition / imaging / vascular / metabolic variables
        for BOTH raw-clean and z-clean versions of the SAME variables
      - dedicated stats CSV for all attempted correlations
      - ROC curves for APOE4 carriage and cognitive status
      - boxplots by diagnosis/group if available
6. Writes logs and summary files

Expected environment:
- WORK environment variable defined
- results already produced by your training/evaluation pipeline
"""

import os
import re
import sys
import glob
import warnings

# =========================================================
# MAKE CUSTOM UTILS IMPORTABLE
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)

for p in [SCRIPT_DIR, CODE_DIR]:
    if p not in sys.path:
        sys.path.append(p)

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
matplotlib.set_loglevel("error")
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr, linregress, ttest_ind, f_oneway, mannwhitneyu, kruskal

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_curve,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# =========================================================
# CONFIG
# =========================================================
WORK = os.environ["WORK"]

COHORT_NAME = "HABS"   # <<< CHANGE ONLY THIS
RESULTS_ROOT = os.path.join(WORK, "ines/results")

RESULTS_DIR_MAP = {
    "ADNI": "BrainAgePredictionADNI",
    "ADRC": "BrainAgePredictionADRC",
    "HABS": "BrainAgePredictionHABS",
    "AD_DECODE": "BrainAgePredictionADDECODE",
}

if COHORT_NAME not in RESULTS_DIR_MAP:
    raise ValueError(f"Unsupported COHORT_NAME: {COHORT_NAME}")

results_dir = os.path.join(RESULTS_ROOT, RESULTS_DIR_MAP[COHORT_NAME])
val_outdir = os.path.join(results_dir, "validation")
os.makedirs(val_outdir, exist_ok=True)

BRAIN_METRIC_COL = "cBAG_global"
CORR_METHOD = "pearson"

HABS_CLINICAL_PATH = os.path.join(
    WORK,
    "ines/data/harmonization/HABS/metadata/RP_HD_7_Clinical.xlsx"
)

# Sentinel codes treated as missing
SENTINEL_VALUES = {-9999, -8888, -7777, 9999, 8888, 7777}

if COHORT_NAME == "AD_DECODE":
    COHORT_FILE_STEM = "addecode"
else:
    COHORT_FILE_STEM = COHORT_NAME.lower()

COHORT_LOWER = COHORT_NAME.lower()


# =========================================================
# HELPERS
# =========================================================
def keep_existing_cols(df, cols):
    return [c for c in cols if c in df.columns]


def find_existing_file(candidates):
    for fp in candidates:
        if fp is not None and os.path.exists(fp):
            return fp
    return None


def load_table_auto(path):
    if path is None:
        return None
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


def ensure_subject_id_col(df):
    if "Subject_ID" not in df.columns:
        if "graph_id" in df.columns:
            df["Subject_ID"] = df["graph_id"].astype(str)
        elif "PTID" in df.columns:
            df["Subject_ID"] = df["PTID"].astype(str)
        elif "connectome_key" in df.columns:
            df["Subject_ID"] = df["connectome_key"].astype(str)
        elif "connectome_id" in df.columns:
            df["Subject_ID"] = df["connectome_id"].astype(str)
        elif "subject_id" in df.columns:
            df["Subject_ID"] = df["subject_id"].astype(str)
        elif "RID" in df.columns:
            df["Subject_ID"] = df["RID"].astype(str)
        elif "ID" in df.columns:
            df["Subject_ID"] = df["ID"].astype(str)
        elif "MRI_Exam" in df.columns:
            df["Subject_ID"] = df["MRI_Exam"].astype(str)
        elif "match_id" in df.columns:
            df["Subject_ID"] = df["match_id"].astype(str)
        else:
            raise KeyError(
                "No usable subject identifier found. "
                "Need one of: Subject_ID, graph_id, PTID, connectome_key, "
                "connectome_id, subject_id, RID, ID, MRI_Exam, match_id."
            )
    df["Subject_ID"] = df["Subject_ID"].astype(str)
    return df


def normalize_id_series(series):
    s = series.astype(str).str.strip().str.upper()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s


def first_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_clean_str_series(s):
    return s.astype(str).str.strip().str.upper()


def sanitize_filename(name):
    safe = str(name)
    for old, new in [
        (" ", "_"),
        ("/", "_"),
        ("\\", "_"),
        ("(", ""),
        (")", ""),
        ("[", ""),
        ("]", ""),
        (":", "_"),
        (";", "_"),
        (",", "_"),
        ("__", "_"),
    ]:
        safe = safe.replace(old, new)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe

def strip_clean_suffix(col_name):
    s = str(col_name)
    for suf in ["_raw_clean", "_z_clean"]:
        if s.endswith(suf):
            return s[:-len(suf)]
    return s

def clear_image_files(folder):
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.svg", "*.pdf"]
    for pat in patterns:
        for fp in glob.glob(os.path.join(folder, pat)):
            try:
                os.remove(fp)
            except Exception:
                pass


def _format_p_value(p):
    if pd.isna(p):
        return "nan"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def numeric_series(s):
    return pd.to_numeric(s, errors="coerce")


def clean_numeric_with_sentinels(series, extra_sentinels=None):
    s = pd.to_numeric(series, errors="coerce").copy()
    sentinels = set(SENTINEL_VALUES)
    if extra_sentinels is not None:
        sentinels.update(extra_sentinels)
    for val in sentinels:
        s = s.mask(s == val, np.nan)
    return s


def zscore_series(series):
    s = clean_numeric_with_sentinels(series)
    valid = s.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=s.index, dtype=float)
    std = valid.std(ddof=1)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=s.index, dtype=float)
    return (s - valid.mean()) / std


def add_clean_and_z_versions(df, cols):
    df = df.copy()
    created = []

    for c in cols:
        if c not in df.columns:
            continue

        raw_clean_col = f"{c}_raw_clean"
        z_clean_col = f"{c}_z_clean"

        df[raw_clean_col] = clean_numeric_with_sentinels(df[c])
        df[z_clean_col] = zscore_series(df[c])

        created.extend([raw_clean_col, z_clean_col])

    return df, created


def normalize_prediction_columns(df):
    rename_map = {}

    aliases = {
        "Real_Age": [
            "Real_Age", "Age", "age", "VISIT_AGE", "AGE", "age_true"
        ],
        "Predicted_Age_RAW": [
            "Predicted_Age_RAW", "Predicted_Age_raw", "Pred_raw",
            "PredictedAgeRaw", "y_pred_raw", "pred_raw"
        ],
        "Predicted_Age_BiasCorrected": [
            "Predicted_Age_BiasCorrected", "Predicted_Age_corrected",
            "Pred_corr", "Pred_corr_foldwise", "y_pred_corrected",
            "pred_bias_corrected"
        ],
        "Predicted_Age_GlobalCorrected": [
            "Predicted_Age_GlobalCorrected", "Pred_corr_global",
            "Predicted_Age_corrected_global", "y_pred_global_corrected",
            "pred_global_corrected"
        ],
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

    return df.rename(columns=rename_map).copy()


def coalesce_meta_columns(df):
    df = df.copy()
    meta_cols = [c for c in df.columns if c.endswith("_meta")]

    for meta_col in meta_cols:
        base_col = meta_col[:-5]
        if base_col in df.columns:
            df[base_col] = df[base_col].combine_first(df[meta_col])
        else:
            df[base_col] = df[meta_col]

    return df


def unique_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def find_existing_columns_by_patterns(df, patterns):
    cols_found = []
    for col in df.columns:
        col_low = str(col).strip().lower()
        for pat in patterns:
            if pat.lower() in col_low:
                cols_found.append(col)
                break
    return cols_found


def find_best_metadata_merge_key(metadata_df, graph_ids):
    candidate_cols = [
        "connectome_key",
        "match_id",
        "subject_id",
        "PTID",
        "ptid",
        "regional_id",
        "Subject_ID",
        "RID",
        "MRI_Exam",
        "graph_id",
    ]
    graph_id_set = set(map(str, graph_ids))

    best_col = None
    best_matches = -1

    for col in candidate_cols:
        if col in metadata_df.columns:
            meta_vals = set(normalize_id_series(metadata_df[col]).tolist())
            overlap = len(graph_id_set.intersection(meta_vals))
            if overlap > best_matches:
                best_matches = overlap
                best_col = col

    return best_col, best_matches


def compute_scatter_metrics(x, y, corr_method="pearson", use_identity_r2=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return {
            "n": len(x),
            "r": np.nan,
            "p": np.nan,
            "r2": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
        }

    if corr_method.lower() == "spearman":
        r, p = spearmanr(x, y)
    else:
        r, p = pearsonr(x, y)

    mae = mean_absolute_error(x, y)
    rmse = np.sqrt(mean_squared_error(x, y))

    if use_identity_r2:
        r2 = r2_score(x, y)
    else:
        r2 = r ** 2

    lr = linregress(x, y)

    return {
        "n": int(len(x)),
        "r": float(r),
        "p": float(p),
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "slope": float(lr.slope),
        "intercept": float(lr.intercept),
    }


def compute_distribution_metrics(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)

    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "median": float(np.median(x)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def extract_habs_subject_base(x):
    s = str(x).strip()
    m = re.search(r"H(\d{4})_y\d+", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return np.nan


def extract_habs_visit_num(x):
    s = str(x).strip()
    m = re.search(r"_y(\d+)$", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return np.nan


def valid_delta_scatter_xy(df, x_col, y_col):
    if x_col not in df.columns or y_col not in df.columns:
        return False, f"missing column: {x_col if x_col not in df.columns else y_col}"

    tmp = df[[x_col, y_col]].copy()
    tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce")
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
    tmp = tmp.dropna()

    if len(tmp) < 3:
        return False, "fewer than 3 complete rows"

    if tmp[x_col].nunique(dropna=True) < 2:
        return False, f"{x_col} is constant"
    if tmp[y_col].nunique(dropna=True) < 2:
        return False, f"{y_col} is constant"

    return True, tmp

def add_metrics_box(ax, metrics, loc=(0.03, 0.97), fontsize=10, include_error_metrics=True):
    if include_error_metrics:
        text = (
            f"n = {metrics['n']}\n"
            f"R = {metrics['r']:.3f}\n"
            f"R² = {metrics['r2']:.3f}\n"
            f"p = {_format_p_value(metrics['p'])}\n"
            f"MAE = {metrics['mae']:.3f}\n"
            f"RMSE = {metrics['rmse']:.3f}"
        )
    else:
        text = (
            f"n = {metrics['n']}\n"
            f"R = {metrics['r']:.3f}\n"
            f"R² = {metrics['r2']:.3f}\n"
            f"p = {_format_p_value(metrics['p'])}"
        )

    ax.text(
        loc[0],
        loc[1],
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fontsize,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def add_distribution_box(ax, metrics, loc=(0.03, 0.97), fontsize=10):
    text = (
        f"n = {metrics['n']}\n"
        f"mean = {metrics['mean']:.3f}\n"
        f"std = {metrics['std']:.3f}\n"
        f"median = {metrics['median']:.3f}\n"
        f"IQR = {metrics['iqr']:.3f}"
    )

    ax.text(
        loc[0],
        loc[1],
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fontsize,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

def compute_group_pvalue(grouped):
    """
    grouped = list of 1D numeric arrays, one per group
    Returns:
        dict with test name, p-value, and group count
    """
    grouped = [np.asarray(g, dtype=float) for g in grouped]
    grouped = [g[np.isfinite(g)] for g in grouped if len(g) > 0]

    if len(grouped) < 2:
        return {"test": None, "p": np.nan, "k": len(grouped)}

    # 2 groups -> Welch t-test
    if len(grouped) == 2:
        try:
            stat, p = ttest_ind(grouped[0], grouped[1], equal_var=False, nan_policy="omit")
            return {"test": "Welch t-test", "p": float(p), "k": 2}
        except Exception:
            try:
                stat, p = mannwhitneyu(grouped[0], grouped[1], alternative="two-sided")
                return {"test": "Mann-Whitney U", "p": float(p), "k": 2}
            except Exception:
                return {"test": None, "p": np.nan, "k": 2}

    # >2 groups -> one-way ANOVA
    try:
        stat, p = f_oneway(*grouped)
        return {"test": "One-way ANOVA", "p": float(p), "k": len(grouped)}
    except Exception:
        try:
            stat, p = kruskal(*grouped)
            return {"test": "Kruskal-Wallis", "p": float(p), "k": len(grouped)}
        except Exception:
            return {"test": None, "p": np.nan, "k": len(grouped)}


def add_boxplot_stats_box(ax, dist_metrics, pinfo, loc=(0.03, 0.97), fontsize=10):
    test_name = pinfo["test"] if pinfo["test"] is not None else "NA"
    p_text = _format_p_value(pinfo["p"])

    text = (
        f"n = {dist_metrics['n']}\n"
        f"mean = {dist_metrics['mean']:.3f}\n"
        f"std = {dist_metrics['std']:.3f}\n"
        f"median = {dist_metrics['median']:.3f}\n"
        f"IQR = {dist_metrics['iqr']:.3f}\n"
        f"test = {test_name}\n"
        f"p = {p_text}"
    )

    ax.text(
        loc[0],
        loc[1],
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fontsize,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def valid_scatter_xy(df, x_col, y_col):
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


def save_histogram_with_stats(values, out_png, title, xlabel):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return False, "no finite values"

    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.hist(values, bins=20, alpha=0.85)
    metrics = compute_distribution_metrics(values)
    add_distribution_box(ax, metrics)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    return True, None


def save_correlation_scatter(df, x_col, y_col, out_png, title, method="pearson"):
    ok, tmp = valid_scatter_xy(df, x_col, y_col)
    if not ok:
        return False, tmp

    x = tmp[x_col].values
    y = tmp[y_col].values

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.scatter(x, y, alpha=0.7, edgecolors="k")

    try:
        lr = linregress(x, y)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        yy = lr.slope * xx + lr.intercept
        ax.plot(xx, yy, linestyle="--")
    except Exception:
        pass

    metrics = compute_scatter_metrics(x=x, y=y, corr_method=method, use_identity_r2=False)
    add_metrics_box(ax, metrics, include_error_metrics=False)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    return True, None


def save_boxplot_with_stats(df, group_col, value_col, out_png, title):
    if group_col not in df.columns or value_col not in df.columns:
        return False, "missing grouping/value column"

    tmp = df[[group_col, value_col]].copy()
    tmp[value_col] = clean_numeric_with_sentinels(tmp[value_col])

    # quitar Unknown solo en HABS
    if COHORT_NAME == "HABS":
        group_str = tmp[group_col].astype(str).str.strip().str.upper()
        tmp = tmp.loc[~group_str.isin(["UNKNOWN", "UNK", "NAN", "NONE", ""])]
        # por si viene algo tipo "Unknown subject"
        tmp = tmp.loc[~group_str.str.contains("UNKNOWN", na=False)]

    tmp = tmp.dropna()

    if len(tmp) == 0:
        return False, "no complete rows"

    grouped = []
    labels = []
    group_sizes = {}

    for grp, g in tmp.groupby(group_col):
        vals = pd.to_numeric(g[value_col], errors="coerce").values
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            grouped.append(vals)
            labels.append(str(grp))
            group_sizes[str(grp)] = len(vals)

    if len(grouped) < 2:
        return False, "need at least 2 non-empty groups"

    print(f"Boxplot {group_col} sizes:", group_sizes)

    dist_metrics = compute_distribution_metrics(tmp[value_col].values)
    pinfo = compute_group_pvalue(grouped)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.boxplot(grouped, tick_labels=labels, vert=True)
    add_boxplot_stats_box(ax, dist_metrics, pinfo)

    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    return True, None

def find_apoe_carrier_column(df):
    candidates = [
        "APOE4_Positivity_y",
        "APOE4_Positivity",
        "APOE4_carrier",
        "APOE4",
        "APOE",
        "apoe4_carrier",
        "apoe4",
        "genotype",
        "APOE_genotype",
    ]
    return first_existing_column(df, candidates)

def merge_habs_clinical_columns(df_validation, clinical_path):
    if not os.path.exists(clinical_path):
        print(f"HABS clinical file not found: {clinical_path}")
        return df_validation

    clinical_df = pd.read_excel(clinical_path)

    needed_cols = [
        "Med_ID",
        "CDX_Diabetes",
        "CDX_Hypertension",
        "IMH_HighBP",
        "OM_BMI",
    ]
    keep_cols = [c for c in needed_cols if c in clinical_df.columns]

    if "Med_ID" not in keep_cols:
        print("HABS clinical file does not contain Med_ID; skipping clinical merge.")
        return df_validation

    clinical_small = clinical_df[keep_cols].copy()
    clinical_small["Med_ID"] = (
        clinical_small["Med_ID"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(4)
    )

    out = df_validation.copy()

    # usa runno/connectome key longitudinal y extrae los 4 dígitos
    merge_source_col = None
    for c in ["runno", "connectome_key", "Subject_ID", "graph_id"]:
        if c in out.columns:
            merge_source_col = c
            break

    if merge_source_col is None:
        print("No usable HABS ID column found in validation dataframe.")
        return out

    out["_med_id_tmp"] = out[merge_source_col].astype(str).map(extract_habs_med_id)
    out["_med_id_tmp"] = out["_med_id_tmp"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)

    print("\n===== HABS CLINICAL MERGE DEBUG =====")
    print("merge_source_col:", merge_source_col)
    print("Sample validation IDs:", out[merge_source_col].head(10).tolist())
    print("Sample extracted med ids:", out["_med_id_tmp"].head(10).tolist())
    print("Sample clinical Med_ID:", clinical_small["Med_ID"].head(10).tolist())

    overlap = len(set(out["_med_id_tmp"]).intersection(set(clinical_small["Med_ID"])))
    print("Overlap with clinical Med_ID:", overlap)

    merged = out.merge(
        clinical_small,
        left_on="_med_id_tmp",
        right_on="Med_ID",
        how="left",
        suffixes=("", "_clinical")
    )

    merged = merged.drop(columns=["_med_id_tmp", "Med_ID"], errors="ignore")

    for c in ["CDX_Diabetes", "CDX_Hypertension", "IMH_HighBP", "OM_BMI"]:
        print(f"{c} present:", c in merged.columns)
        if c in merged.columns:
            print(f"  non-null = {merged[c].notna().sum()}")
            print(f"  unique = {pd.Series(merged[c]).dropna().unique()[:10]}")

    return merged

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

    out[s_str.isin(["0", "NO", "FALSE", "NEGATIVE", "N"])] = 0
    out[s_str.isin(["1", "YES", "TRUE", "POSITIVE", "Y"])] = 1

    return out

def extract_habs_med_id(x):
    s = str(x).strip()
    m = re.search(r"H(\d{4})_y\d+", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return s

def derive_bmi_27_binary(series):
    s = pd.to_numeric(series, errors="coerce")
    s = s.mask(s < 10, np.nan)
    s = s.mask(s > 80, np.nan)

    out = pd.Series(np.nan, index=series.index, dtype=float)
    BMI_thresh = 30
    out[s < BMI_thresh] = 0
    out[s >= BMI_thresh] = 1
    return out


def derive_apoe4_carrier(series):
    s_num = clean_numeric_with_sentinels(series)

    # Caso simple: ya viene binaria 0/1
    uniq_num = set(s_num.dropna().unique().tolist())
    if len(uniq_num) > 0 and uniq_num.issubset({0, 1}):
        return s_num.astype(float)

    s_str = series.astype(str).str.strip().str.upper()
    out = pd.Series(np.nan, index=series.index, dtype=float)

    out[s_str.isin(["0", "NON-CARRIER", "NONCARRIER", "NEGATIVE", "FALSE", "NO"])] = 0
    out[s_str.isin(["1", "CARRIER", "POSITIVE", "TRUE", "YES"])] = 1

    out[s_str.isin(["E4-", "E4 NEG", "APOE4-", "NON E4", "NON-E4"])] = 0
    out[s_str.isin(["E4+", "E4 POS", "APOE4+", "E4 CARRIER", "APOE4 CARRIER"])] = 1

    has_4 = s_str.str.contains(r"(^|[^0-9])4([^0-9]|$)", regex=True, na=False)
    genotype_like = s_str.str.contains(r"[234]/[234]|E[234]/E[234]|[234][ ]*/[ ]*[234]", regex=True, na=False)
    out[genotype_like & has_4] = 1
    out[genotype_like & (~has_4)] = 0

    explicit_carrier = s_str.str.contains(
        r"E?4\s*/\s*E?[234]|E?[234]\s*/\s*E?4|^4/4$|^3/4$|^2/4$",
        regex=True,
        na=False,
    )
    explicit_noncarrier = s_str.str.contains(
        r"^2/2$|^2/3$|^3/3$|^E2/E2$|^E2/E3$|^E3/E3$",
        regex=True,
        na=False,
    )

    out[explicit_carrier] = 1
    out[explicit_noncarrier] = 0

    return out


def find_cognition_status_column(df):
    candidates = [
        "group_status",
        "NORMCOG",
        "DEMENTED",
        "Research Group",
        "Diagnosis",
        "DX",
        "DX_bl",
        "Diagnostic_Group",
        "Group",
        "cognitive_status",
        "Cognitive_Status",
    ]
    return first_existing_column(df, candidates)


def derive_binary_cognitive_status(df, preferred_col=None):
    col = preferred_col if preferred_col is not None else find_cognition_status_column(df)
    if col is None:
        return None, None, None

    s = df[col]

    if col == "NORMCOG":
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().sum() > 0:
            y = pd.Series(np.nan, index=s.index, dtype=float)
            y[s_num == 1] = 0
            y[s_num == 0] = 1
            return y, col, "Normal vs impaired"

    if col == "DEMENTED":
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().sum() > 0:
            y = pd.Series(np.nan, index=s.index, dtype=float)
            y[s_num == 0] = 0
            y[s_num == 1] = 1
            return y, col, "Non-demented vs demented"

    s_str = s.astype(str).str.strip().str.upper()
    y = pd.Series(np.nan, index=s.index, dtype=float)

    control_tokens = {
        "CN", "HC", "CONTROL", "CONTROLS", "HEALTHY", "NORMAL",
        "NORMCOG", "CU", "COGNITIVELY NORMAL", "NONDEMENTED",
        "NON-DEMENTED", "NO RISK/FAMILIAL", "NORISK/FAMILIAL", "NO RISK"
    }

    impaired_tokens = {
        "MCI", "LMCI", "EMCI", "AD", "DEMENTIA", "DEMENTED",
        "ALZHEIMER", "ALZHEIMER'S", "ALZ", "IMPAIRED",
        "COGNITIVE IMPAIRMENT", "CI", "CASE", "PATIENT",
        "ATRISK", "AT_RISK", "AT RISK", "IMPAIRED_NON_MCI"
    }

    for idx, val in s_str.items():
        if val in control_tokens:
            y.loc[idx] = 0
            continue
        if val in impaired_tokens:
            y.loc[idx] = 1
            continue

        if any(tok in val for tok in ["CN", "CONTROL", "HEALTHY", "NORMAL", "HC", "CU", "NORMCOG"]):
            y.loc[idx] = 0
            continue

        if any(tok in val for tok in ["MCI", "LMCI", "EMCI", "AD", "DEMENT", "ALZ", "IMPAIRED", "CASE", "PATIENT", "RISK"]):
            y.loc[idx] = 1
            continue

    return y, col, "Control/CN vs MCI/AD/impaired"


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


def save_roc_curve_from_score(y_true, y_score, out_png, title, score_name):
    y_true = pd.to_numeric(pd.Series(y_true), errors="coerce")
    y_score = pd.to_numeric(pd.Series(y_score), errors="coerce")

    mask = y_true.notna() & y_score.notna()
    y_true = y_true[mask].astype(int)
    y_score = y_score[mask].astype(float)

    if len(y_true) < 2:
        return False, "fewer than 2 complete rows"

    uniq = sorted(y_true.unique().tolist())
    if uniq != [0, 1]:
        return False, f"target is not binary: {uniq}"

    n0 = int((y_true == 0).sum())
    n1 = int((y_true == 1).sum())
    if n0 < 1 or n1 < 1:
        return False, f"need both classes present (n0={n0}, n1={n1})"

    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.plot(fpr, tpr, lw=2, label=f"{score_name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    text = f"n = {len(y_true)}\ncontrols = {n0}\ncases = {n1}\nAUC = {auc:.3f}"
    ax.text(
        0.03,
        0.20,
        text,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    return True, {"auc": float(auc), "n": int(len(y_true)), "n0": n0, "n1": n1}


# =========================================================
# PATH DISCOVERY
# =========================================================
cv_preds_path = find_existing_file([
    os.path.join(results_dir, f"cv_predictions_{COHORT_FILE_STEM}.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_cv_oof_predictions.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_oof_predictions.csv"),
    os.path.join(results_dir, f"cv_predictions_{COHORT_FILE_STEM}.xlsx"),
])

global_oof_path = find_existing_file([
    os.path.join(results_dir, f"cv_predictions_{COHORT_FILE_STEM}_with_global_oof.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_cv_predictions_with_global_oof.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_global_oof_predictions.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_global_oof_predictions.xlsx"),
])

metadata_path = find_existing_file([
    os.path.join(WORK, "ines/results/harmonized", COHORT_NAME, "graphs", f"{COHORT_LOWER}_metadata_aligned_raw.csv"),
    os.path.join(WORK, "ines/results/harmonized", COHORT_NAME, "graphs", f"{COHORT_LOWER}_metadata_aligned.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_aligned_raw.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_aligned.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_aligned.xlsx"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_with_cv_predictions.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_with_cv_predictions.xlsx"),
])

full_cohort_preds_path = find_existing_file([
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_full_cohort_predictions.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_full_cohort_predictions.xlsx"),
])

metadata_all_path = find_existing_file([
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_all_raw_with_predictions_plus_brainvol_hipp_fa.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_all_raw_with_predictions_plus_brainvol_hipp_fa.xlsx"),

    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_all_with_predictions_plus_brainvol_hipp_fa.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_all_with_predictions_plus_brainvol_hipp_fa.xlsx"),

    os.path.join(WORK, "ines/results/harmonized", COHORT_NAME, "graphs", f"{COHORT_LOWER}_metadata_all_aligned_raw.csv"),
    os.path.join(WORK, "ines/results/harmonized", COHORT_NAME, "graphs", f"{COHORT_LOWER}_metadata_all_aligned.csv"),

    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_all_with_predictions.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_all_aligned_raw.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_all_aligned.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_all_with_predictions.xlsx"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_all_aligned.xlsx"),
])

clear_image_files(val_outdir)

if metadata_all_path is not None:
    aligned_metadata_all = load_table_auto(metadata_all_path)
    aligned_metadata_all = ensure_subject_id_col(aligned_metadata_all)
    aligned_metadata_all = normalize_prediction_columns(aligned_metadata_all)
    print("Loaded metadata_all_path:", metadata_all_path)
else:
    aligned_metadata_all = None


if cv_preds_path is None:
    raise FileNotFoundError(f"Missing OOF predictions file for cohort {COHORT_NAME}")
if metadata_path is None:
    raise FileNotFoundError(f"Missing metadata file for cohort {COHORT_NAME}")

df_preds = load_table_auto(cv_preds_path)
aligned_metadata = load_table_auto(metadata_path)

df_preds = ensure_subject_id_col(df_preds)
aligned_metadata = ensure_subject_id_col(aligned_metadata)

df_preds = normalize_prediction_columns(df_preds)
aligned_metadata = normalize_prediction_columns(aligned_metadata)

if full_cohort_preds_path is not None:
    df_full_preds = load_table_auto(full_cohort_preds_path)
    df_full_preds = ensure_subject_id_col(df_full_preds)
    df_full_preds = normalize_prediction_columns(df_full_preds)
else:
    df_full_preds = None

if metadata_all_path is not None:
    aligned_metadata_all = load_table_auto(metadata_all_path)
    aligned_metadata_all = ensure_subject_id_col(aligned_metadata_all)
    aligned_metadata_all = normalize_prediction_columns(aligned_metadata_all)
else:
    aligned_metadata_all = None

if global_oof_path is not None:
    df_global = load_table_auto(global_oof_path)
    df_global = ensure_subject_id_col(df_global)
    df_global = normalize_prediction_columns(df_global)
else:
    df_global = None

print("\nLoaded:")
print("  predictions:", df_preds.shape)
print("  metadata:", aligned_metadata.shape)
print("  global_oof:", None if df_global is None else df_global.shape)
print("  full_cohort_predictions:", None if df_full_preds is None else df_full_preds.shape)
print("  metadata_all:", None if aligned_metadata_all is None else aligned_metadata_all.shape)


# =========================================================
# CHOOSE VALIDATION DATAFRAME
# =========================================================
if df_full_preds is not None:
    print("Validation will use FULL-COHORT predictions.")
    df_for_validation = df_full_preds.copy()
elif df_global is not None:
    print("Validation will use subject-level / global OOF dataframe.")
    df_for_validation = df_global.copy()
else:
    print("Validation will fall back to OOF prediction dataframe.")
    df_for_validation = df_preds.copy()

df_for_validation = ensure_subject_id_col(df_for_validation)
df_for_validation["Subject_ID"] = normalize_id_series(df_for_validation["Subject_ID"])

if BRAIN_METRIC_COL in df_for_validation.columns:
    validation_brain_metric_col = BRAIN_METRIC_COL
elif "cBAG" in df_for_validation.columns:
    validation_brain_metric_col = "cBAG"
elif "BAG" in df_for_validation.columns:
    validation_brain_metric_col = "BAG"
else:
    raise KeyError("No usable brain metric found. Need cBAG_global, cBAG, or BAG.")

print("Validation brain metric:", validation_brain_metric_col)


# =========================================================
# CHOOSE METADATA SOURCE
# =========================================================
if aligned_metadata_all is not None:
    print("Validation metadata source: FULL-COHORT metadata.")
    aligned_metadata_tmp = aligned_metadata_all.copy()
else:
    print("Validation metadata source: standard aligned metadata.")
    aligned_metadata_tmp = aligned_metadata.copy()

for col in aligned_metadata_tmp.columns:
    if col in ["connectome_key", "match_id", "subject_id", "PTID", "ptid", "regional_id", "Subject_ID", "RID", "MRI_Exam", "graph_id"]:
        aligned_metadata_tmp[col] = normalize_id_series(aligned_metadata_tmp[col])


# =========================================================
# MERGE METADATA
# =========================================================
merge_key, overlap = find_best_metadata_merge_key(
    aligned_metadata_tmp,
    df_for_validation["Subject_ID"].astype(str).tolist(),
)

print("\n===== VALIDATION MERGE DEBUG =====")
print("df_for_validation rows before merge:", len(df_for_validation))
print("aligned_metadata rows:", len(aligned_metadata_tmp))
print("Chosen merge_key:", merge_key)
print("Overlap:", overlap)
print("Sample validation IDs:", df_for_validation["Subject_ID"].head(10).tolist())
if merge_key is not None:
    print("Sample metadata IDs:", aligned_metadata_tmp[merge_key].head(10).tolist())

if merge_key is not None and overlap > 0:
    aligned_metadata_subject = aligned_metadata_tmp.drop_duplicates(subset=[merge_key]).copy()
    aligned_metadata_subject = aligned_metadata_subject[
        [c for c in aligned_metadata_subject.columns if not c.endswith("_meta")]
    ].copy()

    keep_cols = [merge_key] + [
        c for c in aligned_metadata_subject.columns
        if c != merge_key and c not in df_for_validation.columns
    ]
    aligned_metadata_subject = aligned_metadata_subject[keep_cols].copy()

    df_for_validation = df_for_validation.merge(
        aligned_metadata_subject,
        left_on="Subject_ID",
        right_on=merge_key,
        how="left",
    )
    df_for_validation = coalesce_meta_columns(df_for_validation)
else:
    print("Warning: no reliable merge key found. Metadata columns will not be merged.")

print("Merged validation df rows:", len(df_for_validation))

if COHORT_NAME == "HABS":
    df_for_validation = merge_habs_clinical_columns(
        df_for_validation,
        HABS_CLINICAL_PATH
    )

for c in df_for_validation.columns:
    if c != "Subject_ID":
        try:
            df_for_validation[c] = pd.to_numeric(df_for_validation[c], errors="ignore")
        except Exception:
            pass

for c in df_for_validation.columns:
    if c == "Subject_ID":
        continue
    if pd.api.types.is_numeric_dtype(df_for_validation[c]):
        df_for_validation[c] = clean_numeric_with_sentinels(df_for_validation[c])


# =========================================================
# DEFINE CANDIDATE VARIABLES
# =========================================================
candidate_cognition_vars = [
    "MMSE_total",
    "MOCA_total_corrected",
    "MOCA_total",
    "ADAS_total",
    "CDGLOBAL",
    "CDRSB",
    "cognition_composite",
    "Memory_Composite",
    "Executive_Function_Composite",
    "Processing_Speed_Composite",
    "Language_Composite",
    "Visuospatial_Composite",
    "Global_Cognition_Composite",
    "Memory_Composite_resid",
    "Executive_Function_Composite_resid",
    "Processing_Speed_Composite_resid",
    "Language_Composite_resid",
    "Visuospatial_Composite_resid",
    "Global_Cognition_Composite_resid",
]

candidate_imaging_vars = [
    "Clustering_Coeff",
    "Path_Length",
    "Global_Efficiency",
    "Local_Efficiency",

    "HC_Clustering_Coeff",
    "HC_Path_Length",
    "Hippocampus_Clustering_Coeff",
    "Hippocampus_Path_Length",
    "Left_HC_Clustering_Coeff",
    "Right_HC_Clustering_Coeff",
    "Left_HC_Path_Length",
    "Right_HC_Path_Length",

    "FA_mean",
    "FA_median",
    "Volume_mean",
    "Volume_median",

    "Hippocampus_Total_pct",
    "Left_Hippocampus_pct",
    "Right_Hippocampus_pct",
    "Hippocampus_FA_Mean",
    "Hippocampus_FA_Total",
    "Left_Hippocampus_FA",
    "Right_Hippocampus_FA",
    "Total_Brain_volume",

    "ABETA42",
    "ABETA40",
    "TAU",
    "PTAU",
    "PLASMA_PTAU217",
    "GFAP",
    "NfL",
    "amyloid_42",
    "amyloid_40",
    "tau_total",
    "ptau",
    "ptau217",
    "gfap",
    "nfl",

    "BMI",
    "bmi",
    "OM_BMI",

    "VSBPSYS",
    "VSBPDIA",
    "VSPULSE",
    "BPSYS_AVG",
    "BPDIA_AVG",
    "bp_sys",
    "bp_dia",
    "pulse",
    "pulse_pressure",
    "MAP",

    "BW_Glucose_y",
    "Glucose",
    "glucose",
    "fasting_glucose",
    "BW_HBA1c_y",
    "HbA1c",
    "hba1c",

    "BW_CholTotal_y",
    "BW_HDLChol_y",
    "BW_LDLchol_y",
    "BW_Triglycerides_y",
    "CholTotal",
    "HDL",
    "LDL",
    "Triglycerides",
    "chol_total",
    "hdl",
    "ldl",
    "triglycerides",

    "ATN_composite",
]

auto_graph_metric_cols = find_existing_columns_by_patterns(
    df_for_validation,
    patterns=[
        "clustering",
        "cluster_coeff",
        "clustering_coeff",
        "path_length",
        "path length",
        "charpath",
        "characteristic_path",
        "hippocampus",
        "hc_",
        "glucose",
        "hba1c",
        "chol",
        "ldl",
        "hdl",
        "triglycer",
        "systolic",
        "diastolic",
        "pulse",
        "map",
        "blood_pressure",
        "bp_",
        "bmi",
    ],
)

candidate_imaging_vars = unique_preserve_order(candidate_imaging_vars + auto_graph_metric_cols)

COGNITION_VARS = keep_existing_cols(df_for_validation, candidate_cognition_vars)
IMAGING_VARS = keep_existing_cols(df_for_validation, candidate_imaging_vars)

COGNITION_VARS = [
    c for c in COGNITION_VARS
    if c in df_for_validation.columns and clean_numeric_with_sentinels(df_for_validation[c]).notna().sum() >= 3
]

IMAGING_VARS = [
    c for c in IMAGING_VARS
    if c in df_for_validation.columns and clean_numeric_with_sentinels(df_for_validation[c]).notna().sum() >= 3
]

print("\nFINAL COGNITION_VARS:")
print(COGNITION_VARS)
print("\nFINAL IMAGING_VARS:")
print(IMAGING_VARS)

graph_metric_debug = [
    c for c in IMAGING_VARS
    if any(tok in c.lower() for tok in ["clustering", "path", "charpath", "hippocampus", "hc_"])
]
print("\nGRAPH / HC METRICS SELECTED FOR VALIDATION:")
print(graph_metric_debug)

candidate_group_cols = [
    "Research Group",
    "Diagnosis",
    "DX",
    "DX_bl",
    "Group",
    "group_status",
    "DEMENTED",
    "Diagnostic_Group",
    "NORMCOG",
]

GROUP_COL = None
for gc in candidate_group_cols:
    if gc in df_for_validation.columns:
        non_na_groups = df_for_validation[gc].dropna()
        if non_na_groups.nunique() >= 2:
            GROUP_COL = gc
            break

print("Selected GROUP_COL:", GROUP_COL)


# =========================================================
# CREATE RAW-CLEAN AND Z-CLEAN VERSIONS
# =========================================================
BASE_VALIDATION_VARS = unique_preserve_order(
    [validation_brain_metric_col] + COGNITION_VARS + IMAGING_VARS
)

df_for_validation, derived_clean_cols = add_clean_and_z_versions(
    df_for_validation,
    BASE_VALIDATION_VARS
)

brain_metric_raw = f"{validation_brain_metric_col}_raw_clean"
brain_metric_z = f"{validation_brain_metric_col}_z_clean"

RAW_CLEAN_VARS = [f"{c}_raw_clean" for c in BASE_VALIDATION_VARS if f"{c}_raw_clean" in df_for_validation.columns]
Z_CLEAN_VARS = [f"{c}_z_clean" for c in BASE_VALIDATION_VARS if f"{c}_z_clean" in df_for_validation.columns]

print("\nRAW CLEAN VARS:")
print(RAW_CLEAN_VARS)

print("\nZ CLEAN VARS:")
print(Z_CLEAN_VARS)

df_for_validation.to_csv(os.path.join(val_outdir, "subject_level_validation_input.csv"), index=False)


# =========================================================
# MAIN VALIDATION OUTPUTS
# =========================================================
validation_plot_log = []

for metric_col, scale_tag in [(brain_metric_raw, "raw_clean"), (brain_metric_z, "z_clean")]:
    if metric_col in df_for_validation.columns:
        metric_base = strip_clean_suffix(metric_col)
        scale_short = "raw" if scale_tag == "raw_clean" else "z"
        hist_name = f"{sanitize_filename(metric_base)}_histogram_{scale_short}.png"
        ok, reason = save_histogram_with_stats(
            values=pd.to_numeric(df_for_validation[metric_col], errors="coerce").values,
            out_png=os.path.join(val_outdir, hist_name),
            title=f"{COHORT_NAME}: {metric_col} distribution ({scale_tag})",
            xlabel=metric_col,
        )
        validation_plot_log.append({
            "plot": hist_name,
            "status": "saved" if ok else "skipped",
            "reason": reason,
            "scale": scale_tag,
        })

all_validation_pairs = []

for var in unique_preserve_order(COGNITION_VARS + IMAGING_VARS):
    raw_var = f"{var}_raw_clean"
    z_var = f"{var}_z_clean"

    if raw_var in df_for_validation.columns and brain_metric_raw in df_for_validation.columns:
        all_validation_pairs.append((raw_var, brain_metric_raw, "raw_clean"))

    if z_var in df_for_validation.columns and brain_metric_z in df_for_validation.columns:
        all_validation_pairs.append((z_var, brain_metric_z, "z_clean"))

for x_col, y_col, scale_tag in all_validation_pairs:
    if x_col == y_col:
        continue

    ok0, reason0 = valid_scatter_xy(df_for_validation, x_col, y_col)
    x_base = strip_clean_suffix(x_col)
    y_base = strip_clean_suffix(y_col)

    scale_short = "raw" if scale_tag == "raw_clean" else "z"
    scale_label = "raw" if scale_tag == "raw_clean" else "z-scored"

    fname = f"{sanitize_filename(y_base)}_vs_{sanitize_filename(x_base)}_{scale_short}.png"

    if ok0:
        ok, reason = save_correlation_scatter(
            df=df_for_validation,
            x_col=x_col,
            y_col=y_col,
            out_png=os.path.join(val_outdir, fname),
            method=CORR_METHOD,
            title=f"{COHORT_NAME}: {y_base} vs {x_base} ({scale_label})"
        )
    else:
        ok, reason = False, reason0
        
    validation_plot_log.append({
        "plot": fname,
        "status": "saved" if ok else "skipped",
        "reason": reason,
        "scale": scale_tag,
    })

correlation_rows = []

for x_col, y_col, scale_tag in all_validation_pairs:
    if x_col == y_col:
        continue

    ok0, tmp = valid_scatter_xy(df_for_validation, x_col, y_col)
    if not ok0:
        correlation_rows.append({
            "brain_metric": y_col,
            "variable": x_col,
            "scale": scale_tag,
            "status": "skipped",
            "reason": tmp,
            "n": np.nan,
            "r": np.nan,
            "p": np.nan,
            "r2": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
        })
        continue

    metrics = compute_scatter_metrics(
        x=tmp[x_col].values,
        y=tmp[y_col].values,
        corr_method=CORR_METHOD,
        use_identity_r2=False,
    )

    correlation_rows.append({
        "brain_metric": y_col,
        "variable": x_col,
        "scale": scale_tag,
        "status": "ok",
        "reason": None,
        **metrics
    })

correlation_stats_path = os.path.join(
    val_outdir,
    f"{sanitize_filename(validation_brain_metric_col)}_correlation_stats_raw_and_z.csv"
)
pd.DataFrame(correlation_rows).to_csv(correlation_stats_path, index=False)

if GROUP_COL is not None:
    for value_col, scale_tag in [(brain_metric_raw, "raw_clean"), (brain_metric_z, "z_clean")]:
        if value_col in df_for_validation.columns:
            value_base = strip_clean_suffix(value_col)
            scale_short = "raw" if scale_tag == "raw_clean" else "z"
            fname = f"{sanitize_filename(value_base)}_by_{sanitize_filename(GROUP_COL)}_{scale_short}.png"
            ok, reason = save_boxplot_with_stats(
                df=df_for_validation,
                group_col=GROUP_COL,
                value_col=value_col,
                out_png=os.path.join(val_outdir, fname),
                title=f"{COHORT_NAME}: {value_col} by {GROUP_COL} ({scale_tag})",
            )
            validation_plot_log.append({
                "plot": fname,
                "status": "saved" if ok else "skipped",
                "reason": reason,
                "scale": scale_tag,
            })


# =========================================================
# ROC / AUC
# =========================================================
roc_plot_log = []

print("\n===== ROC DEBUG =====")
print("APOE col detected:", find_apoe_carrier_column(df_for_validation))
print("Cognition col detected:", find_cognition_status_column(df_for_validation))

for score_col, scale_tag in [(brain_metric_raw, "raw_clean"), (brain_metric_z, "z_clean")]:
    if score_col not in df_for_validation.columns:
        continue

    print("score_col:", score_col)

    apoe_col = find_apoe_carrier_column(df_for_validation)

    if apoe_col is not None:
        y_apoe = derive_apoe4_carrier(df_for_validation[apoe_col])
        score_series = clean_numeric_with_sentinels(df_for_validation[score_col])
        score_used, flipped = maybe_flip_auc_direction(y_apoe, score_series)

        out_name = f"roc_{sanitize_filename(score_col)}_vs_apoe4_carriage_{scale_tag}.png"
        ok, info = save_roc_curve_from_score(
            y_true=y_apoe,
            y_score=score_used,
            out_png=os.path.join(val_outdir, out_name),
            title=f"ROC Curve: APOE4 carriage ({scale_tag})",
            score_name=score_col + (" (flipped)" if flipped else ""),
        )

        print("\n===== APOE DEBUG =====")
        print("APOE column selected:", apoe_col)
        if apoe_col is not None:
            print("Raw APOE unique values:", pd.Series(df_for_validation[apoe_col]).dropna().astype(str).unique()[:20])
            print("Derived APOE4 counts:")
            print(derive_apoe4_carrier(df_for_validation[apoe_col]).value_counts(dropna=False))
            
        roc_plot_log.append({
            "plot": out_name,
            "status": "saved" if ok else "skipped",
            "target": "APOE4_carriage",
            "source_column": apoe_col,
            "score_column": score_col,
            "scale": scale_tag,
            "flipped_score": flipped,
            "reason": None if ok else info,
        })
    else:
        roc_plot_log.append({
            "plot": f"roc_{sanitize_filename(score_col)}_vs_apoe4_carriage_{scale_tag}.png",
            "status": "skipped",
            "target": "APOE4_carriage",
            "source_column": None,
            "score_column": score_col,
            "scale": scale_tag,
            "flipped_score": False,
            "reason": "No APOE/APOE4/genotype column found",
        })

    y_cog, cog_col, cog_desc = derive_binary_cognitive_status(df_for_validation)

    if y_cog is not None and cog_col is not None:
        score_series = clean_numeric_with_sentinels(df_for_validation[score_col])
        score_used, flipped = maybe_flip_auc_direction(y_cog, score_series)

        out_name = f"roc_{sanitize_filename(score_col)}_vs_cognitive_status_{scale_tag}.png"
        ok, info = save_roc_curve_from_score(
            y_true=y_cog,
            y_score=score_used,
            out_png=os.path.join(val_outdir, out_name),
            title=f"ROC Curve: {cog_desc} ({scale_tag})",
            score_name=score_col + (" (flipped)" if flipped else ""),
        )

        roc_plot_log.append({
            "plot": out_name,
            "status": "saved" if ok else "skipped",
            "target": "cognitive_status",
            "source_column": cog_col,
            "score_column": score_col,
            "scale": scale_tag,
            "flipped_score": flipped,
            "reason": None if ok else info,
        })
    else:
        roc_plot_log.append({
            "plot": f"roc_{sanitize_filename(score_col)}_vs_cognitive_status_{scale_tag}.png",
            "status": "skipped",
            "target": "cognitive_status",
            "source_column": None,
            "score_column": score_col,
            "scale": scale_tag,
            "flipped_score": False,
            "reason": "No cognitive status column could be binarized",
        })

# =========================================================
# HABS-SPECIFIC ROC / AUC
# =========================================================
if COHORT_NAME == "HABS":
    habs_auc_targets = [
        ("CDX_Diabetes", "CDX_Diabetes"),
        ("CDX_Hypertension", "CDX_Hypertension"),
        ("IMH_HighBP", "IMH_HighBP"),
    ]

    for score_col, scale_tag in [(brain_metric_raw, "raw_clean"), (brain_metric_z, "z_clean")]:
        if score_col not in df_for_validation.columns:
            continue

        score_series = clean_numeric_with_sentinels(df_for_validation[score_col])

        for col_name, label in habs_auc_targets:
            if col_name not in df_for_validation.columns:
                roc_plot_log.append({
                    "plot": f"roc_{sanitize_filename(score_col)}_vs_{sanitize_filename(label)}_{scale_tag}.png",
                    "status": "skipped",
                    "target": label,
                    "source_column": col_name,
                    "score_column": score_col,
                    "scale": scale_tag,
                    "flipped_score": False,
                    "reason": f"missing column {col_name}",
                })
                continue

            y_bin = derive_binary_0_1(df_for_validation[col_name])
            score_used, flipped = maybe_flip_auc_direction(y_bin, score_series)

            out_name = f"roc_{sanitize_filename(score_col)}_vs_{sanitize_filename(label)}_{'raw' if scale_tag == 'raw_clean' else 'z'}.png"
            ok, info = save_roc_curve_from_score(
                y_true=y_bin,
                y_score=score_used,
                out_png=os.path.join(val_outdir, out_name),
                title=f"ROC Curve: {label} ({'raw' if scale_tag == 'raw_clean' else 'z-scored'})",
                score_name=score_col + (" (flipped)" if flipped else ""),
            )

            roc_plot_log.append({
                "plot": out_name,
                "status": "saved" if ok else "skipped",
                "target": label,
                "source_column": col_name,
                "score_column": score_col,
                "scale": scale_tag,
                "flipped_score": flipped,
                "reason": None if ok else info,
            })

        # BMI >= 27
        bmi_source_col = None
        for candidate in ["OM_BMI", "BMI", "bmi"]:
            if candidate in df_for_validation.columns:
                bmi_source_col = candidate
                break

        if bmi_source_col is not None:
            y_bmi27 = derive_bmi_27_binary(df_for_validation[bmi_source_col])
            score_used, flipped = maybe_flip_auc_direction(y_bmi27, score_series)

            out_name = f"roc_{sanitize_filename(score_col)}_vs_BMI_ge_27_{'raw' if scale_tag == 'raw_clean' else 'z'}.png"
            ok, info = save_roc_curve_from_score(
                y_true=y_bmi27,
                y_score=score_used,
                out_png=os.path.join(val_outdir, out_name),
                title=f"ROC Curve: BMI >= 27 ({'raw' if scale_tag == 'raw_clean' else 'z-scored'})",
                score_name=score_col + (" (flipped)" if flipped else ""),
            )

            roc_plot_log.append({
                "plot": out_name,
                "status": "saved" if ok else "skipped",
                "target": "BMI_ge_27",
                "source_column": bmi_source_col,
                "score_column": score_col,
                "scale": scale_tag,
                "flipped_score": flipped,
                "reason": None if ok else info,
            })
        else:
            roc_plot_log.append({
                "plot": f"roc_{sanitize_filename(score_col)}_vs_BMI_ge_27_{'raw' if scale_tag == 'raw_clean' else 'z'}.png",
                "status": "skipped",
                "target": "BMI_ge_27",
                "source_column": None,
                "score_column": score_col,
                "scale": scale_tag,
                "flipped_score": False,
                "reason": "No BMI/OM_BMI column found",
            })
            
# =========================================================
# HABS LONGITUDINAL DELTA ANALYSIS (ONLY HABS)
# =========================================================
if COHORT_NAME == "HABS":
    print("\n" + "=" * 80)
    print("HABS LONGITUDINAL DELTA ANALYSIS")
    print("=" * 80)

    long_outdir = os.path.join(val_outdir, "longitudinal_habs")
    os.makedirs(long_outdir, exist_ok=True)

    df_long = df_for_validation.copy()

    # -----------------------------------------------------
    # choose longitudinal source column
    # -----------------------------------------------------
    longitudinal_id_col = None
    preferred_long_cols = ["runno", "connectome_key", "Subject_ID", "graph_id"]

    for c in preferred_long_cols:
        if c in df_long.columns:
            sample_vals = df_long[c].dropna().astype(str).head(20).tolist()
            if any(re.search(r"H\d{4}_y\d+", v, flags=re.IGNORECASE) for v in sample_vals):
                longitudinal_id_col = c
                break

    if longitudinal_id_col is None:
        for c in preferred_long_cols:
            if c in df_long.columns:
                longitudinal_id_col = c
                break

    longitudinal_plot_log = []
    longitudinal_stats_rows = []

    if longitudinal_id_col is None:
        print("No usable longitudinal ID column found for HABS. Skipping longitudinal analysis.")

        pd.DataFrame(longitudinal_plot_log).to_csv(
            os.path.join(long_outdir, "longitudinal_plot_log.csv"), index=False
        )
        pd.DataFrame(longitudinal_stats_rows).to_csv(
            os.path.join(long_outdir, "habs_longitudinal_delta_stats.csv"), index=False
        )
        pd.DataFrame([
            {"key": "COHORT_NAME", "value": COHORT_NAME},
            {"key": "longitudinal_id_col", "value": None},
            {"key": "reason", "value": "No usable longitudinal ID column found"},
        ]).to_csv(
            os.path.join(long_outdir, "habs_longitudinal_summary.csv"),
            index=False
        )

    else:
        # -------------------------------------------------
        # parse longitudinal id
        # -------------------------------------------------
        df_long["_long_id"] = df_long[longitudinal_id_col].astype(str).str.strip()
        df_long["_long_id"] = df_long["_long_id"].str.replace("_Y", "_y", regex=False)

        df_long["habs_subject_base"] = df_long["_long_id"].map(extract_habs_subject_base)
        df_long["habs_visit_num"] = df_long["_long_id"].map(extract_habs_visit_num)

        print("Longitudinal source column:", longitudinal_id_col)
        print("Sample longitudinal IDs:", df_long["_long_id"].head(10).tolist())
        print("Sample subject bases:", df_long["habs_subject_base"].head(10).tolist())
        print("Sample visit nums:", df_long["habs_visit_num"].head(10).tolist())
        print("Subjects with parsed base ID:", df_long["habs_subject_base"].notna().sum())
        print("Rows with parsed visit num:", df_long["habs_visit_num"].notna().sum())

        # -------------------------------------------------
        # use the actual selected metric, not hard-coded cBAG
        # -------------------------------------------------
        long_brain_metric = validation_brain_metric_col

        needed_for_long = ["habs_subject_base", "habs_visit_num", long_brain_metric]
        existing_needed = [c for c in needed_for_long if c in df_long.columns]
        df_long = df_long.dropna(subset=existing_needed).copy()

        # -------------------------------------------------
        # visit counts
        # -------------------------------------------------
        visit_counts = (
            df_long.groupby("habs_subject_base")["habs_visit_num"]
            .nunique()
            .reset_index(name="n_visits")
        )

        print("\nVisit count distribution:")
        if len(visit_counts) > 0:
            print(visit_counts["n_visits"].value_counts().sort_index())
        else:
            print("No visit counts available")

        two_visit_subjects = visit_counts.loc[
            visit_counts["n_visits"] == 2, "habs_subject_base"
        ].tolist()

        df_long_2v = df_long[df_long["habs_subject_base"].isin(two_visit_subjects)].copy()

        print("Subjects with exactly 2 visits:", len(two_visit_subjects))
        print("Rows in exactly-2-visit subset:", len(df_long_2v))

        if len(df_long_2v) == 0:
            print("No HABS subjects with exactly 2 visits. Saving empty longitudinal outputs.")

            pd.DataFrame(longitudinal_plot_log).to_csv(
                os.path.join(long_outdir, "longitudinal_plot_log.csv"), index=False
            )
            pd.DataFrame(longitudinal_stats_rows).to_csv(
                os.path.join(long_outdir, "habs_longitudinal_delta_stats.csv"), index=False
            )
            pd.DataFrame([
                {"key": "COHORT_NAME", "value": COHORT_NAME},
                {"key": "longitudinal_id_col", "value": longitudinal_id_col},
                {"key": "long_brain_metric", "value": long_brain_metric},
                {"key": "n_subjects_with_exactly_2_visits", "value": 0},
                {"key": "n_rows_in_2visit_subset", "value": 0},
                {"key": "reason", "value": "No subjects with exactly 2 visits after filtering"},
            ]).to_csv(
                os.path.join(long_outdir, "habs_longitudinal_summary.csv"),
                index=False
            )

        else:
            # -------------------------------------------------
            # order visits
            # -------------------------------------------------
            df_long_2v = df_long_2v.sort_values(
                ["habs_subject_base", "habs_visit_num"]
            ).copy()

            longitudinal_vars = [
                long_brain_metric,
                "Hippocampus_FA_Mean",
                "Hippocampus_Total_pct",
                "Left_Hippocampus_FA",
                "Right_Hippocampus_FA",
                "Left_Hippocampus_pct",
                "Right_Hippocampus_pct",
                "Total_Brain_volume",
            ]
            longitudinal_vars = [c for c in longitudinal_vars if c in df_long_2v.columns]

            optional_keep_cols = [
                longitudinal_id_col,
                "Subject_ID",
                "connectome_key",
                "match_id",
                "subject_id",
                "runno",
                "age",
                "Real_Age",
                "Age",
                "group_status",
                "NORMCOG",
            ]
            optional_keep_cols = [c for c in optional_keep_cols if c in df_long_2v.columns]

            keep_cols = unique_preserve_order(
                ["habs_subject_base", "habs_visit_num"] + optional_keep_cols + longitudinal_vars
            )

            df_long_2v_small = df_long_2v[keep_cols].copy()

            # first and second visit per subject
            first_visits = (
                df_long_2v_small.groupby("habs_subject_base", as_index=False)
                .nth(0)
                .reset_index(drop=True)
            )
            second_visits = (
                df_long_2v_small.groupby("habs_subject_base", as_index=False)
                .nth(1)
                .reset_index(drop=True)
            )

            first_visits = first_visits.add_suffix("_v1")
            second_visits = second_visits.add_suffix("_v2")

            delta_df = pd.concat([first_visits, second_visits], axis=1).copy()

            if "habs_subject_base_v1" in delta_df.columns:
                delta_df["habs_subject_base"] = delta_df["habs_subject_base_v1"]

            # -------------------------------------------------
            # deltas
            # -------------------------------------------------
            for c in longitudinal_vars:
                c_v1 = f"{c}_v1"
                c_v2 = f"{c}_v2"
                if c_v1 in delta_df.columns and c_v2 in delta_df.columns:
                    delta_df[f"delta_{c}"] = (
                        pd.to_numeric(delta_df[c_v2], errors="coerce") -
                        pd.to_numeric(delta_df[c_v1], errors="coerce")
                    )

            if "habs_visit_num_v1" in delta_df.columns and "habs_visit_num_v2" in delta_df.columns:
                delta_df["delta_visit_num"] = (
                    pd.to_numeric(delta_df["habs_visit_num_v2"], errors="coerce") -
                    pd.to_numeric(delta_df["habs_visit_num_v1"], errors="coerce")
                )

            age_v1_col = None
            age_v2_col = None
            for base_age_col in ["age", "Real_Age", "Age"]:
                if f"{base_age_col}_v1" in delta_df.columns and f"{base_age_col}_v2" in delta_df.columns:
                    age_v1_col = f"{base_age_col}_v1"
                    age_v2_col = f"{base_age_col}_v2"
                    break

            if age_v1_col is not None and age_v2_col is not None:
                delta_df["delta_age"] = (
                    pd.to_numeric(delta_df[age_v2_col], errors="coerce") -
                    pd.to_numeric(delta_df[age_v1_col], errors="coerce")
                )

            delta_table_path = os.path.join(long_outdir, "habs_longitudinal_delta_table.csv")
            delta_df.to_csv(delta_table_path, index=False)
            print("Saved:", delta_table_path)

            # -------------------------------------------------
            # delta pairs
            # -------------------------------------------------
            delta_metric = f"delta_{long_brain_metric}"

            delta_pairs = [
                ("delta_Hippocampus_FA_Mean", delta_metric),
                ("delta_Hippocampus_Total_pct", delta_metric),
                ("delta_Left_Hippocampus_FA", delta_metric),
                ("delta_Right_Hippocampus_FA", delta_metric),
                ("delta_Left_Hippocampus_pct", delta_metric),
                ("delta_Right_Hippocampus_pct", delta_metric),
                ("delta_Total_Brain_volume", delta_metric),
            ]
            delta_pairs = [(x, y) for x, y in delta_pairs if x in delta_df.columns and y in delta_df.columns]

            print("\n===== LONGITUDINAL DELTA DEBUG =====")
            print("delta_df shape:", delta_df.shape)
            print("delta_df columns containing 'delta_':")
            print([c for c in delta_df.columns if c.startswith("delta_")])

            print("\nDelta pairs to attempt:")
            print(delta_pairs)

            for c in [col for col in delta_df.columns if col.startswith("delta_")]:
                vals = pd.to_numeric(delta_df[c], errors="coerce")
                print(
                    f"{c}: n_nonnull={vals.notna().sum()}, "
                    f"n_unique={vals.dropna().nunique()}, "
                    f"min={vals.min()}, max={vals.max()}"
                )

            # -------------------------------------------------
            # plots
            # -------------------------------------------------
            for x_col, y_col in delta_pairs:
                ok0, tmp = valid_delta_scatter_xy(delta_df, x_col, y_col)

                print(f"\nTrying longitudinal plot: {y_col} vs {x_col}")
                print("valid:", ok0)
                if not ok0:
                    print("reason:", tmp)
                else:
                    print("n rows for plot:", len(tmp))

                fname = f"{sanitize_filename(y_col)}_vs_{sanitize_filename(x_col)}.png"

                if ok0:
                    plt.figure(figsize=(8, 6))
                    ax = plt.gca()
                    ax.scatter(tmp[x_col].values, tmp[y_col].values, alpha=0.7, edgecolors="k")

                    try:
                        lr = linregress(tmp[x_col].values, tmp[y_col].values)
                        xx = np.linspace(np.nanmin(tmp[x_col].values), np.nanmax(tmp[x_col].values), 100)
                        yy = lr.slope * xx + lr.intercept
                        ax.plot(xx, yy, linestyle="--")
                    except Exception:
                        pass

                    metrics = compute_scatter_metrics(
                        x=tmp[x_col].values,
                        y=tmp[y_col].values,
                        corr_method=CORR_METHOD,
                        use_identity_r2=False,
                    )
                    add_metrics_box(ax, metrics, include_error_metrics=False)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f"{COHORT_NAME}: {y_col} vs {x_col} (2-visit delta)")
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(long_outdir, fname), dpi=300, bbox_inches="tight")
                    plt.close()

                    ok, reason = True, None
                else:
                    ok, reason = False, tmp

                longitudinal_plot_log.append({
                    "plot": fname,
                    "status": "saved" if ok else "skipped",
                    "reason": reason,
                    "x_col": x_col,
                    "y_col": y_col,
                })

                if ok0:
                    metrics = compute_scatter_metrics(
                        x=tmp[x_col].values,
                        y=tmp[y_col].values,
                        corr_method=CORR_METHOD,
                        use_identity_r2=False,
                    )
                    longitudinal_stats_rows.append({
                        "x_col": x_col,
                        "y_col": y_col,
                        "status": "ok",
                        "reason": None,
                        **metrics
                    })
                else:
                    longitudinal_stats_rows.append({
                        "x_col": x_col,
                        "y_col": y_col,
                        "status": "skipped",
                        "reason": tmp,
                        "n": np.nan,
                        "r": np.nan,
                        "p": np.nan,
                        "r2": np.nan,
                        "mae": np.nan,
                        "rmse": np.nan,
                        "slope": np.nan,
                        "intercept": np.nan,
                    })

            # optional delta metric vs delta age
            if "delta_age" in delta_df.columns and delta_metric in delta_df.columns:
                ok0, tmp = valid_delta_scatter_xy(delta_df, "delta_age", delta_metric)
                fname = f"{sanitize_filename(delta_metric)}_vs_delta_age.png"

                print(f"\nTrying longitudinal plot: {delta_metric} vs delta_age")
                print("valid:", ok0)
                if not ok0:
                    print("reason:", tmp)
                else:
                    print("n rows for plot:", len(tmp))

                if ok0:
                    plt.figure(figsize=(8, 6))
                    ax = plt.gca()
                    ax.scatter(tmp["delta_age"].values, tmp[delta_metric].values, alpha=0.7, edgecolors="k")

                    try:
                        lr = linregress(tmp["delta_age"].values, tmp[delta_metric].values)
                        xx = np.linspace(np.nanmin(tmp["delta_age"].values), np.nanmax(tmp["delta_age"].values), 100)
                        yy = lr.slope * xx + lr.intercept
                        ax.plot(xx, yy, linestyle="--")
                    except Exception:
                        pass

                    metrics = compute_scatter_metrics(
                        x=tmp["delta_age"].values,
                        y=tmp[delta_metric].values,
                        corr_method=CORR_METHOD,
                        use_identity_r2=False,
                    )
                    add_metrics_box(ax, metrics, include_error_metrics=False)
                    ax.set_xlabel("delta_age")
                    ax.set_ylabel(delta_metric)
                    ax.set_title(f"{COHORT_NAME}: {delta_metric} vs delta_age (2-visit delta)")
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(long_outdir, fname), dpi=300, bbox_inches="tight")
                    plt.close()

                    ok, reason = True, None
                else:
                    ok, reason = False, tmp

                longitudinal_plot_log.append({
                    "plot": fname,
                    "status": "saved" if ok else "skipped",
                    "reason": reason,
                    "x_col": "delta_age",
                    "y_col": delta_metric,
                })

                if ok0:
                    metrics = compute_scatter_metrics(
                        x=tmp["delta_age"].values,
                        y=tmp[delta_metric].values,
                        corr_method=CORR_METHOD,
                        use_identity_r2=False,
                    )
                    longitudinal_stats_rows.append({
                        "x_col": "delta_age",
                        "y_col": delta_metric,
                        "status": "ok",
                        "reason": None,
                        **metrics
                    })
                else:
                    longitudinal_stats_rows.append({
                        "x_col": "delta_age",
                        "y_col": delta_metric,
                        "status": "skipped",
                        "reason": tmp,
                        "n": np.nan,
                        "r": np.nan,
                        "p": np.nan,
                        "r2": np.nan,
                        "mae": np.nan,
                        "rmse": np.nan,
                        "slope": np.nan,
                        "intercept": np.nan,
                    })

            longitudinal_log_path = os.path.join(long_outdir, "longitudinal_plot_log.csv")
            pd.DataFrame(longitudinal_plot_log).to_csv(longitudinal_log_path, index=False)
            print("Saved:", longitudinal_log_path)

            longitudinal_stats_path = os.path.join(long_outdir, "habs_longitudinal_delta_stats.csv")
            pd.DataFrame(longitudinal_stats_rows).to_csv(longitudinal_stats_path, index=False)
            print("Saved:", longitudinal_stats_path)

            pd.DataFrame([
                {"key": "COHORT_NAME", "value": COHORT_NAME},
                {"key": "longitudinal_id_col", "value": longitudinal_id_col},
                {"key": "long_brain_metric", "value": long_brain_metric},
                {"key": "n_subjects_with_exactly_2_visits", "value": len(two_visit_subjects)},
                {"key": "n_rows_in_2visit_subset", "value": len(df_long_2v)},
                {"key": "n_delta_rows", "value": len(delta_df)},
                {"key": "delta_table_path", "value": delta_table_path},
                {"key": "longitudinal_stats_path", "value": longitudinal_stats_path},
            ]).to_csv(
                os.path.join(long_outdir, "habs_longitudinal_summary.csv"),
                index=False
            )
            print("Saved:", os.path.join(long_outdir, "habs_longitudinal_summary.csv"))
# =========================================================
# SAVE LOGS AND SUMMARY
# =========================================================
validation_log_df = pd.DataFrame(validation_plot_log)
validation_log_df.to_csv(os.path.join(val_outdir, "image_generation_log.csv"), index=False)

roc_log_df = pd.DataFrame(roc_plot_log)
roc_log_df.to_csv(os.path.join(val_outdir, "roc_image_generation_log.csv"), index=False)

all_validation_logs = pd.concat(
    [validation_log_df, roc_log_df],
    axis=0,
    ignore_index=True,
    sort=False,
)
all_validation_logs.to_csv(os.path.join(val_outdir, "all_image_generation_log.csv"), index=False)

skipped_df = all_validation_logs[all_validation_logs["status"] != "saved"].copy()
skipped_df.to_csv(os.path.join(val_outdir, "skipped_plots.csv"), index=False)

with open(os.path.join(val_outdir, "skipped_plots.txt"), "w", encoding="utf-8") as f:
    if len(skipped_df) == 0:
        f.write("No skipped validation plots.\n")
    else:
        f.write("Skipped validation plots:\n")
        for _, row in skipped_df.iterrows():
            f.write(f"- {row['plot']}: {row['reason']}\n")

summary_rows = [
    {"key": "COHORT_NAME", "value": COHORT_NAME},
    {"key": "results_dir", "value": results_dir},
    {"key": "val_outdir", "value": val_outdir},
    {"key": "cv_preds_path", "value": cv_preds_path},
    {"key": "global_oof_path", "value": global_oof_path},
    {"key": "full_cohort_preds_path", "value": full_cohort_preds_path},
    {"key": "metadata_path", "value": metadata_path},
    {"key": "metadata_all_path", "value": metadata_all_path},
    {"key": "validation_brain_metric_col", "value": validation_brain_metric_col},
    {"key": "brain_metric_raw", "value": brain_metric_raw},
    {"key": "brain_metric_z", "value": brain_metric_z},
    {"key": "n_validation_rows", "value": len(df_for_validation)},
    {"key": "n_validation_columns", "value": len(df_for_validation.columns)},
    {"key": "group_col", "value": GROUP_COL},
    {"key": "n_cognition_vars", "value": len(COGNITION_VARS)},
    {"key": "n_imaging_vars", "value": len(IMAGING_VARS)},
    {"key": "correlation_stats_path", "value": correlation_stats_path},
]

pd.DataFrame(summary_rows).to_csv(
    os.path.join(val_outdir, f"{COHORT_LOWER}_validation_summary.csv"),
    index=False,
)

print("\nValidation finished.")
print("Check:")
print(" ", os.path.join(val_outdir, "subject_level_validation_input.csv"))
print(" ", correlation_stats_path)
print(" ", os.path.join(val_outdir, "image_generation_log.csv"))
print(" ", os.path.join(val_outdir, "roc_image_generation_log.csv"))
print(" ", os.path.join(val_outdir, "all_image_generation_log.csv"))
print(" ", os.path.join(val_outdir, "skipped_plots.csv"))
print(" ", os.path.join(val_outdir, f"{COHORT_LOWER}_validation_summary.csv"))