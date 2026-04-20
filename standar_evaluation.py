#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic Brain Age evaluation + validation pipeline
-------------------------------------------------
- No carga el modelo GNN
- No rehace inferencia
- Solo usa CSV/XLSX ya generados por training/post-training
- Genera evaluation y validation plots con métricas embebidas
- Funciona para distintas cohortes cambiando solo COHORT_NAME
- SIN SHAP / SIN interpretation
"""
import re
import os
import sys
import glob
import warnings

# =========================================================HABS
# MAKE CUSTOM UTILS IMPORTABLE
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)

for p in [SCRIPT_DIR, CODE_DIR]:
    if p not in sys.path:
        sys.path.append(p)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use("Agg")
matplotlib.set_loglevel("error")
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from evaluation_utils import run_standard_evaluation
from validation_utils import run_standard_validation

warnings.filterwarnings("ignore")


# =========================================================
# CONFIG
# =========================================================
WORK = os.environ["WORK"]

COHORT_NAME = "ADRC"   # <<< CAMBIA SOLO ESTO
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

eval_outdir = os.path.join(results_dir, "evaluation")
val_outdir = os.path.join(results_dir, "validation")

os.makedirs(eval_outdir, exist_ok=True)
os.makedirs(val_outdir, exist_ok=True)

BRAIN_METRIC_COL = "cBAG_global"
CORR_METHOD = "pearson"

if COHORT_NAME == "AD_DECODE":
    COHORT_FILE_STEM = "addecode"
else:
    COHORT_FILE_STEM = COHORT_NAME.lower()

COHORT_LOWER = COHORT_NAME.lower()
COHORT_UPPER = COHORT_NAME.upper()


# =========================================================
# HELPERS
# =========================================================
def keep_existing_cols(df, cols):
    return [c for c in cols if c in df.columns]


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
                "No encuentro Subject_ID en dataframe. "
                "Necesito Subject_ID, graph_id, PTID, connectome_key, "
                "connectome_id, subject_id, RID, ID, MRI_Exam o match_id."
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


def find_apoe_carrier_column(df):
    candidates = [
        "APOE4_carrier",
        "APOE4",
        "APOE",
        "apoe4_carrier",
        "apoe4",
        "genotype",
        "APOE_genotype",
    ]
    return first_existing_column(df, candidates)


def derive_apoe4_carrier(series):
    """
    Devuelve serie binaria:
    1 = carrier APOE4
    0 = non-carrier
    NaN = no interpretable
    """
    s = series.copy()

    # numérico ya binario
    s_num = pd.to_numeric(s, errors="coerce")
    uniq_num = set(s_num.dropna().unique().tolist())
    if len(uniq_num) > 0 and uniq_num.issubset({0, 1}):
        return s_num

    s_str = to_clean_str_series(s)

    out = pd.Series(np.nan, index=s.index, dtype=float)

    # casos obvios texto
    out[s_str.isin(["0", "NON-CARRIER", "NONCARRIER", "NEGATIVE", "FALSE", "NO"])] = 0
    out[s_str.isin(["1", "CARRIER", "POSITIVE", "TRUE", "YES"])] = 1

    out[s_str.isin(["E4-", "E4 NEG", "APOE4-", "NON E4", "NON-E4"])] = 0
    out[s_str.isin(["E4+", "E4 POS", "APOE4+", "E4 CARRIER", "APOE4 CARRIER"])] = 1
    # genotipos tipo 2/3, 3/4, e3/e4, 4/4...
    has_4 = s_str.str.contains(r"(^|[^0-9])4([^0-9]|$)", regex=True, na=False)
    genotype_like = s_str.str.contains(r"[234]/[234]|E[234]/E[234]|[234][ ]*/[ ]*[234]", regex=True, na=False)
    out[genotype_like & has_4] = 1
    out[genotype_like & (~has_4)] = 0

    # e4/e4, e3/e4, 3/4, 4/4, etc.
    explicit_carrier = s_str.str.contains(r"E?4\s*/\s*E?[234]|E?[234]\s*/\s*E?4|^4/4$|^3/4$|^2/4$", regex=True, na=False)
    explicit_noncarrier = s_str.str.contains(r"^2/2$|^2/3$|^3/3$|^E2/E2$|^E2/E3$|^E3/E3$", regex=True, na=False)

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
    """
    Devuelve:
    y_bin: 1 = cognitively impaired / MCI / AD / dementia / at-risk / case
           0 = control / CN / healthy / normal
    used_col: columna usada
    label_desc: descripción comparación
    """
    col = preferred_col if preferred_col is not None else find_cognition_status_column(df)
    if col is None:
        return None, None, None

    s = df[col]

    # -----------------------------------------------------
    # Binary direct columns
    # -----------------------------------------------------
    if col == "NORMCOG":
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().sum() > 0:
            y = pd.Series(np.nan, index=s.index, dtype=float)
            y[s_num == 1] = 0   # normal
            y[s_num == 0] = 1   # impaired
            return y, col, "Normal vs impaired"

    if col == "DEMENTED":
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().sum() > 0:
            y = pd.Series(np.nan, index=s.index, dtype=float)
            y[s_num == 0] = 0
            y[s_num == 1] = 1
            return y, col, "Non-demented vs demented"

    # -----------------------------------------------------
    # Text-based columns
    # -----------------------------------------------------
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

        # flexible matching
        if any(tok in val for tok in ["CN", "CONTROL", "HEALTHY", "NORMAL", "HC", "CU", "NORMCOG"]):
            y.loc[idx] = 0
            continue

        if any(tok in val for tok in ["MCI", "LMCI", "EMCI", "AD", "DEMENT", "ALZ", "IMPAIRED", "CASE", "PATIENT", "RISK"]):
            y.loc[idx] = 1
            continue

    return y, col, "Control/CN vs MCI/AD/impaired"


def save_roc_curve_from_score(y_true, y_score, out_png, title, score_name):
    y_true = pd.to_numeric(pd.Series(y_true), errors="coerce")
    y_score = pd.to_numeric(pd.Series(y_score), errors="coerce")

    mask = y_true.notna() & y_score.notna()
    y_true = y_true[mask].astype(int)
    y_score = y_score[mask].astype(float)

    if len(y_true) < 6:
        return False, "fewer than 6 complete rows"

    uniq = sorted(y_true.unique().tolist())
    if uniq != [0, 1]:
        return False, f"target is not binary: {uniq}"

    n0 = int((y_true == 0).sum())
    n1 = int((y_true == 1).sum())
    if n0 < 3 or n1 < 3:
        return False, f"insufficient class counts (n0={n0}, n1={n1})"

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


def maybe_flip_auc_direction(y_true, y_score):
    """
    Si AUC < 0.5, invierte el score para que la interpretación sea más estable.
    """
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


def _format_p_value(p):
    if pd.isna(p):
        return "nan"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


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


def clear_image_files(folder):
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.svg", "*.pdf"]
    for pat in patterns:
        for fp in glob.glob(os.path.join(folder, pat)):
            try:
                os.remove(fp)
            except Exception:
                pass


def remove_duplicate_plot_subdirs(folder):
    for sub in ["metric_plots", "correlations", "group_comparisons", "plots"]:
        subdir = os.path.join(folder, sub)
        if os.path.isdir(subdir):
            for fp in glob.glob(os.path.join(subdir, "*")):
                try:
                    os.remove(fp)
                except Exception:
                    pass


def safe_dict_keys(x):
    return list(x.keys()) if isinstance(x, dict) else []


def find_existing_file(candidates):
    for fp in candidates:
        if fp is not None and os.path.exists(fp):
            return fp
    return None


def numeric_series(s):
    return pd.to_numeric(s, errors="coerce")


def load_table_auto(path):
    if path is None:
        return None
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


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


def valid_scatter_xy(df, x_col, y_col):
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


def save_predicted_vs_real_plot(df, y_col, out_png, title, ylabel):
    ok, tmp = valid_scatter_xy(df, "Real_Age", y_col)
    if not ok:
        return False, tmp

    x = tmp["Real_Age"].values
    y = tmp[y_col].values

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    ax.scatter(x, y, alpha=0.7, edgecolors="k")

    mn = float(min(np.nanmin(x), np.nanmin(y)))
    mx = float(max(np.nanmax(x), np.nanmax(y)))
    ax.plot([mn, mx], [mn, mx], color="red", linestyle="dashed")

    metrics = compute_scatter_metrics(x=x, y=y, corr_method="pearson", use_identity_r2=False)
    add_metrics_box(ax, metrics, include_error_metrics=True)

    ax.set_xlabel("Real Age")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    return True, None


def save_bag_vs_age_plot(df, bag_col, out_png, title):
    ok, tmp = valid_scatter_xy(df, "Real_Age", bag_col)
    if not ok:
        return False, tmp

    x = tmp["Real_Age"].values
    y = tmp[bag_col].values

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.scatter(x, y, alpha=0.7, edgecolors="k")
    ax.axhline(0.0, linestyle="--")

    try:
        lr = linregress(x, y)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        yy = lr.slope * xx + lr.intercept
        ax.plot(xx, yy, linestyle="--")
    except Exception:
        pass

    metrics = compute_scatter_metrics(x=x, y=y, corr_method="pearson", use_identity_r2=False)
    add_metrics_box(ax, metrics, include_error_metrics=False)

    ax.set_xlabel("Real Age")
    ax.set_ylabel(bag_col)
    ax.set_title(title)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    return True, None

def extract_4digit_match_id(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    groups = re.findall(r"(\d+)", s)
    if len(groups) == 0:
        return np.nan
    digits = "".join(groups)
    return digits[-4:].zfill(4)


def build_regional_id_from_match_id(match_id, cohort_name):
    if pd.isna(match_id):
        return np.nan

    match_id = str(match_id).zfill(4)

    if cohort_name == "ADRC":
        return f"D{match_id}"
    if cohort_name == "HABS":
        return f"H{match_id}"
    if cohort_name == "ADNI":
        return f"R{match_id}"
    if cohort_name == "AD_DECODE":
        return f"S0{match_id}"
    return np.nan


def extract_connectome_identifiers_from_stats_col(colname, cohort_name):
    s = str(colname).strip()
    s_up = s.upper()

    if cohort_name == "ADNI":
        m = re.match(r"^(R\d+_Y\d+|R\d+_y\d+)$", s, flags=re.IGNORECASE)
        if not m:
            return None, None, None
        connectome_key = re.sub(r"_Y", "_y", m.group(1))
        match_id = extract_4digit_match_id(connectome_key)
        regional_id = build_regional_id_from_match_id(match_id, cohort_name)
        return match_id, regional_id, connectome_key

    elif cohort_name == "HABS":
        m = re.match(r"^(H\d+_Y\d+|H\d+_y\d+)$", s, flags=re.IGNORECASE)
        if not m:
            return None, None, None
        connectome_key = re.sub(r"_Y", "_y", m.group(1))
        match_id = extract_4digit_match_id(connectome_key)
        regional_id = build_regional_id_from_match_id(match_id, cohort_name)
        return match_id, regional_id, connectome_key

    elif cohort_name == "ADRC":
        m = re.search(r"D(\d+)", s_up)
        if not m:
            return None, None, None
        match_id = m.group(1)[-4:].zfill(4)
        regional_id = build_regional_id_from_match_id(match_id, cohort_name)
        return match_id, regional_id, s_up

    elif cohort_name == "AD_DECODE":
        m = re.search(r"S(\d+)", s_up)
        if not m:
            return None, None, None
        match_id = m.group(1)[-4:].zfill(4)
        regional_id = build_regional_id_from_match_id(match_id, cohort_name)
        return match_id, regional_id, s_up

    return None, None, None


def load_region_metric_table(path, sep="\t"):
    lower = path.lower()
    if lower.endswith(".txt"):
        return pd.read_csv(path, sep=sep, low_memory=False)
    if lower.endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported regional stats format: {path}")


def summarize_external_fa_vol_metrics(cohort_name, fa_path, vol_path, fa_sep="\t"):
    df_fa = load_region_metric_table(fa_path, sep=fa_sep)
    df_vol = load_region_metric_table(vol_path, sep=",")

    # Remove ROI helper rows exactly as in graph builder logic
    if "ROI" in df_fa.columns:
        try:
            df_fa["ROI_numeric"] = pd.to_numeric(df_fa["ROI"], errors="coerce")
            if len(df_fa) > 0 and pd.isna(df_fa["ROI_numeric"].iloc[0]):
                df_fa = df_fa.iloc[1:].copy()
            df_fa = df_fa[df_fa["ROI"].astype(str) != "0"].copy().reset_index(drop=True)
            df_fa = df_fa.drop(columns=["ROI_numeric"], errors="ignore")
        except Exception:
            pass

    if "ROI" in df_vol.columns:
        df_vol = df_vol[df_vol["ROI"].astype(str) != "-1"].copy().reset_index(drop=True)

    fa_subject_cols = [c for c in df_fa.columns if str(c).strip() != "ROI"]
    vol_subject_cols = [c for c in df_vol.columns if str(c).strip() != "ROI"]

    common_subject_cols = sorted(set(fa_subject_cols).intersection(vol_subject_cols))

    rows = []
    for subj_col in common_subject_cols:
        match_id, regional_id, connectome_key = extract_connectome_identifiers_from_stats_col(
            subj_col, cohort_name
        )
        if match_id is None:
            continue

        fa_vals = pd.to_numeric(df_fa[subj_col], errors="coerce")
        vol_vals = pd.to_numeric(df_vol[subj_col], errors="coerce")

        rows.append({
            "stats_source_col": str(subj_col).strip(),
            "match_id": match_id,
            "regional_id": regional_id,
            "connectome_key": connectome_key,
            "FA_mean": fa_vals.mean(skipna=True),
            "FA_median": fa_vals.median(skipna=True),
            "Volume_mean": vol_vals.mean(skipna=True),
            "Volume_median": vol_vals.median(skipna=True),
        })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    # Standardized string keys
    for c in ["match_id", "regional_id", "connectome_key"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()

    return out


def save_bland_altman_plot(df, pred_col, out_png, title):
    ok, tmp = valid_scatter_xy(df, "Real_Age", pred_col)
    if not ok:
        return False, tmp

    true = tmp["Real_Age"].values
    pred = tmp[pred_col].values

    mean_vals = (true + pred) / 2.0
    diff_vals = pred - true

    mean_diff = np.mean(diff_vals)
    sd_diff = np.std(diff_vals, ddof=1) if len(diff_vals) > 1 else 0.0
    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.scatter(mean_vals, diff_vals, alpha=0.7, edgecolors="k")
    ax.axhline(mean_diff, linestyle="--")
    ax.axhline(loa_upper, linestyle=":")
    ax.axhline(loa_lower, linestyle=":")

    dist = compute_distribution_metrics(diff_vals)
    text = (
        f"n = {dist['n']}\n"
        f"mean diff = {mean_diff:.3f}\n"
        f"SD diff = {sd_diff:.3f}\n"
        f"LoA upper = {loa_upper:.3f}\n"
        f"LoA lower = {loa_lower:.3f}"
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax.set_xlabel("Mean of Real and Predicted Age")
    ax.set_ylabel("Predicted - Real")
    ax.set_title(title)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    return True, None


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


def save_residual_histogram(df, pred_col, out_png, title):
    ok, tmp = valid_scatter_xy(df, "Real_Age", pred_col)
    if not ok:
        return False, tmp

    residuals = tmp[pred_col].values - tmp["Real_Age"].values
    return save_histogram_with_stats(
        values=residuals,
        out_png=out_png,
        title=title,
        xlabel="Residual = Predicted - Real",
    )


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
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna()

    if len(tmp) == 0:
        return False, "no complete rows"

    grouped = []
    labels = []
    for grp, g in tmp.groupby(group_col):
        vals = g[value_col].values
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            grouped.append(vals)
            labels.append(str(grp))

    if len(grouped) < 1:
        return False, "no non-empty groups"

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.boxplot(grouped, tick_labels=labels, vert=True)
    add_distribution_box(ax, compute_distribution_metrics(tmp[value_col].values))
    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    return True, None


# =========================================================
# PATH DISCOVERY
# =========================================================
cv_preds_path = find_existing_file([
    os.path.join(results_dir, f"cv_predictions_{COHORT_FILE_STEM}.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_cv_oof_predictions.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_oof_predictions.csv"),
    os.path.join(results_dir, f"cv_predictions_{COHORT_FILE_STEM}.xlsx"),

    # fallback legacy names
    os.path.join(results_dir, f"cv_predictions_{COHORT_LOWER}.csv"),
    os.path.join(results_dir, f"{COHORT_LOWER}_cv_oof_predictions.csv"),
    os.path.join(results_dir, f"{COHORT_LOWER}_oof_predictions.csv"),
    os.path.join(results_dir, f"cv_predictions_{COHORT_LOWER}.xlsx"),
])

global_oof_path = find_existing_file([
    os.path.join(results_dir, f"cv_predictions_{COHORT_FILE_STEM}_with_global_oof.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_cv_predictions_with_global_oof.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_global_oof_predictions.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_global_oof_predictions.xlsx"),

    # fallback legacy names
    os.path.join(results_dir, f"cv_predictions_{COHORT_LOWER}_with_global_oof.csv"),
    os.path.join(results_dir, f"{COHORT_LOWER}_cv_predictions_with_global_oof.csv"),
    os.path.join(results_dir, f"{COHORT_LOWER}_global_oof_predictions.csv"),
    os.path.join(results_dir, f"{COHORT_LOWER}_global_oof_predictions.xlsx"),
])

metadata_path = find_existing_file([
    os.path.join(WORK, "ines/results/harmonized", COHORT_NAME, "graphs", f"{COHORT_LOWER}_metadata_aligned_raw.csv"),
    os.path.join(WORK, "ines/results/harmonized", COHORT_NAME, "graphs", f"{COHORT_LOWER}_metadata_aligned.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_aligned_raw.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_aligned.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_aligned.xlsx"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_with_cv_predictions.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_with_cv_predictions.xlsx"),

    # fallback legacy names
    os.path.join(results_dir, f"{COHORT_LOWER}_metadata_aligned_raw.csv"),
    os.path.join(results_dir, f"{COHORT_LOWER}_metadata_aligned.csv"),
    os.path.join(results_dir, f"{COHORT_LOWER}_metadata_aligned.xlsx"),
    os.path.join(results_dir, f"{COHORT_LOWER}_metadata_with_cv_predictions.csv"),
    os.path.join(results_dir, f"{COHORT_LOWER}_metadata_with_cv_predictions.xlsx"),
])

training_history_path = find_existing_file([
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_final_model_training_history.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_training_history.csv"),

    # fallback legacy names
    os.path.join(results_dir, f"{COHORT_LOWER}_final_model_training_history.csv"),
    os.path.join(results_dir, f"{COHORT_LOWER}_training_history.csv"),
])

full_cohort_preds_path = find_existing_file([
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_full_cohort_predictions.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_full_cohort_predictions.xlsx"),

    # fallback legacy names
    os.path.join(results_dir, f"{COHORT_LOWER}_full_cohort_predictions.csv"),
    os.path.join(results_dir, f"{COHORT_LOWER}_full_cohort_predictions.xlsx"),
])

metadata_all_path = find_existing_file([
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_all_with_predictions_plus_brainvol_hipp_fa.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_all_with_predictions_plus_brainvol_hipp_fa.xlsx"),

    os.path.join(WORK, "ines/results/harmonized", COHORT_NAME, "graphs", f"{COHORT_LOWER}_metadata_all_aligned_raw.csv"),
    os.path.join(WORK, "ines/results/harmonized", COHORT_NAME, "graphs", f"{COHORT_LOWER}_metadata_all_aligned.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_all_with_predictions.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_all_aligned_raw.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_all_aligned.csv"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_all_with_predictions.xlsx"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_metadata_all_aligned.xlsx"),

    # fallback legacy names
    os.path.join(results_dir, f"{COHORT_LOWER}_metadata_all_with_predictions_plus_brainvol_hipp_fa.csv"),
    os.path.join(results_dir, f"{COHORT_LOWER}_metadata_all_with_predictions_plus_brainvol_hipp_fa.xlsx"),
    os.path.join(results_dir, f"{COHORT_LOWER}_metadata_all_with_predictions.csv"),
    os.path.join(results_dir, f"{COHORT_LOWER}_metadata_all_aligned_raw.csv"),
    os.path.join(results_dir, f"{COHORT_LOWER}_metadata_all_aligned.csv"),
    os.path.join(results_dir, f"{COHORT_LOWER}_metadata_all_with_predictions.xlsx"),
    os.path.join(results_dir, f"{COHORT_LOWER}_metadata_all_aligned.xlsx"),
])
raw_model_pt = find_existing_file([
    os.path.join(results_dir, f"brainage_{COHORT_FILE_STEM}_prediction_model.pt"),

    # fallback legacy names
    os.path.join(results_dir, f"brainage_{COHORT_LOWER}_prediction_model.pt"),
])

bias_corrected_model_pt = find_existing_file([
    os.path.join(results_dir, f"brainage_{COHORT_FILE_STEM}_prediction_BIAS_CORRECTED_model.pt"),

    # fallback legacy names
    os.path.join(results_dir, f"brainage_{COHORT_LOWER}_prediction_BIAS_CORRECTED_model.pt"),
])

fold_metrics_path = find_existing_file([
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_cv_fold_metrics_bias_corrected.xlsx"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_cv_fold_metrics.xlsx"),
    os.path.join(results_dir, f"{COHORT_FILE_STEM}_cv_fold_metrics_bias_corrected.csv"),

    # fallback legacy names
    os.path.join(results_dir, f"{COHORT_LOWER}_cv_fold_metrics_bias_corrected.xlsx"),
    os.path.join(results_dir, f"{COHORT_LOWER}_cv_fold_metrics.xlsx"),
    os.path.join(results_dir, f"{COHORT_LOWER}_cv_fold_metrics_bias_corrected.csv"),
])
# =========================================================
# CLEAN OLD OUTPUTS ONCE
# =========================================================
clear_image_files(eval_outdir)
clear_image_files(val_outdir)
remove_duplicate_plot_subdirs(val_outdir)
remove_duplicate_plot_subdirs(eval_outdir)


# =========================================================
# LOAD REQUIRED FILES
# =========================================================
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

print("  full_cohort_predictions:", None if df_full_preds is None else df_full_preds.shape)
print("  metadata_all:", None if aligned_metadata_all is None else aligned_metadata_all.shape)


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
print("  prediction columns:", df_preds.columns.tolist())
if df_global is not None:
    print("  global_oof columns:", df_global.columns.tolist())


# =========================================================
# OPTIONAL TRAINING HISTORY PLOT
# =========================================================
if training_history_path is not None:
    try:
        hist_df = pd.read_csv(training_history_path)

        epoch_col = None
        loss_col = None

        for c in hist_df.columns:
            if c.lower() in ["epoch", "epochs"]:
                epoch_col = c
                break

        for c in hist_df.columns:
            if c.lower() in ["training_loss", "train_loss", "loss"]:
                loss_col = c
                break

        if epoch_col is not None and loss_col is not None:
            tmp = hist_df[[epoch_col, loss_col]].copy()
            tmp[epoch_col] = pd.to_numeric(tmp[epoch_col], errors="coerce")
            tmp[loss_col] = pd.to_numeric(tmp[loss_col], errors="coerce")
            tmp = tmp.dropna()

            if len(tmp) > 0:
                plt.figure(figsize=(10, 6))
                ax = plt.gca()
                ax.plot(tmp[epoch_col].values, tmp[loss_col].values)
                ax.set_title(f"{COHORT_NAME} Final Model Training Loss")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Training Loss")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(eval_outdir, f"{COHORT_LOWER}_final_model_training_loss.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

                tmp.to_csv(
                    os.path.join(eval_outdir, f"{COHORT_LOWER}_training_history_used.csv"),
                    index=False,
                )
    except Exception as e:
        print(f"Training history plot skipped: {e}")


# =========================================================
# EVALUATION
# =========================================================
eval_plot_log = []

try:
    eval_results = run_standard_evaluation(
        df_preds=df_preds,
        outdir=eval_outdir,
        train_losses=None,
        val_losses=None,
    )
    print("Evaluation utils done.")
    print("Evaluation outputs:", safe_dict_keys(eval_results))
except Exception as e:
    eval_results = {}
    print(f"run_standard_evaluation failed: {e}")
    eval_plot_log.append({
        "plot": "run_standard_evaluation",
        "status": "failed",
        "reason": str(e),
    })

if "Predicted_Age_RAW" in df_preds.columns:
    ok, reason = save_predicted_vs_real_plot(
        df=df_preds,
        y_col="Predicted_Age_RAW",
        out_png=os.path.join(eval_outdir, "predicted_vs_real_raw.png"),
        title=f"{COHORT_NAME}: Predicted vs Real Age (RAW OOF)",
        ylabel="Predicted Age (RAW)",
    )
    eval_plot_log.append({"plot": "predicted_vs_real_raw.png", "status": "saved" if ok else "skipped", "reason": reason})

    ok, reason = save_bland_altman_plot(
        df=df_preds,
        pred_col="Predicted_Age_RAW",
        out_png=os.path.join(eval_outdir, "bland_altman_raw.png"),
        title=f"{COHORT_NAME}: Bland-Altman (RAW OOF)",
    )
    eval_plot_log.append({"plot": "bland_altman_raw.png", "status": "saved" if ok else "skipped", "reason": reason})

    ok, reason = save_residual_histogram(
        df=df_preds,
        pred_col="Predicted_Age_RAW",
        out_png=os.path.join(eval_outdir, "residual_histogram_raw.png"),
        title=f"{COHORT_NAME}: Residual Histogram (RAW OOF)",
    )
    eval_plot_log.append({"plot": "residual_histogram_raw.png", "status": "saved" if ok else "skipped", "reason": reason})

if "Predicted_Age_BiasCorrected" in df_preds.columns:
    ok, reason = save_predicted_vs_real_plot(
        df=df_preds,
        y_col="Predicted_Age_BiasCorrected",
        out_png=os.path.join(eval_outdir, "predicted_vs_real_foldwise_bias_corrected.png"),
        title=f"{COHORT_NAME}: Predicted vs Real Age (Fold-wise Bias Corrected OOF)",
        ylabel="Predicted Age (Bias Corrected)",
    )
    eval_plot_log.append({"plot": "predicted_vs_real_foldwise_bias_corrected.png", "status": "saved" if ok else "skipped", "reason": reason})

    ok, reason = save_bland_altman_plot(
        df=df_preds,
        pred_col="Predicted_Age_BiasCorrected",
        out_png=os.path.join(eval_outdir, "bland_altman_foldwise_bias_corrected.png"),
        title=f"{COHORT_NAME}: Bland-Altman (Fold-wise Bias Corrected OOF)",
    )
    eval_plot_log.append({"plot": "bland_altman_foldwise_bias_corrected.png", "status": "saved" if ok else "skipped", "reason": reason})

    ok, reason = save_residual_histogram(
        df=df_preds,
        pred_col="Predicted_Age_BiasCorrected",
        out_png=os.path.join(eval_outdir, "residual_histogram_foldwise_bias_corrected.png"),
        title=f"{COHORT_NAME}: Residual Histogram (Fold-wise Bias Corrected OOF)",
    )
    eval_plot_log.append({"plot": "residual_histogram_foldwise_bias_corrected.png", "status": "saved" if ok else "skipped", "reason": reason})

if "BAG" in df_preds.columns:
    ok, reason = save_histogram_with_stats(
        values=df_preds["BAG"].values,
        out_png=os.path.join(eval_outdir, "bag_histogram.png"),
        title=f"{COHORT_NAME}: BAG distribution",
        xlabel="BAG",
    )
    eval_plot_log.append({"plot": "bag_histogram.png", "status": "saved" if ok else "skipped", "reason": reason})

    ok, reason = save_bag_vs_age_plot(
        df=df_preds,
        bag_col="BAG",
        out_png=os.path.join(eval_outdir, "bag_vs_age.png"),
        title=f"{COHORT_NAME}: BAG vs Age",
    )
    eval_plot_log.append({"plot": "bag_vs_age.png", "status": "saved" if ok else "skipped", "reason": reason})

if "cBAG" in df_preds.columns:
    ok, reason = save_histogram_with_stats(
        values=df_preds["cBAG"].values,
        out_png=os.path.join(eval_outdir, "cbag_histogram.png"),
        title=f"{COHORT_NAME}: cBAG distribution",
        xlabel="cBAG",
    )
    eval_plot_log.append({"plot": "cbag_histogram.png", "status": "saved" if ok else "skipped", "reason": reason})

    ok, reason = save_bag_vs_age_plot(
        df=df_preds,
        bag_col="cBAG",
        out_png=os.path.join(eval_outdir, "cbag_vs_age.png"),
        title=f"{COHORT_NAME}: cBAG vs Age",
    )
    eval_plot_log.append({"plot": "cbag_vs_age.png", "status": "saved" if ok else "skipped", "reason": reason})

df_subject_eval = None

if isinstance(eval_results, dict) and "df_subject_with_global_corr" in eval_results:
    df_subject_eval = eval_results["df_subject_with_global_corr"].copy()
    df_subject_eval = ensure_subject_id_col(df_subject_eval)
    df_subject_eval = normalize_prediction_columns(df_subject_eval)
elif df_global is not None:
    df_subject_eval = df_global.copy()
    df_subject_eval = ensure_subject_id_col(df_subject_eval)
    df_subject_eval = normalize_prediction_columns(df_subject_eval)

if df_subject_eval is not None:
    if "Predicted_Age_RAW" in df_subject_eval.columns and "Real_Age" in df_subject_eval.columns:
        ok, reason = save_predicted_vs_real_plot(
            df=df_subject_eval,
            y_col="Predicted_Age_RAW",
            out_png=os.path.join(eval_outdir, "subject_level_predicted_vs_real_raw.png"),
            title=f"{COHORT_NAME}: Subject-level Predicted vs Real Age (RAW mean)",
            ylabel="Predicted Age (RAW mean)",
        )
        eval_plot_log.append({"plot": "subject_level_predicted_vs_real_raw.png", "status": "saved" if ok else "skipped", "reason": reason})

    if "Predicted_Age_GlobalCorrected" in df_subject_eval.columns and "Real_Age" in df_subject_eval.columns:
        ok, reason = save_predicted_vs_real_plot(
            df=df_subject_eval,
            y_col="Predicted_Age_GlobalCorrected",
            out_png=os.path.join(eval_outdir, "subject_level_predicted_vs_real_global_corrected.png"),
            title=f"{COHORT_NAME}: Subject-level Predicted vs Real Age (Global OOF Corrected)",
            ylabel="Predicted Age (Global OOF Corrected)",
        )
        eval_plot_log.append({"plot": "subject_level_predicted_vs_real_global_corrected.png", "status": "saved" if ok else "skipped", "reason": reason})

        ok, reason = save_bland_altman_plot(
            df=df_subject_eval,
            pred_col="Predicted_Age_GlobalCorrected",
            out_png=os.path.join(eval_outdir, "subject_level_bland_altman_global_corrected.png"),
            title=f"{COHORT_NAME}: Subject-level Bland-Altman (Global Corrected)",
        )
        eval_plot_log.append({"plot": "subject_level_bland_altman_global_corrected.png", "status": "saved" if ok else "skipped", "reason": reason})

        ok, reason = save_residual_histogram(
            df=df_subject_eval,
            pred_col="Predicted_Age_GlobalCorrected",
            out_png=os.path.join(eval_outdir, "subject_level_residual_histogram_global_corrected.png"),
            title=f"{COHORT_NAME}: Subject-level Residual Histogram (Global Corrected)",
        )
        eval_plot_log.append({"plot": "subject_level_residual_histogram_global_corrected.png", "status": "saved" if ok else "skipped", "reason": reason})

    if "cBAG_global" in df_subject_eval.columns:
        ok, reason = save_histogram_with_stats(
            values=df_subject_eval["cBAG_global"].values,
            out_png=os.path.join(eval_outdir, "subject_level_cbag_global_histogram.png"),
            title=f"{COHORT_NAME}: Subject-level cBAG_global distribution",
            xlabel="cBAG_global",
        )
        eval_plot_log.append({"plot": "subject_level_cbag_global_histogram.png", "status": "saved" if ok else "skipped", "reason": reason})

        ok, reason = save_bag_vs_age_plot(
            df=df_subject_eval,
            bag_col="cBAG_global",
            out_png=os.path.join(eval_outdir, "subject_level_cbag_global_vs_age.png"),
            title=f"{COHORT_NAME}: Subject-level cBAG_global vs Age",
        )
        eval_plot_log.append({"plot": "subject_level_cbag_global_vs_age.png", "status": "saved" if ok else "skipped", "reason": reason})

pd.DataFrame(eval_plot_log).to_csv(
    os.path.join(eval_outdir, "image_generation_log.csv"),
    index=False,
)


# =========================================================
# VALIDATION
# =========================================================
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


# ---------------------------------------------------------
# Choose prediction dataframe for validation
# ---------------------------------------------------------
if df_full_preds is not None:
    print("Validation will use FULL-COHORT predictions.")
    df_for_validation = df_full_preds.copy()
elif df_subject_eval is not None:
    print("Validation will use subject-level evaluation dataframe.")
    df_for_validation = df_subject_eval.copy()
else:
    print("Global OOF / subject-level dataframe missing. Validation will use aggregated df_preds if possible.")
    df_for_validation = df_preds.copy()

df_for_validation = ensure_subject_id_col(df_for_validation)
df_for_validation["Subject_ID"] = normalize_id_series(df_for_validation["Subject_ID"])


# ---------------------------------------------------------
# Brain metric fallback for validation
# ---------------------------------------------------------
validation_brain_metric_col = BRAIN_METRIC_COL

if validation_brain_metric_col not in df_for_validation.columns:
    if "cBAG" in df_for_validation.columns:
        print("cBAG_global not found in validation dataframe. Using cBAG instead.")
        validation_brain_metric_col = "cBAG"
    elif "BAG" in df_for_validation.columns:
        print("cBAG_global not found in validation dataframe. Using BAG instead.")
        validation_brain_metric_col = "BAG"
    else:
        raise KeyError(
            f"No usable brain metric found for validation. "
            f"Tried '{BRAIN_METRIC_COL}', 'cBAG', and 'BAG'."
        )


# ---------------------------------------------------------
# Choose metadata source
# ---------------------------------------------------------
if aligned_metadata_all is not None:
    print("Validation metadata source: FULL-COHORT metadata.")
    aligned_metadata_tmp = aligned_metadata_all.copy()
else:
    print("Validation metadata source: standard aligned metadata.")
    aligned_metadata_tmp = aligned_metadata.copy()


for col in aligned_metadata_tmp.columns:
    if col in ["connectome_key", "match_id", "subject_id", "PTID", "ptid", "regional_id", "Subject_ID", "RID"]:
        aligned_metadata_tmp[col] = normalize_id_series(aligned_metadata_tmp[col])


# ---------------------------------------------------------
# Merge metadata into validation dataframe
# ---------------------------------------------------------
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

    # Remove pre-existing _meta columns from right side
    aligned_metadata_subject = aligned_metadata_subject[
        [c for c in aligned_metadata_subject.columns if not c.endswith("_meta")]
    ].copy()

    # Keep merge key + columns not already present in left df
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
    print("Warning: no reliable merge key found for validation. Metadata columns will not be merged.")

print("\nColumns with _meta:")
print([c for c in df_for_validation.columns if c.endswith("_meta")][:30])

print("Merged validation df rows:", len(df_for_validation))


# ---------------------------------------------------------
# Debug available columns
# ---------------------------------------------------------
check_cols = [
    validation_brain_metric_col,
    "MMSE_total",
    "MOCA_total_corrected",
    "MOCA_total",
    "ADAS_total",
    "CDGLOBAL",
    "CDRSB",
    "Clustering_Coeff",
    "Path_Length",
    "ABETA42",
    "ABETA40",
    "TAU",
    "PTAU",
    "PLASMA_PTAU217",
    "BMI",
    "pulse_pressure",
    "MAP",
    "amyloid_42",
    "amyloid_40",
    "tau_total",
    "ptau",
    "ptau217",
    "bmi",
    "group_status",
    "NORMCOG",
    "APOE",
    "genotype",
]

for c in check_cols:
    if c in df_for_validation.columns:
        print(f"{c}: non-NaN = {df_for_validation[c].notna().sum()}")
    else:
        print(f"{c}: MISSING")

for meta_name in ["ABETA42_meta", "ABETA40_meta", "TAU_meta", "PTAU_meta", "PLASMA_PTAU217_meta", "BMI_meta"]:
    if meta_name in df_for_validation.columns:
        print(f"{meta_name}: non-NaN =", df_for_validation[meta_name].notna().sum())


# ---------------------------------------------------------
# Convert columns where possible
# ---------------------------------------------------------
for c in df_for_validation.columns:
    if c != "Subject_ID":
        try:
            df_for_validation[c] = pd.to_numeric(df_for_validation[c], errors="ignore")
        except Exception:
            pass


# ---------------------------------------------------------
# Candidate variables
# ---------------------------------------------------------
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
    "FA_mean",
    "FA_median",
    "Volume_mean",
    "Volume_median",
    "ABETA42",
    "ABETA40",
    "TAU",
    "PTAU",
    "PLASMA_PTAU217",
    "GFAP",
    "NfL",
    "BMI",
    "VSBPSYS",
    "VSBPDIA",
    "VSPULSE",
    "BPSYS_AVG",
    "BPDIA_AVG",
    "amyloid_42",
    "amyloid_40",
    "tau_total",
    "ptau",
    "ptau217",
    "gfap",
    "nfl",
    "bmi",
    "bp_sys",
    "bp_dia",
    "pulse",
    "ATN_composite",
    "pulse_pressure",
    "MAP",
    "OM_BMI",
    "BW_Glucose_y",
    "BW_HBA1c_y",
    "BW_CholTotal_y",
    "BW_HDLChol_y",
    "BW_LDLchol_y",
    "BW_Triglycerides_y",
]

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

COGNITION_VARS = keep_existing_cols(df_for_validation, candidate_cognition_vars)
IMAGING_VARS = keep_existing_cols(df_for_validation, candidate_imaging_vars)

# Keep only variables that have at least 3 non-missing values
COGNITION_VARS = [
    c for c in COGNITION_VARS
    if c in df_for_validation.columns and pd.to_numeric(df_for_validation[c], errors="coerce").notna().sum() >= 3
]

IMAGING_VARS = [
    c for c in IMAGING_VARS
    if c in df_for_validation.columns and pd.to_numeric(df_for_validation[c], errors="coerce").notna().sum() >= 3
]

print("\nFINAL COGNITION_VARS:")
print(COGNITION_VARS)
print("\nFINAL IMAGING_VARS:")
print(IMAGING_VARS)

GROUP_COL = None
GROUP_COMPARISONS = None

for gc in candidate_group_cols:
    if gc in df_for_validation.columns:
        non_na_groups = df_for_validation[gc].dropna()
        if non_na_groups.nunique() >= 2:
            GROUP_COL = gc
            break

print("Selected GROUP_COL:", GROUP_COL)

if GROUP_COL is not None and GROUP_COL in ["Research Group", "group_status"]:
    present_groups = set(
        df_for_validation[GROUP_COL]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )

    candidate_comparisons = [("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")]
    valid_comparisons = [(a, b) for (a, b) in candidate_comparisons if a in present_groups and b in present_groups]

    if len(valid_comparisons) > 0:
        GROUP_COMPARISONS = valid_comparisons

print("GROUP_COMPARISONS:", GROUP_COMPARISONS)


# ---------------------------------------------------------
# Run validation utils
# ---------------------------------------------------------
validation_plot_log = []

try:
    val_results = run_standard_validation(
        df=df_for_validation,
        outdir=val_outdir,
        cognition_vars=COGNITION_VARS,
        imaging_vars=IMAGING_VARS,
        group_col=GROUP_COL,
        group_comparisons=GROUP_COMPARISONS,
        brain_metric_col=validation_brain_metric_col,
        method=CORR_METHOD,
    )
    print("Validation utils done.")
    print("Validation outputs:", safe_dict_keys(val_results))
except Exception as e:
    val_results = {}
    print(f"run_standard_validation failed: {e}")
    validation_plot_log.append({
        "plot": "run_standard_validation",
        "status": "failed",
        "reason": str(e),
    })

validation_subject_csv = os.path.join(val_outdir, "subject_level_validation_input.csv")
df_for_validation.to_csv(validation_subject_csv, index=False)


# ---------------------------------------------------------
# Histogram
# ---------------------------------------------------------
if validation_brain_metric_col in df_for_validation.columns:
    hist_name = f"{sanitize_filename(validation_brain_metric_col)}_histogram.png"
    ok, reason = save_histogram_with_stats(
        values=pd.to_numeric(df_for_validation[validation_brain_metric_col], errors="coerce").values,
        out_png=os.path.join(val_outdir, hist_name),
        title=f"{COHORT_NAME}: {validation_brain_metric_col} distribution",
        xlabel=validation_brain_metric_col,
    )
    validation_plot_log.append({
        "plot": hist_name,
        "status": "saved" if ok else "skipped",
        "reason": reason,
    })


# ---------------------------------------------------------
# Correlation scatters
# IMPORTANT: use save_correlation_scatter, not save_scatter_with_corr
# # ---------------------------------------------------------
# for var in COGNITION_VARS + IMAGING_VARS:
#     ok0, reason0 = valid_scatter_xy(df_for_validation, var, validation_brain_metric_col)
#     fname = f"{sanitize_filename(validation_brain_metric_col)}_vs_{sanitize_filename(var)}.png"

#     if ok0:
#         ok, reason = save_correlation_scatter(
#             df=df_for_validation,
#             x_col=var,
#             y_col=validation_brain_metric_col,
#             out_png=os.path.join(val_outdir, fname),
#             method=CORR_METHOD,
#             title=f"{COHORT_NAME}: {validation_brain_metric_col} vs {var}",
#         )
#     else:
#         ok, reason = False, reason0

#     validation_plot_log.append({
#         "plot": fname,
#         "status": "saved" if ok else "skipped",
#         "reason": reason,
#     })
# =========================================================
# EXTRA: cBAG vs NEW BRAIN METRICS (hippocampus, FA, volume)
# =========================================================

EXTRA_VARS = [
    "Hippocampus_Total_pct",
    "Left_Hippocampus_pct",
    "Right_Hippocampus_pct",
    "Hippocampus_FA_Mean",
    "Hippocampus_FA_Total",
    "Left_Hippocampus_FA",
    "Right_Hippocampus_FA",
    "Total_Brain_volume",
]

EXTRA_VARS = [v for v in EXTRA_VARS if v in df_for_validation.columns]



for var in EXTRA_VARS:

    ok0, reason0 = valid_scatter_xy(df_for_validation, var, validation_brain_metric_col)

    fname = f"{sanitize_filename(validation_brain_metric_col)}_vs_{sanitize_filename(var)}.png"

    if ok0:
        ok, reason = save_correlation_scatter(
            df=df_for_validation,
            x_col=var,
            y_col=validation_brain_metric_col,
            out_png=os.path.join(val_outdir, fname),
            method=CORR_METHOD,
            title=f"{COHORT_NAME}: {validation_brain_metric_col} vs {var}",
        )
    else:
        ok, reason = False, reason0

    validation_plot_log.append({
        "plot": fname,
        "status": "saved" if ok else "skipped",
        "reason": reason,
    })
# =========================================================
# EXTRA: cBAG vs NEW BRAIN METRICS (hippocampus, FA, volume)
# =========================================================

EXTRA_VARS = [
    "Hippocampus_Total_pct",
    "Left_Hippocampus_pct",
    "Right_Hippocampus_pct",
    "Hippocampus_FA_Mean",
    "Hippocampus_FA_Total",
    "Left_Hippocampus_FA",
    "Right_Hippocampus_FA",
    "Total_Brain_volume",
]

EXTRA_VARS = [v for v in EXTRA_VARS if v in df_for_validation.columns]

print("\nEXTRA BRAIN VARIABLES FOUND:")
print(EXTRA_VARS)

for var in EXTRA_VARS:

    ok0, reason0 = valid_scatter_xy(df_for_validation, var, validation_brain_metric_col)

    fname = f"{sanitize_filename(validation_brain_metric_col)}_vs_{sanitize_filename(var)}_EXTRA.png"

    if ok0:
        ok, reason = save_correlation_scatter(
            df=df_for_validation,
            x_col=var,
            y_col=validation_brain_metric_col,
            out_png=os.path.join(val_outdir, fname),
            method=CORR_METHOD,
            title=f"{COHORT_NAME}: {validation_brain_metric_col} vs {var}",
        )
    else:
        ok, reason = False, reason0

    validation_plot_log.append({
        "plot": fname,
        "status": "saved" if ok else "skipped",
        "reason": reason,
    })
# ---------------------------------------------------------
# ROC / AUC plots
# ---------------------------------------------------------
roc_plot_log = []

score_col = validation_brain_metric_col
print("\n===== ROC DEBUG =====")
print("score_col:", score_col)
print("APOE col detected:", find_apoe_carrier_column(df_for_validation))
print("Cognition col detected:", find_cognition_status_column(df_for_validation))

if score_col not in df_for_validation.columns:
    print(f"ROC skipped: score column '{score_col}' not present.")
else:
    # 1) APOE4 carriage ROC
    apoe_col = find_apoe_carrier_column(df_for_validation)

    if apoe_col is not None:
        y_apoe = derive_apoe4_carrier(df_for_validation[apoe_col])

        print("APOE derived counts:")
        print(pd.Series(y_apoe).value_counts(dropna=False))

        score_series = pd.to_numeric(df_for_validation[score_col], errors="coerce")
        score_used, flipped = maybe_flip_auc_direction(y_apoe, score_series)

        out_name = f"roc_{sanitize_filename(score_col)}_vs_apoe4_carriage.png"
        ok, info = save_roc_curve_from_score(
            y_true=y_apoe,
            y_score=score_used,
            out_png=os.path.join(val_outdir, out_name),
            title="ROC Curve: APOE4 carriage",
            score_name=score_col + (" (flipped)" if flipped else ""),
        )

        roc_plot_log.append({
            "plot": out_name,
            "status": "saved" if ok else "skipped",
            "target": "APOE4_carriage",
            "source_column": apoe_col,
            "score_column": score_col,
            "flipped_score": flipped,
            "reason": None if ok else info,
        })
    else:
        roc_plot_log.append({
            "plot": f"roc_{sanitize_filename(score_col)}_vs_apoe4_carriage.png",
            "status": "skipped",
            "target": "APOE4_carriage",
            "source_column": None,
            "score_column": score_col,
            "flipped_score": False,
            "reason": "No APOE/APOE4/genotype column found",
        })

    # 2) Cognitive status ROC
    y_cog, cog_col, cog_desc = derive_binary_cognitive_status(df_for_validation)

    if y_cog is not None and cog_col is not None:
        print("Cognitive derived counts:")
        print(pd.Series(y_cog).value_counts(dropna=False))

        score_series = pd.to_numeric(df_for_validation[score_col], errors="coerce")
        score_used, flipped = maybe_flip_auc_direction(y_cog, score_series)

        out_name = f"roc_{sanitize_filename(score_col)}_vs_cognitive_status.png"
        ok, info = save_roc_curve_from_score(
            y_true=y_cog,
            y_score=score_used,
            out_png=os.path.join(val_outdir, out_name),
            title=f"ROC Curve: {cog_desc}",
            score_name=score_col + (" (flipped)" if flipped else ""),
        )

        roc_plot_log.append({
            "plot": out_name,
            "status": "saved" if ok else "skipped",
            "target": "cognitive_status",
            "source_column": cog_col,
            "score_column": score_col,
            "flipped_score": flipped,
            "reason": None if ok else info,
        })
    else:
        roc_plot_log.append({
            "plot": f"roc_{sanitize_filename(score_col)}_vs_cognitive_status.png",
            "status": "skipped",
            "target": "cognitive_status",
            "source_column": None,
            "score_column": score_col,
            "flipped_score": False,
            "reason": "No cognitive status column could be binarized",
        })


# ---------------------------------------------------------
# Group boxplot
# ---------------------------------------------------------
if GROUP_COL is not None:
    fname = f"{sanitize_filename(validation_brain_metric_col)}_by_{sanitize_filename(GROUP_COL)}.png"
    ok, reason = save_boxplot_with_stats(
        df=df_for_validation,
        group_col=GROUP_COL,
        value_col=validation_brain_metric_col,
        out_png=os.path.join(val_outdir, fname),
        title=f"{COHORT_NAME}: {validation_brain_metric_col} by {GROUP_COL}",
    )
    validation_plot_log.append({
        "plot": fname,
        "status": "saved" if ok else "skipped",
        "reason": reason,
    })


# ---------------------------------------------------------
# Save logs
# ---------------------------------------------------------
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


# =========================================================
# SAVE SUMMARY OF DETECTED INPUTS
# =========================================================
summary_df = pd.DataFrame([
    {"key": "COHORT_NAME", "value": COHORT_NAME},
    {"key": "results_dir", "value": results_dir},
    {"key": "cv_preds_path", "value": cv_preds_path},
    {"key": "global_oof_path", "value": global_oof_path},
    {"key": "full_cohort_preds_path", "value": full_cohort_preds_path},
    {"key": "metadata_path", "value": metadata_path},
    {"key": "metadata_all_path", "value": metadata_all_path},
    {"key": "training_history_path", "value": training_history_path},
    {"key": "raw_model_pt", "value": raw_model_pt},
    {"key": "bias_corrected_model_pt", "value": bias_corrected_model_pt},
    {"key": "fold_metrics_path", "value": fold_metrics_path},
    {"key": "n_predictions_rows", "value": len(df_preds)},
    {"key": "n_metadata_rows", "value": len(aligned_metadata)},
    {"key": "n_global_rows", "value": None if df_global is None else len(df_global)},
    {"key": "n_full_predictions_rows", "value": None if df_full_preds is None else len(df_full_preds)},
    {"key": "n_metadata_all_rows", "value": None if aligned_metadata_all is None else len(aligned_metadata_all)},
    {"key": "validation_brain_metric_col", "value": validation_brain_metric_col},
])
summary_df.to_csv(
    os.path.join(results_dir, f"{COHORT_LOWER}_eval_validation_pipeline_inputs_summary.csv"),
    index=False,
)

print("\nEvaluation + validation pipeline finished.")
print("Check these logs if something was skipped:")
print("  ", os.path.join(eval_outdir, "image_generation_log.csv"))
print("  ", os.path.join(val_outdir, "image_generation_log.csv"))
print("  ", os.path.join(val_outdir, "roc_image_generation_log.csv"))
print("  ", os.path.join(val_outdir, "all_image_generation_log.csv"))
print("  ", os.path.join(results_dir, f"{COHORT_LOWER}_eval_validation_pipeline_inputs_summary.csv"))