#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Brain-age training with cross-validation, within-fold bias correction,
Excel/CSV result exports, learning curves, and final model saving.

What this script does
---------------------
1) loads healthy-control graphs and aligned metadata for one cohort
2) runs K-fold CV
3) computes raw CV metrics fold by fold
4) computes within-fold bias-corrected CV metrics fold by fold
5) saves fold-level metrics to Excel/CSV
6) saves out-of-fold (OOF) subject-level predictions with BAG and cBAG
7) appends predictions/cBAG back to metadata in results
8) saves train/validation learning curves for each fold and their summaries
9) fits a global bias correction from OOF predictions
10) trains a final model on all healthy controls
11) saves final raw and bias-corrected checkpoints

Notes
-----
- This script performs internal validation only.
- "testing" curves here correspond to validation curves during CV.
- For a true external test cohort, evaluate the saved final model separately.
"""

import os
import json
import random
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

warnings.filterwarnings("ignore")


# =========================
# USER CONFIG
# =========================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set your cohort here: ADNI, ADRC, AD_DECODE, HABS
COHORT = "ADNI"

# Cohort-specific paths.
# Update these only if your folder structure changes.
COHORT_CONFIG = {
    "ADNI": {
        "graph_path": "/mnt/newStor/paros/paros_WORK/ines/results/harmonized/ADNI/graphs/graph_data_list_adni.pt",
        "metadata_path": "/mnt/newStor/paros/paros_WORK/ines/results/harmonized/ADNI/graphs/adni_metadata_aligned.csv",
        "encoding_info_path": "/mnt/newStor/paros/paros_WORK/ines/results/harmonized/ADNI/graphs/adni_feature_encoding_info.json",
        "out_dir": "/mnt/newStor/paros/paros_WORK/ines/results/BrainAgePredictionADNI",
        "prefix": "adni",
    },
    "ADRC": {
        "graph_path": "/mnt/newStor/paros/paros_WORK/ines/results/harmonized/ADRC/graphs/graph_data_list_adrc.pt",
        "metadata_path": "/mnt/newStor/paros/paros_WORK/ines/results/harmonized/ADRC/graphs/adrc_metadata_aligned.csv",
        "encoding_info_path": "/mnt/newStor/paros/paros_WORK/ines/results/harmonized/ADRC/graphs/adrc_feature_encoding_info.json",
        "out_dir": "/mnt/newStor/paros/paros_WORK/ines/results/BrainAgePredictionADRC",
        "prefix": "adrc",
    },
    "AD_DECODE": {
        "graph_path": "/mnt/newStor/paros/paros_WORK/ines/results/harmonized/AD_DECODE/graphs/graph_data_list_ad_decode.pt",
        "metadata_path": "/mnt/newStor/paros/paros_WORK/ines/results/harmonized/AD_DECODE/graphs/ad_decode_metadata_aligned.csv",
        "encoding_info_path": "/mnt/newStor/paros/paros_WORK/ines/results/harmonized/AD_DECODE/graphs/ad_decode_feature_encoding_info.json",
        "out_dir": "/mnt/newStor/paros/paros_WORK/ines/results/BrainAgePredictionADDECODE",
        "prefix": "addecode",
    },
    "HABS": {
        "graph_path": "/mnt/newStor/paros/paros_WORK/ines/results/harmonized/HABS/graphs/graph_data_list_habs.pt",
        "metadata_path": "/mnt/newStor/paros/paros_WORK/ines/results/harmonized/HABS/graphs/habs_metadata_aligned.csv",
        "encoding_info_path": "/mnt/newStor/paros/paros_WORK/ines/results/harmonized/HABS/graphs/habs_feature_encoding_info.json",
        "out_dir": "/mnt/newStor/paros/paros_WORK/ines/results/BrainAgePredictionHABS",
        "prefix": "habs",
    },
}

if COHORT not in COHORT_CONFIG:
    raise ValueError(f"Unknown COHORT='{COHORT}'. Valid options: {list(COHORT_CONFIG.keys())}")

GRAPH_PATH = COHORT_CONFIG[COHORT]["graph_path"]
METADATA_PATH = COHORT_CONFIG[COHORT]["metadata_path"]
ENCODING_INFO_PATH = COHORT_CONFIG[COHORT]["encoding_info_path"]
OUT_DIR = COHORT_CONFIG[COHORT]["out_dir"]
PREFIX = COHORT_CONFIG[COHORT]["prefix"]
os.makedirs(OUT_DIR, exist_ok=True)
GRAPH_DIR = os.path.dirname(GRAPH_PATH)
METADATA_DIR = os.path.dirname(METADATA_PATH)

graph_filename = os.path.basename(GRAPH_PATH)
metadata_filename = os.path.basename(METADATA_PATH)

graph_stem = os.path.splitext(graph_filename)[0]
metadata_stem = os.path.splitext(metadata_filename)[0]

# healthy -> all
# graph_data_list_ad_decode.pt -> graph_data_list_ad_decode_all.pt
if graph_stem.endswith("_all"):
    GRAPH_PATH_ALL = GRAPH_PATH
else:
    GRAPH_PATH_ALL = os.path.join(GRAPH_DIR, f"{graph_stem}_all.pt")

# ad_decode_metadata_aligned.csv -> ad_decode_metadata_all_aligned.csv
if metadata_stem.endswith("_all_aligned"):
    METADATA_PATH_ALL = METADATA_PATH
elif metadata_stem.endswith("_aligned"):
    METADATA_PATH_ALL = os.path.join(
        METADATA_DIR,
        metadata_stem.replace("_aligned", "_all_aligned") + ".csv"
    )
else:
    METADATA_PATH_ALL = os.path.join(METADATA_DIR, f"{metadata_stem}_all_aligned.csv")

# ad_decode_metadata_aligned.csv -> ad_decode_metadata_all_aligned_raw.csv
if metadata_stem.endswith("_all_aligned_raw"):
    METADATA_PATH_ALL_RAW = METADATA_PATH
elif metadata_stem.endswith("_aligned"):
    METADATA_PATH_ALL_RAW = os.path.join(
        METADATA_DIR,
        metadata_stem.replace("_aligned", "_all_aligned_raw") + ".csv"
    )
else:
    METADATA_PATH_ALL_RAW = os.path.join(METADATA_DIR, f"{metadata_stem}_all_aligned_raw.csv")
    
print("\n=== FULL COHORT PATH DEBUG ===")
print("GRAPH_PATH_ALL:", GRAPH_PATH_ALL)
print("Exists:", os.path.exists(GRAPH_PATH_ALL))
print("METADATA_PATH_ALL:", METADATA_PATH_ALL)
print("Exists:", os.path.exists(METADATA_PATH_ALL))
print("METADATA_PATH_ALL_RAW:", METADATA_PATH_ALL_RAW)
print("Exists:", os.path.exists(METADATA_PATH_ALL_RAW))
# Training hyperparameters
N_SPLITS = 5
BATCH_SIZE = 16
EPOCHS = 250
LR = 5e-4
WEIGHT_DECAY = 5e-4
HIDDEN_DIM = 64
DROPOUT = 0.35
PATIENCE = 20
NUM_WORKERS = 0

# Output paths
MODEL_RAW_PATH = os.path.join(OUT_DIR, f"brainage_{PREFIX}_prediction_model.pt")
MODEL_BC_PATH = os.path.join(OUT_DIR, f"brainage_{PREFIX}_prediction_BIAS_CORRECTED_model.pt")
OOF_CSV_PATH = os.path.join(OUT_DIR, f"{PREFIX}_cv_oof_predictions.csv")
OOF_XLSX_PATH = os.path.join(OUT_DIR, f"{PREFIX}_cv_oof_predictions.xlsx")
CV_FOLD_RAW_CSV = os.path.join(OUT_DIR, f"{PREFIX}_cv_fold_metrics_raw.csv")
CV_FOLD_RAW_XLSX = os.path.join(OUT_DIR, f"{PREFIX}_cv_fold_metrics_raw.xlsx")
CV_FOLD_BC_CSV = os.path.join(OUT_DIR, f"{PREFIX}_cv_fold_metrics_bias_corrected.csv")
CV_FOLD_BC_XLSX = os.path.join(OUT_DIR, f"{PREFIX}_cv_fold_metrics_bias_corrected.xlsx")
CV_SUMMARY_CSV = os.path.join(OUT_DIR, f"{PREFIX}_cv_summary_metrics.csv")
CV_SUMMARY_XLSX = os.path.join(OUT_DIR, f"{PREFIX}_cv_summary_metrics.xlsx")
FINAL_MODEL_SUMMARY_CSV = os.path.join(OUT_DIR, f"{PREFIX}_final_model_summary.csv")
FINAL_MODEL_SUMMARY_XLSX = os.path.join(OUT_DIR, f"{PREFIX}_final_model_summary.xlsx")
METADATA_RESULTS_CSV = os.path.join(OUT_DIR, f"{PREFIX}_metadata_with_cv_predictions.csv")
METADATA_RESULTS_XLSX = os.path.join(OUT_DIR, f"{PREFIX}_metadata_with_cv_predictions.xlsx")
RESIDUAL_AGE_DEP_CSV = os.path.join(OUT_DIR, f"{PREFIX}_residual_age_dependence.csv")
RESIDUAL_AGE_DEP_XLSX = os.path.join(OUT_DIR, f"{PREFIX}_residual_age_dependence.xlsx")
LEARNING_CURVE_DIR = os.path.join(OUT_DIR, "learning_curves")
FOLD_CURVE_XLSX = os.path.join(LEARNING_CURVE_DIR, f"{PREFIX}_all_fold_learning_histories.xlsx")
CURVE_SUMMARY_XLSX = os.path.join(LEARNING_CURVE_DIR, f"{PREFIX}_learning_curve_summaries.xlsx")
FULL_COHORT_PRED_CSV = os.path.join(OUT_DIR, f"{PREFIX}_full_cohort_predictions.csv")
FULL_COHORT_PRED_XLSX = os.path.join(OUT_DIR, f"{PREFIX}_full_cohort_predictions.xlsx")

FULL_COHORT_METADATA_RESULTS_CSV = os.path.join(OUT_DIR, f"{PREFIX}_metadata_all_with_predictions.csv")
FULL_COHORT_METADATA_RESULTS_XLSX = os.path.join(OUT_DIR, f"{PREFIX}_metadata_all_with_predictions.xlsx")

FULL_COHORT_SUMMARY_CSV = os.path.join(OUT_DIR, f"{PREFIX}_full_cohort_prediction_summary.csv")
FULL_COHORT_SUMMARY_XLSX = os.path.join(OUT_DIR, f"{PREFIX}_full_cohort_prediction_summary.xlsx")

# =========================
# REPRODUCIBILITY
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(SEED)


# =========================
# HELPERS
# =========================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def safe_pearsonr(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return np.nan
    return pearsonr(y_true, y_pred)[0]


def safe_polyfit(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if len(x) < 2 or np.std(x) == 0:
        return np.nan, np.nan
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def compute_metrics(y_true, y_pred, label=""):
    out = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "r": safe_pearsonr(y_true, y_pred),
    }
    if label:
        print(f"\n=== {label} ===")
    else:
        print("\n=== Metrics ===")
    print(f"MAE : {out['MAE']:.4f}")
    print(f"RMSE: {out['RMSE']:.4f}")
    print(f"R2  : {out['R2']:.4f}")
    print(f"r   : {out['r']:.4f}" if not np.isnan(out["r"]) else "r   : nan")
    return out


def fit_bias_correction(y_true_train, y_pred_train):
    """
    Fit linear correction:
        pred = a * age + b
    corrected_pred = pred - (a*age + b - age)
    """
    x = np.asarray(y_true_train).reshape(-1)
    y = np.asarray(y_pred_train).reshape(-1)

    if len(x) < 2 or np.std(x) == 0:
        return 1.0, 0.0

    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def apply_bias_correction(y_true, y_pred, a, b):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    bias = a * y_true + b - y_true
    return y_pred - bias


def get_global_feature_tensor(data):
    candidate_keys = [
        "global_features",
        "global_feats",
        "graph_features",
        "graph_feats",
        "u",
        "globals",
    ]

    for key in candidate_keys:
        if hasattr(data, key):
            val = getattr(data, key)
            if val is None:
                continue
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float)
            val = val.float()
            if val.dim() == 1:
                val = val.unsqueeze(0)
            return val

    return torch.zeros((1, 0), dtype=torch.float)


def get_target_from_graph_or_metadata(data, metadata_df):
    if hasattr(data, "y") and data.y is not None:
        y = data.y
        if torch.is_tensor(y):
            y = y.view(-1).float()
            if len(y) > 0:
                return float(y[0].item())
        else:
            return float(y)

    candidate_id_fields = ["match_id", "subject_id", "ptid", "PTID", "connectome_key", "regional_id"]
    age_candidates = ["age", "Age", "AGE", "brain_age_target"]

    for field in candidate_id_fields:
        if hasattr(data, field):
            graph_id = getattr(data, field)
            if torch.is_tensor(graph_id):
                if graph_id.numel() == 1:
                    graph_id = graph_id.item()
                else:
                    continue

            if field in metadata_df.columns:
                row = metadata_df.loc[metadata_df[field] == graph_id]
                if len(row) == 1:
                    for age_col in age_candidates:
                        if age_col in metadata_df.columns:
                            return float(row.iloc[0][age_col])

    raise ValueError("Could not recover target age from graph.y or metadata.")


def get_graph_identifier(data, idx):
    candidate_keys = ["match_id", "subject_id", "PTID", "ptid", "connectome_key", "regional_id"]
    for key in candidate_keys:
        if hasattr(data, key):
            value = getattr(data, key)
            if torch.is_tensor(value):
                if value.numel() == 1:
                    value = value.item()
                else:
                    continue
            return value
    return f"graph_{idx}"


def prepare_graphs(graph_list, metadata_df):
    processed = []

    for i, data in enumerate(graph_list):
        d = deepcopy(data)

        age_value = get_target_from_graph_or_metadata(d, metadata_df)
        d.y = torch.tensor([age_value], dtype=torch.float)

        gf = get_global_feature_tensor(d)
        d.global_features = gf.float()

        if not hasattr(d, "edge_attr") or d.edge_attr is None:
            num_edges = d.edge_index.shape[1]
            d.edge_attr = torch.ones((num_edges, 1), dtype=torch.float)
        else:
            if d.edge_attr.dim() == 1:
                d.edge_attr = d.edge_attr.unsqueeze(-1)
            d.edge_attr = d.edge_attr.float()

        d.x = d.x.float()
        processed.append(d)

    return processed


def summarize_learning_histories(all_histories, metric_col):
    hist_df = pd.concat(all_histories, ignore_index=True)

    summary = (
        hist_df.groupby("epoch")[metric_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n"})
    )

    summary["std"] = summary["std"].fillna(0.0)
    summary["sem"] = summary["std"] / np.sqrt(summary["n"])
    summary["ci95"] = 1.96 * summary["sem"]

    return hist_df, summary


def plot_learning_curve_with_ci(summary_df, metric_col, ylabel, title, out_path):
    plt.figure(figsize=(8, 5))

    x = summary_df["epoch"].values
    y = summary_df["mean"].values
    ci = summary_df["ci95"].values

    plt.plot(x, y, label=f"Mean {metric_col}")
    plt.fill_between(x, y - ci, y + ci, alpha=0.25, label="95% CI")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {out_path}")


def save_learning_curve_summaries(all_histories, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)

    all_hist_df = pd.concat(all_histories, ignore_index=True)
    all_hist_csv = os.path.join(out_dir, f"{prefix}_all_fold_learning_histories.csv")
    all_hist_df.to_csv(all_hist_csv, index=False)

    _, train_summary = summarize_learning_histories(all_histories, "train_loss")
    _, val_mae_summary = summarize_learning_histories(all_histories, "val_mae")
    _, val_rmse_summary = summarize_learning_histories(all_histories, "val_rmse")

    train_summary_path = os.path.join(out_dir, f"{prefix}_learning_curve_train_loss_summary.csv")
    val_mae_summary_path = os.path.join(out_dir, f"{prefix}_learning_curve_val_mae_summary.csv")
    val_rmse_summary_path = os.path.join(out_dir, f"{prefix}_learning_curve_val_rmse_summary.csv")

    train_summary.to_csv(train_summary_path, index=False)
    val_mae_summary.to_csv(val_mae_summary_path, index=False)
    val_rmse_summary.to_csv(val_rmse_summary_path, index=False)

    plot_learning_curve_with_ci(
        summary_df=train_summary,
        metric_col="train_loss",
        ylabel="Training Loss",
        title=f"{prefix.upper()} Training Loss Across Folds",
        out_path=os.path.join(out_dir, f"{prefix}_learning_curve_train_loss_ci.png"),
    )
    plot_learning_curve_with_ci(
        summary_df=val_mae_summary,
        metric_col="val_mae",
        ylabel="Validation MAE",
        title=f"{prefix.upper()} Validation MAE Across Folds",
        out_path=os.path.join(out_dir, f"{prefix}_learning_curve_val_mae_ci.png"),
    )
    plot_learning_curve_with_ci(
        summary_df=val_rmse_summary,
        metric_col="val_rmse",
        ylabel="Validation RMSE",
        title=f"{prefix.upper()} Validation RMSE Across Folds",
        out_path=os.path.join(out_dir, f"{prefix}_learning_curve_val_rmse_ci.png"),
    )

    with pd.ExcelWriter(CURVE_SUMMARY_XLSX, engine="openpyxl") as writer:
        all_hist_df.to_excel(writer, sheet_name="all_fold_histories", index=False)
        train_summary.to_excel(writer, sheet_name="train_loss_summary", index=False)
        val_mae_summary.to_excel(writer, sheet_name="val_mae_summary", index=False)
        val_rmse_summary.to_excel(writer, sheet_name="val_rmse_summary", index=False)

    print(f"Saved learning curve tables to: {CURVE_SUMMARY_XLSX}")
    return all_hist_df, train_summary, val_mae_summary, val_rmse_summary


def find_best_metadata_merge_key(metadata_df, graph_ids):
    candidate_cols = ["match_id", "subject_id", "PTID", "ptid", "connectome_key", "regional_id"]
    graph_id_set = set(map(str, graph_ids))

    best_col = None
    best_matches = -1
    for col in candidate_cols:
        if col in metadata_df.columns:
            meta_vals = set(metadata_df[col].astype(str).tolist())
            overlap = len(graph_id_set.intersection(meta_vals))
            if overlap > best_matches:
                best_matches = overlap
                best_col = col
    return best_col, best_matches


# =========================
# MODEL
# =========================
class GNNBrainAge(nn.Module):
    def __init__(self, node_feat_dim, global_feat_dim, hidden_dim=64, dropout=0.2, edge_dim=1):
        super().__init__()

        nn1 = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.conv1 = GINEConv(nn1, edge_dim=edge_dim)
        self.conv2 = GINEConv(nn2, edge_dim=edge_dim)
        self.conv3 = GINEConv(nn3, edge_dim=edge_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        fusion_in = hidden_dim + global_feat_dim

        self.regressor = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)

        gnn_emb = global_mean_pool(x, batch)

        if hasattr(data, "global_features") and data.global_features is not None:
            gf = data.global_features.float()
            if gf.dim() == 1:
                gf = gf.unsqueeze(0)
        else:
            gf = torch.zeros((gnn_emb.shape[0], 0), device=gnn_emb.device)

        fused = torch.cat([gnn_emb, gf], dim=1)
        out = self.regressor(fused).squeeze(-1)
        return out


# =========================
# TRAIN / EVAL
# =========================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)
        target = batch.y.view(-1).float()

        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        bs = target.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    trues = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).detach().cpu().numpy()
        true = batch.y.view(-1).detach().cpu().numpy()

        preds.extend(pred.tolist())
        trues.extend(true.tolist())

    return np.array(trues), np.array(preds)

@torch.no_grad()
def predict_with_graph_ids(model, graph_list, device):
    loader = DataLoader(graph_list, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model.eval()
    preds = []
    trues = []
    graph_ids = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).detach().cpu().numpy()
        true = batch.y.view(-1).detach().cpu().numpy()

        preds.extend(pred.tolist())
        trues.extend(true.tolist())

    for i, g in enumerate(graph_list):
        graph_ids.append(str(get_graph_identifier(g, i)))

    return np.array(trues), np.array(preds), graph_ids

def apply_final_model_to_full_cohort(
    model,
    metadata_all_df,
    graphs_all,
    a_global,
    b_global,
    out_dir,
    prefix,
    cohort_name,
):
    print(f"\n=== Applying final model to ALL {cohort_name} subjects ===")

    graphs_all = prepare_graphs(graphs_all, metadata_all_df)

    y_all, p_all_raw, graph_ids_all = predict_with_graph_ids(model, graphs_all, DEVICE)
    p_all_bc = apply_bias_correction(y_all, p_all_raw, a_global, b_global)

    full_pred_df = pd.DataFrame({
        "graph_id": graph_ids_all,
        "age_true": y_all,
        "pred_raw": p_all_raw,
        "pred_bias_corrected": p_all_bc,
    })
    full_pred_df["BAG_raw"] = full_pred_df["pred_raw"] - full_pred_df["age_true"]
    full_pred_df["cBAG"] = full_pred_df["pred_bias_corrected"] - full_pred_df["age_true"]

    full_pred_df.to_csv(FULL_COHORT_PRED_CSV, index=False)
    full_pred_df.to_excel(FULL_COHORT_PRED_XLSX, index=False)

    print(f"Saved full-cohort predictions CSV: {FULL_COHORT_PRED_CSV}")
    print(f"Saved full-cohort predictions XLSX: {FULL_COHORT_PRED_XLSX}")

    merge_key, overlap = find_best_metadata_merge_key(metadata_all_df, full_pred_df["graph_id"].tolist())
    print(f"Best full-cohort metadata merge key: {merge_key} (overlap={overlap})")

    if merge_key is not None and overlap > 0:
        metadata_all_pred_df = metadata_all_df.copy()
        metadata_all_pred_df[merge_key] = metadata_all_pred_df[merge_key].astype(str)

        tmp_pred = full_pred_df.rename(columns={"graph_id": merge_key})
        tmp_pred = tmp_pred[[merge_key, "age_true", "pred_raw", "pred_bias_corrected", "BAG_raw", "cBAG"]]
        tmp_pred[merge_key] = tmp_pred[merge_key].astype(str)

        metadata_all_with_preds = metadata_all_pred_df.merge(tmp_pred, on=merge_key, how="left")
    else:
        metadata_all_with_preds = metadata_all_df.copy()
        print("Warning: could not confidently merge full-cohort predictions back into metadata.")

    metadata_all_with_preds.to_csv(FULL_COHORT_METADATA_RESULTS_CSV, index=False)
    metadata_all_with_preds.to_excel(FULL_COHORT_METADATA_RESULTS_XLSX, index=False)

    print(f"Saved full-cohort metadata with predictions CSV: {FULL_COHORT_METADATA_RESULTS_CSV}")
    print(f"Saved full-cohort metadata with predictions XLSX: {FULL_COHORT_METADATA_RESULTS_XLSX}")

    full_summary_df = pd.DataFrame([
        {
            "cohort": cohort_name,
            "n_subjects_full_cohort": len(graphs_all),
            "mean_age": float(np.mean(y_all)),
            "std_age": float(np.std(y_all)),
            "mean_pred_raw": float(np.mean(p_all_raw)),
            "std_pred_raw": float(np.std(p_all_raw)),
            "mean_pred_bias_corrected": float(np.mean(p_all_bc)),
            "std_pred_bias_corrected": float(np.std(p_all_bc)),
            "global_bias_a": float(a_global),
            "global_bias_b": float(b_global),
            "graph_path_all": GRAPH_PATH_ALL,
            "metadata_path_all": METADATA_PATH_ALL,
        }
    ])
    full_summary_df.to_csv(FULL_COHORT_SUMMARY_CSV, index=False)
    full_summary_df.to_excel(FULL_COHORT_SUMMARY_XLSX, index=False)

    print(f"Saved full-cohort summary CSV: {FULL_COHORT_SUMMARY_CSV}")
    print(f"Saved full-cohort summary XLSX: {FULL_COHORT_SUMMARY_XLSX}")

    return full_pred_df, metadata_all_with_preds

def fit_model(train_graphs, val_graphs, node_feat_dim, global_feat_dim, device, fold_id=None, history_dir=None):
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = GNNBrainAge(
        node_feat_dim=node_feat_dim,
        global_feat_dim=global_feat_dim,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        edge_dim=train_graphs[0].edge_attr.shape[1],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_state = None
    best_val_mae = np.inf
    wait = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        y_val, p_val = predict(model, val_loader, device)
        val_mae = mean_absolute_error(y_val, p_val)
        val_rmse = rmse(y_val, p_val)

        history.append({
            "fold": fold_id,
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_mae": float(val_mae),
            "val_rmse": float(val_rmse),
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_mae={val_mae:.4f} | val_rmse={val_rmse:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    history_df = pd.DataFrame(history)

    if history_dir is not None and fold_id is not None:
        os.makedirs(history_dir, exist_ok=True)
        history_path = os.path.join(history_dir, f"fold_{fold_id}_learning_history.csv")
        history_df.to_csv(history_path, index=False)
        print(f"Saved fold history: {history_path}")

    return model, history_df


def train_final_model(all_graphs, node_feat_dim, global_feat_dim, device):
    loader = DataLoader(all_graphs, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model = GNNBrainAge(
        node_feat_dim=node_feat_dim,
        global_feat_dim=global_feat_dim,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        edge_dim=all_graphs[0].edge_attr.shape[1],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_state = None
    best_loss = np.inf
    wait = 0
    final_history = []

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, loader, optimizer, device)
        final_history.append({"epoch": epoch, "train_loss": float(loss)})

        if epoch % 10 == 0 or epoch == 1:
            print(f"Final model epoch {epoch:03d} | loss={loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= PATIENCE:
            print(f"Final model early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    final_history_df = pd.DataFrame(final_history)
    final_history_csv = os.path.join(LEARNING_CURVE_DIR, f"{PREFIX}_final_model_training_history.csv")
    os.makedirs(LEARNING_CURVE_DIR, exist_ok=True)
    final_history_df.to_csv(final_history_csv, index=False)
    print(f"Saved final-model training history: {final_history_csv}")

    plt.figure(figsize=(8, 5))
    plt.plot(final_history_df["epoch"], final_history_df["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"{PREFIX.upper()} Final Model Training Loss")
    plt.tight_layout()
    final_plot = os.path.join(LEARNING_CURVE_DIR, f"{PREFIX}_final_model_training_loss.png")
    plt.savefig(final_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {final_plot}")

    return model, final_history_df


# =========================
# MAIN
# =========================
def main():
    print(f"Using device: {DEVICE}")
    print(f"Cohort: {COHORT}")

    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError(f"Missing graph file: {GRAPH_PATH}")
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")

    print("\n=== Loading metadata ===")
    metadata_df = pd.read_csv(METADATA_PATH)
    print(f"Loaded metadata rows: {len(metadata_df)}")
    print(f"Metadata columns: {list(metadata_df.columns)}")

    if os.path.exists(ENCODING_INFO_PATH):
        with open(ENCODING_INFO_PATH, "r") as f:
            encoding_info = json.load(f)
        print("\nEncoding info found:")
        print(json.dumps(encoding_info, indent=2))
    else:
        print("\nEncoding info JSON not found.")
        encoding_info = None

    print("\n=== Loading graphs ===")
    graphs = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    print(f"Loaded {len(graphs)} graphs")

    graphs = prepare_graphs(graphs, metadata_df)

    node_feat_dim = graphs[0].x.shape[1]
    global_feat_dim = graphs[0].global_features.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]

    print("\n=== Detected input dimensions ===")
    print(f"NODE_FEAT_DIM   = {node_feat_dim}")
    print(f"GLOBAL_FEAT_DIM = {global_feat_dim}")
    print(f"EDGE_DIM        = {edge_dim}")

    ages = np.array([float(g.y.item()) for g in graphs])
    graph_ids = [str(get_graph_identifier(g, i)) for i, g in enumerate(graphs)]

    print("\n=== Training set summary ===")
    print(f"N subjects: {len(graphs)}")
    print(f"Age mean  : {ages.mean():.3f}")
    print(f"Age std   : {ages.std():.3f}")
    print(f"Age min   : {ages.min():.3f}")
    print(f"Age max   : {ages.max():.3f}")

    print("\n=== Running cross-validation ===")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_true = np.zeros(len(graphs), dtype=float)
    oof_pred_raw = np.zeros(len(graphs), dtype=float)
    oof_pred_bc = np.zeros(len(graphs), dtype=float)

    all_fold_histories = []
    fold_raw_rows = []
    fold_bc_rows = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(graphs), start=1):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold}/{N_SPLITS}")
        print(f"{'=' * 60}")

        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]

        train_ages = np.array([float(g.y.item()) for g in train_graphs])
        val_ages = np.array([float(g.y.item()) for g in val_graphs])

        model, fold_history_df = fit_model(
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            node_feat_dim=node_feat_dim,
            global_feat_dim=global_feat_dim,
            device=DEVICE,
            fold_id=fold,
            history_dir=LEARNING_CURVE_DIR,
        )
        all_fold_histories.append(fold_history_df)

        train_loader_eval = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        val_loader_eval = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        y_train, p_train = predict(model, train_loader_eval, DEVICE)
        y_val, p_val = predict(model, val_loader_eval, DEVICE)

        a_fold, b_fold = fit_bias_correction(y_train, p_train)
        p_val_bc = apply_bias_correction(y_val, p_val, a_fold, b_fold)

        oof_true[val_idx] = y_val
        oof_pred_raw[val_idx] = p_val
        oof_pred_bc[val_idx] = p_val_bc

        raw_metrics = compute_metrics(y_val, p_val, label=f"Fold {fold} RAW")
        bc_metrics = compute_metrics(y_val, p_val_bc, label=f"Fold {fold} BIAS-CORRECTED")

        bag_raw = p_val - y_val
        bag_bc = p_val_bc - y_val
        bag_raw_r = safe_pearsonr(y_val, bag_raw)
        bag_bc_r = safe_pearsonr(y_val, bag_bc)
        bag_raw_slope, bag_raw_intercept = safe_polyfit(y_val, bag_raw)
        bag_bc_slope, bag_bc_intercept = safe_polyfit(y_val, bag_bc)

        common_info = {
            "fold": fold,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "train_age_mean": float(train_ages.mean()),
            "train_age_std": float(train_ages.std()),
            "val_age_mean": float(val_ages.mean()),
            "val_age_std": float(val_ages.std()),
            "bias_a": float(a_fold),
            "bias_b": float(b_fold),
        }

        fold_raw_rows.append({
            **common_info,
            **raw_metrics,
            "BAG_age_r": bag_raw_r,
            "BAG_age_slope": bag_raw_slope,
            "BAG_age_intercept": bag_raw_intercept,
        })
        fold_bc_rows.append({
            **common_info,
            **bc_metrics,
            "cBAG_age_r": bag_bc_r,
            "cBAG_age_slope": bag_bc_slope,
            "cBAG_age_intercept": bag_bc_intercept,
        })

    print("\n=== Saving learning curves with 95% CI ===")
    all_hist_df, train_summary, val_mae_summary, val_rmse_summary = save_learning_curve_summaries(
        all_fold_histories, LEARNING_CURVE_DIR, PREFIX
    )

    print("\n" + "#" * 70)
    print("FINAL CROSS-VALIDATED METRICS")
    print("#" * 70)

    raw_metrics_oof = compute_metrics(oof_true, oof_pred_raw, label="OOF RAW")
    bc_metrics_oof = compute_metrics(oof_true, oof_pred_bc, label="OOF BIAS-CORRECTED")

    oof_df = pd.DataFrame({
        "graph_id": graph_ids,
        "age_true": oof_true,
        "pred_raw": oof_pred_raw,
        "pred_bias_corrected": oof_pred_bc,
    })
    oof_df["BAG_raw"] = oof_df["pred_raw"] - oof_df["age_true"]
    oof_df["cBAG"] = oof_df["pred_bias_corrected"] - oof_df["age_true"]
    oof_df.to_csv(OOF_CSV_PATH, index=False)
    oof_df.to_excel(OOF_XLSX_PATH, index=False)
    print(f"\nSaved OOF predictions to: {OOF_CSV_PATH}")
    print(f"Saved OOF predictions to: {OOF_XLSX_PATH}")

    a_global, b_global = fit_bias_correction(oof_true, oof_pred_raw)
    print("\n=== Global bias correction fitted from OOF ===")
    print(f"a = {a_global:.6f}")
    print(f"b = {b_global:.6f}")

    fold_raw_df = pd.DataFrame(fold_raw_rows)
    fold_bc_df = pd.DataFrame(fold_bc_rows)
    fold_raw_df.to_csv(CV_FOLD_RAW_CSV, index=False)
    fold_bc_df.to_csv(CV_FOLD_BC_CSV, index=False)
    fold_raw_df.to_excel(CV_FOLD_RAW_XLSX, index=False)
    fold_bc_df.to_excel(CV_FOLD_BC_XLSX, index=False)
    print(f"Saved fold raw metrics: {CV_FOLD_RAW_XLSX}")
    print(f"Saved fold bias-corrected metrics: {CV_FOLD_BC_XLSX}")

    residual_age_df = pd.DataFrame([
        {
            "metric_set": "OOF_RAW",
            "bag_name": "BAG_raw",
            "age_bag_r": safe_pearsonr(oof_df["age_true"], oof_df["BAG_raw"]),
            "age_bag_slope": safe_polyfit(oof_df["age_true"], oof_df["BAG_raw"])[0],
            "age_bag_intercept": safe_polyfit(oof_df["age_true"], oof_df["BAG_raw"])[1],
        },
        {
            "metric_set": "OOF_BIAS_CORRECTED",
            "bag_name": "cBAG",
            "age_bag_r": safe_pearsonr(oof_df["age_true"], oof_df["cBAG"]),
            "age_bag_slope": safe_polyfit(oof_df["age_true"], oof_df["cBAG"])[0],
            "age_bag_intercept": safe_polyfit(oof_df["age_true"], oof_df["cBAG"])[1],
        },
    ])
    residual_age_df.to_csv(RESIDUAL_AGE_DEP_CSV, index=False)
    residual_age_df.to_excel(RESIDUAL_AGE_DEP_XLSX, index=False)

    cv_summary_df = pd.DataFrame([
        {"evaluation": "OOF_RAW", **raw_metrics_oof},
        {"evaluation": "OOF_BIAS_CORRECTED", **bc_metrics_oof},
    ])
    cv_summary_df["global_bias_a"] = [np.nan, a_global]
    cv_summary_df["global_bias_b"] = [np.nan, b_global]
    cv_summary_df.to_csv(CV_SUMMARY_CSV, index=False)
    cv_summary_df.to_excel(CV_SUMMARY_XLSX, index=False)

    merge_key, overlap = find_best_metadata_merge_key(metadata_df, oof_df["graph_id"].tolist())
    print(f"\nBest metadata merge key: {merge_key} (overlap={overlap})")

    if merge_key is not None and overlap > 0:
        metadata_pred_df = metadata_df.copy()
        metadata_pred_df[merge_key] = metadata_pred_df[merge_key].astype(str)
        tmp_oof = oof_df.rename(columns={"graph_id": merge_key})
        tmp_oof = tmp_oof[[merge_key, "age_true", "pred_raw", "pred_bias_corrected", "BAG_raw", "cBAG"]]
        tmp_oof[merge_key] = tmp_oof[merge_key].astype(str)
        metadata_with_preds = metadata_pred_df.merge(tmp_oof, on=merge_key, how="left")

        # If this file is regenerated in later runs, these prediction columns are replaced
        # by the newest ones because they come from the freshly created OOF table.
        # Saved columns include raw BAG and bias-corrected BAG (cBAG).
    else:
        metadata_with_preds = metadata_df.copy()
        print("Warning: could not confidently merge OOF predictions back into metadata.")

    # Overwrite the previous results file on each run so the metadata always contains
    # the most recent predictions/BAG/cBAG instead of appending duplicate columns.
    metadata_with_preds.to_csv(METADATA_RESULTS_CSV, index=False)
    metadata_with_preds.to_excel(METADATA_RESULTS_XLSX, index=False)
    print(f"Saved metadata with CV predictions: {METADATA_RESULTS_XLSX}")

    print(f"\n=== Training final model on all {COHORT} healthy-control graphs ===")
    final_model, final_history_df = train_final_model(
        all_graphs=graphs,
        node_feat_dim=node_feat_dim,
        global_feat_dim=global_feat_dim,
        device=DEVICE,
    )

    final_model_summary_df = pd.DataFrame([
        {
            "cohort": COHORT,
            "n_subjects": len(graphs),
            "node_feat_dim": node_feat_dim,
            "global_feat_dim": global_feat_dim,
            "edge_dim": edge_dim,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "epochs_max": EPOCHS,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "cv_raw_MAE": raw_metrics_oof["MAE"],
            "cv_raw_RMSE": raw_metrics_oof["RMSE"],
            "cv_raw_R2": raw_metrics_oof["R2"],
            "cv_raw_r": raw_metrics_oof["r"],
            "cv_bc_MAE": bc_metrics_oof["MAE"],
            "cv_bc_RMSE": bc_metrics_oof["RMSE"],
            "cv_bc_R2": bc_metrics_oof["R2"],
            "cv_bc_r": bc_metrics_oof["r"],
            "global_bias_a": a_global,
            "global_bias_b": b_global,
            "graph_path": GRAPH_PATH,
            "metadata_path": METADATA_PATH,
            "encoding_info_path": ENCODING_INFO_PATH if os.path.exists(ENCODING_INFO_PATH) else "",
            "final_model_path_raw": MODEL_RAW_PATH,
            "final_model_path_bias_corrected": MODEL_BC_PATH,
        }
    ])
    final_model_summary_df.to_csv(FINAL_MODEL_SUMMARY_CSV, index=False)
    final_model_summary_df.to_excel(FINAL_MODEL_SUMMARY_XLSX, index=False)

    raw_ckpt = {
        "cohort": COHORT,
        "model_state_dict": final_model.state_dict(),
        "node_feat_dim": node_feat_dim,
        "global_feat_dim": global_feat_dim,
        "edge_dim": edge_dim,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "seed": SEED,
        "cv_raw_metrics": raw_metrics_oof,
        "cv_bias_corrected_metrics": bc_metrics_oof,
        "feature_paths": {
            "graph_path": GRAPH_PATH,
            "metadata_path": METADATA_PATH,
            "encoding_info_path": ENCODING_INFO_PATH,
        },
        "training_config": {
            "n_splits": N_SPLITS,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "patience": PATIENCE,
        },
    }
    torch.save(raw_ckpt, MODEL_RAW_PATH)
    print(f"\nFinal {COHORT} model (RAW) saved as: {MODEL_RAW_PATH}")

    bc_ckpt = {
        "cohort": COHORT,
        "model_state_dict": final_model.state_dict(),
        "node_feat_dim": node_feat_dim,
        "global_feat_dim": global_feat_dim,
        "edge_dim": edge_dim,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "seed": SEED,
        "bias_correction": {
            "a": float(a_global),
            "b": float(b_global),
        },
        "cv_raw_metrics": raw_metrics_oof,
        "cv_bias_corrected_metrics": bc_metrics_oof,
        "feature_paths": {
            "graph_path": GRAPH_PATH,
            "metadata_path": METADATA_PATH,
            "encoding_info_path": ENCODING_INFO_PATH,
        },
        "training_config": {
            "n_splits": N_SPLITS,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "patience": PATIENCE,
        },
    }
    torch.save(bc_ckpt, MODEL_BC_PATH)
    print(f"Final {COHORT} model (BIAS-CORRECTED) saved as: {MODEL_BC_PATH}")

    print("\n=== Loading full-cohort graphs/metadata for inference ===")
    if not os.path.exists(GRAPH_PATH_ALL):
        print(f"Full-cohort graph file not found, skipping: {GRAPH_PATH_ALL}")
    elif not os.path.exists(METADATA_PATH_ALL):
        print(f"Full-cohort metadata file not found, skipping: {METADATA_PATH_ALL}")
    else:
        metadata_all_df = pd.read_csv(METADATA_PATH_ALL)
        print(f"Loaded full-cohort metadata rows: {len(metadata_all_df)}")

        graphs_all = torch.load(GRAPH_PATH_ALL, map_location="cpu", weights_only=False)
        print(f"Loaded full-cohort graphs: {len(graphs_all)}")

        full_pred_df, metadata_all_with_preds = apply_final_model_to_full_cohort(
            model=final_model,
            metadata_all_df=metadata_all_df,
            graphs_all=graphs_all,
            a_global=a_global,
            b_global=b_global,
            out_dir=OUT_DIR,
            prefix=PREFIX,
            cohort_name=COHORT,
        )

    with pd.ExcelWriter(os.path.join(OUT_DIR, f"{PREFIX}_master_results.xlsx"), engine="openpyxl") as writer:
        fold_raw_df.to_excel(writer, sheet_name="cv_fold_raw", index=False)
        fold_bc_df.to_excel(writer, sheet_name="cv_fold_bias_corrected", index=False)
        cv_summary_df.to_excel(writer, sheet_name="cv_summary", index=False)
        oof_df.to_excel(writer, sheet_name="oof_predictions", index=False)
        residual_age_df.to_excel(writer, sheet_name="residual_age_dependence", index=False)
        final_model_summary_df.to_excel(writer, sheet_name="final_model_summary", index=False)

    print("\nDone.")
    print(f"Results saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
