#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import root_mean_squared_error


# ==================================================
# 1) Paths (cluster-safe)
# ==================================================
WORK_DIR = os.environ.get("WORK")
if WORK_DIR is None:
    raise EnvironmentError("Environment variable WORK is not set.")
WORK_DIR = WORK_DIR.rstrip("/")
print("Using WORK directory:", WORK_DIR)

BASE = os.path.join(WORK_DIR, "ines", "code")
GRAPH_PT = os.path.join(BASE, "graph_data_list_addecode.pt")
MODEL_PATTERN = os.path.join(BASE, "model_fold_{fold}_rep_{rep}.pt")
FINAL_MODEL_PATH = os.path.join(BASE, "model_trained_on_all_healthy.pt")

OUT_DIR = os.path.join(WORK_DIR, "ines", "results", "addecode_training_eval_plots_save")
os.makedirs(OUT_DIR, exist_ok=True)

print("BASE:", BASE)
print("GRAPH_PT:", GRAPH_PT)
print("MODEL_PATTERN:", MODEL_PATTERN)
print("FINAL_MODEL_PATH:", FINAL_MODEL_PATH)
print("OUT_DIR:", OUT_DIR)


# ==================================================
# 2) Load graphs
# ==================================================
if not os.path.exists(GRAPH_PT):
    raise FileNotFoundError(f"Missing graph file: {GRAPH_PT}")

graph_data_list = torch.load(GRAPH_PT, weights_only=False)
print("Loaded graphs:", len(graph_data_list))


# ==================================================
# 3) Device
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ==================================================
# 4) Model definition (MUST MATCH training exactly)
# ==================================================
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm

class BrainAgeGATv2(nn.Module):
    def __init__(self, global_feat_dim):
        super(BrainAgeGATv2, self).__init__()

        self.node_embed = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        self.gnn1 = GATv2Conv(64, 16, heads=8, concat=True, edge_dim=1)
        self.bn1 = BatchNorm(128)

        self.gnn2 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn2 = BatchNorm(128)

        self.gnn3 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn3 = BatchNorm(128)

        self.gnn4 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn4 = BatchNorm(128)

        self.dropout = nn.Dropout(0.25)

        self.meta_head = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        self.graph_head = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        self.pca_head = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x = self.node_embed(x)

        x = self.gnn1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        x_res1 = x
        x = self.gnn2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.relu(x + x_res1)

        x_res2 = x
        x = self.gnn3(x, edge_index, edge_attr=edge_attr)
        x = self.bn3(x)
        x = F.relu(x + x_res2)

        x_res3 = x
        x = self.gnn4(x, edge_index, edge_attr=edge_attr)
        x = self.bn4(x)
        x = F.relu(x + x_res3)

        x = self.dropout(x)
        x = global_mean_pool(x, data.batch)

        global_feats = data.global_features.to(x.device).squeeze(1)

        meta_embed = self.meta_head(global_feats[:, 0:4])
        graph_embed = self.graph_head(global_feats[:, 4:6])
        pca_embed = self.pca_head(global_feats[:, 6:])

        global_embed = torch.cat([meta_embed, graph_embed, pca_embed], dim=1)
        x = torch.cat([x, global_embed], dim=1)

        return self.fc(x)


# ==================================================
# 5) CV evaluation from saved checkpoints
# ==================================================
N_FOLDS = 7
REPEATS = 10
BATCH_SIZE = 6

ages = np.array([d.y.item() for d in graph_data_list])
age_bins = pd.qcut(ages, q=5, labels=False)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

all_rows = []
fold_summaries = []

# Also collect global arrays for plots
ALL_TRUE = []
ALL_PRED = []

for fold_idx, (_, test_idx) in enumerate(skf.split(graph_data_list, age_bins), start=1):
    test_data = [graph_data_list[i] for i in test_idx]
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    rep_mae, rep_rmse, rep_r2 = [], [], []

    for rep in range(1, REPEATS + 1):
        ckpt_path = MODEL_PATTERN.format(fold=fold_idx, rep=rep)
        if not os.path.exists(ckpt_path):
            print("Missing:", ckpt_path)
            continue

        model = BrainAgeGATv2(global_feat_dim=16).to(device)
        state_dict = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        y_true, y_pred, subj_ids = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch).view(-1)
                true = batch.y.view(-1)

                y_pred.extend(pred.detach().cpu().numpy().tolist())
                y_true.extend(true.detach().cpu().numpy().tolist())

                # subject_id handling
                if hasattr(batch, "subject_id"):
                    try:
                        subj_ids.extend([str(x) for x in batch.subject_id])
                    except:
                        subj_ids.extend(["NA"] * len(true))
                else:
                    subj_ids.extend(["NA"] * len(true))

        # metrics per rep
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        rep_mae.append(mae)
        rep_rmse.append(rmse)
        rep_r2.append(r2)

        # accumulate for global plots
        ALL_TRUE.extend(y_true)
        ALL_PRED.extend(y_pred)

        # save per-subject rows
        for sid, t, p in zip(subj_ids, y_true, y_pred):
            all_rows.append({
                "Fold": fold_idx,
                "Repeat": rep,
                "Subject_ID": sid,
                "Real_Age": t,
                "Predicted_Age": p
            })

    fold_summaries.append({
        "Fold": fold_idx,
        "MAE_mean": float(np.mean(rep_mae)) if rep_mae else np.nan,
        "MAE_std": float(np.std(rep_mae)) if rep_mae else np.nan,
        "RMSE_mean": float(np.mean(rep_rmse)) if rep_rmse else np.nan,
        "RMSE_std": float(np.std(rep_rmse)) if rep_rmse else np.nan,
        "R2_mean": float(np.mean(rep_r2)) if rep_r2 else np.nan,
        "R2_std": float(np.std(rep_r2)) if rep_r2 else np.nan,
        "n_models_found": int(len(rep_mae))
    })


# ==================================================
# 6) Save CSVs
# ==================================================
df_preds = pd.DataFrame(all_rows)
pred_csv = os.path.join(OUT_DIR, "cv_predictions_from_saved_models.csv")
df_preds.to_csv(pred_csv, index=False)
print("Saved predictions CSV:", pred_csv)

df_folds = pd.DataFrame(fold_summaries)
fold_csv = os.path.join(OUT_DIR, "cv_fold_summary_from_saved_models.csv")
df_folds.to_csv(fold_csv, index=False)
print("Saved fold summary CSV:", fold_csv)

print("\nFold summary:")
print(df_folds)

print("\nGLOBAL (fold means):")
print("MAE:", df_folds["MAE_mean"].mean(), "±", df_folds["MAE_mean"].std())
print("RMSE:", df_folds["RMSE_mean"].mean(), "±", df_folds["RMSE_mean"].std())
print("R2:", df_folds["R2_mean"].mean(), "±", df_folds["R2_mean"].std())


# ==================================================
# 7) Save figures
# ==================================================
ALL_TRUE = np.array(ALL_TRUE, dtype=float)
ALL_PRED = np.array(ALL_PRED, dtype=float)

# --- Predicted vs Real ---
plt.figure(figsize=(7, 6))
plt.scatter(ALL_TRUE, ALL_PRED, alpha=0.6, edgecolors="k", linewidths=0.2)
mn = min(ALL_TRUE.min(), ALL_PRED.min())
mx = max(ALL_TRUE.max(), ALL_PRED.max())
plt.plot([mn, mx], [mn, mx], linestyle="--")
plt.xlabel("Real Age")
plt.ylabel("Predicted Age")
plt.title("Predicted vs Real (all folds × repeats)")

global_mae = mean_absolute_error(ALL_TRUE, ALL_PRED)
global_rmse = root_mean_squared_error(ALL_TRUE, ALL_PRED)
global_r2 = r2_score(ALL_TRUE, ALL_PRED)

txt = f"MAE={global_mae:.2f}\nRMSE={global_rmse:.2f}\nR²={global_r2:.2f}"
plt.text(0.02, 0.98, txt, transform=plt.gca().transAxes,
         va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

fig1 = os.path.join(OUT_DIR, "predicted_vs_real.png")
plt.savefig(fig1, dpi=300, bbox_inches="tight")
plt.close()
print("Saved figure:", fig1)

# --- Brain Age Gap histogram ---
gap = ALL_PRED - ALL_TRUE
plt.figure(figsize=(7, 5))
plt.hist(gap, bins=20, edgecolor="k", alpha=0.8)
plt.xlabel("Brain Age Gap (Predicted - Real)")
plt.ylabel("Count")
plt.title("Brain Age Gap Distribution")
fig2 = os.path.join(OUT_DIR, "brain_age_gap_hist.png")
plt.savefig(fig2, dpi=300, bbox_inches="tight")
plt.close()
print("Saved figure:", fig2)

# --- Residuals vs age ---
plt.figure(figsize=(7, 5))
plt.scatter(ALL_TRUE, gap, alpha=0.6, edgecolors="k", linewidths=0.2)
plt.axhline(0, linestyle="--")
plt.xlabel("Real Age")
plt.ylabel("Residual (Predicted - Real)")
plt.title("Residuals vs Age")
fig3 = os.path.join(OUT_DIR, "residuals_vs_age.png")
plt.savefig(fig3, dpi=300, bbox_inches="tight")
plt.close()
print("Saved figure:", fig3)

# ==================================================
# 8) Fold-level performance plots (mean ± std across repeats)
# ==================================================
import numpy as np
import matplotlib.pyplot as plt
import os

# Drop folds where no models were found (just in case)
df_plot = df_folds.dropna(subset=["MAE_mean", "R2_mean"]).copy()

x = df_plot["Fold"].values

# --- MAE per fold ---
plt.figure(figsize=(9, 4))
plt.errorbar(
    x,
    df_plot["MAE_mean"].values,
    yerr=df_plot["MAE_std"].values,
    fmt="o-",
    capsize=4
)
plt.xlabel("Fold")
plt.ylabel("MAE (years)")
plt.title("MAE per fold (mean ± std across repeats)")
plt.grid(True)
plt.tight_layout()

fig4 = os.path.join(OUT_DIR, "mae_per_fold.png")
plt.savefig(fig4, dpi=300, bbox_inches="tight")
plt.close()
print("Saved figure:", fig4)

# --- R² per fold ---
plt.figure(figsize=(9, 4))
plt.errorbar(
    x,
    df_plot["R2_mean"].values,
    yerr=df_plot["R2_std"].values,
    fmt="o-",
    capsize=4
)
plt.xlabel("Fold")
plt.ylabel("R²")
plt.title("R² per fold (mean ± std across repeats)")
plt.grid(True)
plt.tight_layout()

fig5 = os.path.join(OUT_DIR, "r2_per_fold.png")
plt.savefig(fig5, dpi=300, bbox_inches="tight")
plt.close()
print("Saved figure:", fig5)

# --- (Optional but recommended) RMSE per fold ---
plt.figure(figsize=(9, 4))
plt.errorbar(
    x,
    df_plot["RMSE_mean"].values,
    yerr=df_plot["RMSE_std"].values,
    fmt="o-",
    capsize=4
)
plt.xlabel("Fold")
plt.ylabel("RMSE (years)")
plt.title("RMSE per fold (mean ± std across repeats)")
plt.grid(True)
plt.tight_layout()

fig6 = os.path.join(OUT_DIR, "rmse_per_fold.png")
plt.savefig(fig6, dpi=300, bbox_inches="tight")
plt.close()
print("Saved figure:", fig6)

